"""Calibrate a frozen VaaniRakshak V2 detector on development data only.

The command fits one temperature scalar to the checkpoint's raw detector scores.
It never reads split=test. The existing development-selected operating threshold
is transformed through the same monotonic temperature function, preserving every
binary decision at that operating point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vaanirakshak.v2_calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
    transform_threshold,
)
from vaanirakshak.v2_data_contract import audit_manifest, load_jsonl, manifest_fingerprint
from vaanirakshak.v2_detectors import CALIBRATION_METHOD, CheckpointDetector, V2_CHECKPOINT_SCHEMA
from vaanirakshak.v2_train_ssl import ManifestAudioDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def calibrate(
    manifest: Path,
    checkpoint: Path,
    output_checkpoint: Path,
    *,
    device: str = "cpu",
) -> dict:
    manifest = manifest.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_checkpoint = output_checkpoint.expanduser().resolve()
    if checkpoint == output_checkpoint:
        raise ValueError("Calibration must write a new checkpoint; refusing to overwrite the source checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    records = load_jsonl(manifest)
    audit = audit_manifest(records)
    audit.raise_for_errors()
    fingerprint = manifest_fingerprint(records)
    dev_records = [record for record in records if record.split == "dev"]
    if not dev_records:
        raise ValueError("Calibration requires development records")
    if {record.label for record in dev_records} != {"bonafide", "spoof"}:
        raise ValueError("Calibration development split must contain bonafide and spoof records")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema") != V2_CHECKPOINT_SCHEMA:
        raise ValueError("Calibration requires a V2 detector checkpoint")
    if bool(state.get("calibrated_probability", False)):
        raise ValueError("Checkpoint is already marked calibrated; start from the original uncalibrated checkpoint")
    if state.get("calibration") not in (None, {}):
        raise ValueError("Uncalibrated source checkpoint unexpectedly contains calibration metadata")

    training = state.get("training")
    if not isinstance(training, dict):
        raise ValueError("Checkpoint is missing training provenance")
    training_fingerprint = training.get("manifest_fingerprint")
    if training_fingerprint != fingerprint:
        raise ValueError("Calibration manifest fingerprint differs from the checkpoint training manifest")

    detector = CheckpointDetector(checkpoint, device=device)
    dataset = ManifestAudioDataset(records, manifest.parent, "dev", seed=0)
    dataset.set_epoch(0)

    labels: list[int] = []
    raw_scores: list[float] = []
    for index in range(len(dataset)):
        values, _mask, target = dataset[index]
        labels.append(int(target.item()))
        raw_scores.append(detector.score(values.cpu().numpy(), 16_000))

    temperature = fit_temperature(labels, raw_scores)
    calibrated_scores = apply_temperature(raw_scores, temperature)
    original_threshold = float(detector.threshold)
    calibrated_threshold = transform_threshold(original_threshold, temperature)

    raw_decisions = np.asarray(raw_scores, dtype=np.float64) >= original_threshold
    calibrated_decisions = calibrated_scores >= calibrated_threshold
    if not np.array_equal(raw_decisions, calibrated_decisions):
        raise RuntimeError("Temperature calibration changed the checkpoint operating-point decisions")

    before = calibration_metrics(labels, raw_scores)
    after = calibration_metrics(labels, calibrated_scores)
    if after["nll"] > before["nll"] + 1e-8:
        raise RuntimeError("Fitted calibration unexpectedly worsened development NLL")

    source_sha256 = _sha256(checkpoint)
    metadata = {
        "method": CALIBRATION_METHOD,
        "temperature": float(temperature),
        "fit_split": "dev",
        "manifest_fingerprint": fingerprint,
        "records": len(labels),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "original_threshold": original_threshold,
        "calibrated_threshold": calibrated_threshold,
        "decision_preserving_threshold_transform": True,
        "metrics_before": before,
        "metrics_after": after,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_used": False,
    }

    calibrated_state = dict(state)
    calibrated_state["calibrated_probability"] = True
    calibrated_state["threshold"] = float(calibrated_threshold)
    calibrated_state["calibration"] = metadata
    calibrated_state["notice"] = (
        "VaaniRakshak V2 detector with development-set temperature calibration. "
        "The detector score is calibrated on the recorded development distribution; "
        "call-level risk aggregation is still not a probability of fraud or fakery."
    )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(calibrated_state, output_checkpoint)
    report_path = output_checkpoint.with_suffix(output_checkpoint.suffix + ".calibration.json")
    report = {
        "schema": "vaanirakshak-v2-calibration-report-v1",
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "calibration": metadata,
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    report = calibrate(
        args.manifest,
        args.checkpoint,
        args.output_checkpoint,
        device=args.device,
    )
    print(json.dumps(report["calibration"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
