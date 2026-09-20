"""Calibrate a frozen VaaniRakshak V2 detector on development data only.

The command fits one temperature scalar to recording-balanced sliding-window scores
from the development split. It never reads split=test. The existing development-
selected operating threshold is transformed through the same monotonic temperature
function, preserving every binary decision at that operating point.
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
    binary_nll,
    calibration_metrics,
    fit_temperature,
    transform_threshold,
)
from vaanirakshak.v2_data_contract import audit_manifest, load_jsonl, manifest_fingerprint
from vaanirakshak.v2_detectors import CALIBRATION_METHOD, CheckpointDetector, V2_CHECKPOINT_SCHEMA
from vaanirakshak.v2_evaluate import _read_16khz, _resolve_audio_ref, _windows


MAX_WINDOWS_PER_RECORDING = 4
CALIBRATION_UNIT = "sliding-4s-window-recording-balanced-v1"


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


def _select_calibration_windows(wave: np.ndarray, maximum: int = MAX_WINDOWS_PER_RECORDING) -> list[np.ndarray]:
    if maximum < 1:
        raise ValueError("maximum calibration windows must be positive")
    windows = _windows(wave)
    if len(windows) <= maximum:
        return windows
    if maximum == 1:
        return [windows[len(windows) // 2]]
    indices = [round(index * (len(windows) - 1) / (maximum - 1)) for index in range(maximum)]
    if len(set(indices)) != len(indices):
        raise RuntimeError("Calibration window selection produced duplicate indices")
    return [windows[index] for index in indices]


def calibrate(
    manifest: Path,
    checkpoint: Path,
    output_checkpoint: Path,
    *,
    device: str = "cpu",
    max_windows_per_recording: int = MAX_WINDOWS_PER_RECORDING,
) -> dict:
    manifest = manifest.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_checkpoint = output_checkpoint.expanduser().resolve()
    if checkpoint == output_checkpoint:
        raise ValueError("Calibration must write a new checkpoint; refusing to overwrite the source checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if max_windows_per_recording < 1:
        raise ValueError("max_windows_per_recording must be positive")

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
    labels: list[int] = []
    raw_scores: list[float] = []
    sample_weights: list[float] = []
    windows_per_recording: list[int] = []

    for index, record in enumerate(dev_records, 1):
        wave = _read_16khz(_resolve_audio_ref(record.audio_ref, manifest.parent))
        selected = _select_calibration_windows(wave, max_windows_per_recording)
        if not selected:
            raise ValueError(f"Development record produced no calibration windows: {record.record_id}")
        windows_per_recording.append(len(selected))
        record_weight = 1.0 / len(selected)
        label = 1 if record.label == "spoof" else 0
        for window in selected:
            labels.append(label)
            raw_scores.append(detector.score(window, 16_000))
            sample_weights.append(record_weight)
        if index % 100 == 0 or index == len(dev_records):
            print(f"Calibration scored {index:,}/{len(dev_records):,} development recordings", flush=True)

    temperature = fit_temperature(labels, raw_scores, sample_weights=sample_weights)
    calibrated_scores = apply_temperature(raw_scores, temperature)
    original_threshold = float(detector.threshold)
    calibrated_threshold = transform_threshold(original_threshold, temperature)

    raw_decisions = np.asarray(raw_scores, dtype=np.float64) >= original_threshold
    calibrated_decisions = calibrated_scores >= calibrated_threshold
    if not np.array_equal(raw_decisions, calibrated_decisions):
        raise RuntimeError("Temperature calibration changed the checkpoint operating-point decisions")

    before = calibration_metrics(labels, raw_scores)
    after = calibration_metrics(labels, calibrated_scores)
    weighted_nll_before = binary_nll(labels, raw_scores, sample_weights=sample_weights)
    weighted_nll_after = binary_nll(labels, calibrated_scores, sample_weights=sample_weights)
    if weighted_nll_after > weighted_nll_before + 1e-8:
        raise RuntimeError("Fitted calibration unexpectedly worsened recording-balanced development NLL")

    source_sha256 = _sha256(checkpoint)
    metadata = {
        "method": CALIBRATION_METHOD,
        "temperature": float(temperature),
        "fit_split": "dev",
        "calibration_unit": CALIBRATION_UNIT,
        "recording_balanced_fit": True,
        "max_windows_per_recording": int(max_windows_per_recording),
        "development_recordings": len(dev_records),
        "calibration_windows": len(labels),
        "min_windows_per_recording": min(windows_per_recording),
        "max_windows_observed_per_recording": max(windows_per_recording),
        "manifest_fingerprint": fingerprint,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "original_threshold": original_threshold,
        "calibrated_threshold": calibrated_threshold,
        "decision_preserving_threshold_transform": True,
        "window_metrics_before": before,
        "window_metrics_after": after,
        "recording_balanced_nll_before": weighted_nll_before,
        "recording_balanced_nll_after": weighted_nll_after,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_used": False,
    }

    calibrated_state = dict(state)
    calibrated_state["calibrated_probability"] = True
    calibrated_state["threshold"] = float(calibrated_threshold)
    calibrated_state["calibration"] = metadata
    calibrated_state["notice"] = (
        "VaaniRakshak V2 detector with development-set, recording-balanced sliding-window temperature calibration. "
        "The detector score is calibrated for the live window-level score unit; "
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
    parser.add_argument("--max-windows-per-recording", type=int, default=MAX_WINDOWS_PER_RECORDING)
    args = parser.parse_args()

    report = calibrate(
        args.manifest,
        args.checkpoint,
        args.output_checkpoint,
        device=args.device,
        max_windows_per_recording=args.max_windows_per_recording,
    )
    print(json.dumps(report["calibration"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
