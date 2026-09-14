"""Frozen robustness evaluation for VaaniRakshak V2.

This evaluator never retrains the detector and never changes the threshold saved in
its checkpoint. It measures how a fixed detector degrades under controlled duration,
telephone-channel and synthetic-noise stress conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import numpy as np

from vaanirakshak.v2_data_contract import (
    audit_cross_dataset,
    audit_manifest,
    audit_unseen_generator,
    load_jsonl,
    manifest_fingerprint,
)
from vaanirakshak.v2_detectors import CheckpointDetector
from vaanirakshak.v2_evaluate import _read_16khz, _resolve_audio_ref, _score_recording
from vaanirakshak.v2_metrics import metrics_at_threshold
from vaanirakshak.v2_robustness import (
    center_duration,
    g711_alaw,
    g711_mulaw,
    narrowband_8khz,
    white_noise_at_snr,
)


Transform = Callable[[np.ndarray, str], np.ndarray]


def _identity(wave: np.ndarray, key: str) -> np.ndarray:
    del key
    return wave


def _duration(seconds: float) -> Transform:
    def apply(wave: np.ndarray, key: str) -> np.ndarray:
        del key
        return center_duration(wave, seconds)
    return apply


def _noise(snr_db: float) -> Transform:
    def apply(wave: np.ndarray, key: str) -> np.ndarray:
        return white_noise_at_snr(wave, snr_db, key=key)
    return apply


def _no_key(function: Callable[[np.ndarray], np.ndarray]) -> Transform:
    def apply(wave: np.ndarray, key: str) -> np.ndarray:
        del key
        return function(wave)
    return apply


def default_conditions() -> dict[str, Transform]:
    """Return the fixed V2 robustness suite in a stable report order."""
    return {
        "clean": _identity,
        "duration_2s": _duration(2.0),
        "duration_4s": _duration(4.0),
        "duration_8s": _duration(8.0),
        "narrowband_8khz": _no_key(narrowband_8khz),
        "g711_mulaw": _no_key(g711_mulaw),
        "g711_alaw": _no_key(g711_alaw),
        "noise_20db": _noise(20.0),
        "noise_10db": _noise(10.0),
        "noise_5db": _noise(5.0),
    }


def _audit(records, scenario: str, evaluation_datasets: set[str] | None):
    if scenario == "unseen-generator":
        return audit_unseen_generator(records)
    if scenario == "cross-dataset":
        return audit_cross_dataset(records, evaluation_datasets=evaluation_datasets or set())
    if scenario == "standard":
        return audit_manifest(records)
    raise ValueError(f"Unknown evaluation scenario: {scenario}")


def _delta(value: float, baseline: float) -> float:
    return float(value) - float(baseline)


def evaluate_robustness(
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    device: str = "cpu",
    scenario: str = "standard",
    evaluation_datasets: set[str] | None = None,
    save_scores: bool = False,
    conditions: dict[str, Transform] | None = None,
) -> dict:
    """Evaluate one frozen checkpoint under multiple deterministic stress conditions."""
    manifest = manifest.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifest)
    audit = _audit(records, scenario, evaluation_datasets)
    audit.raise_for_errors()

    test_records = [record for record in records if record.split == "test"]
    if not test_records:
        raise ValueError("Manifest contains no test records")
    if {record.label for record in test_records} != {"bonafide", "spoof"}:
        raise ValueError("Binary robustness evaluation requires both test labels")

    suite = conditions or default_conditions()
    if "clean" not in suite:
        raise ValueError("Robustness suite must include a 'clean' baseline")
    if not suite:
        raise ValueError("Robustness suite is empty")

    detector = CheckpointDetector(checkpoint, device=device)
    labels = np.empty(len(test_records), dtype=np.int8)
    scores = {name: np.empty(len(test_records), dtype=np.float32) for name in suite}
    latency_sum_ms = {name: 0.0 for name in suite}
    score_stream = None
    if save_scores:
        score_stream = (output / "robustness_scores.jsonl").open("w", encoding="utf-8", newline="\n")

    try:
        for index, record in enumerate(test_records):
            labels[index] = 1 if record.label == "spoof" else 0
            wave = _read_16khz(_resolve_audio_ref(record.audio_ref, manifest.parent))
            row_scores: dict[str, float] = {}

            for name, transform in suite.items():
                transformed = transform(wave, record.record_id)
                if transformed.ndim != 1 or transformed.size == 0 or not np.isfinite(transformed).all():
                    raise ValueError(f"Robustness transform {name!r} produced invalid audio for {record.record_id}")
                started = time.perf_counter()
                score, _window_scores, _elapsed_ms = _score_recording(detector, transformed)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                scores[name][index] = score
                latency_sum_ms[name] += elapsed_ms
                row_scores[name] = float(score)

            if score_stream is not None:
                score_stream.write(
                    json.dumps(
                        {
                            "record_id": record.record_id,
                            "dataset": record.dataset,
                            "label": record.label,
                            "generator_id": record.generator_id,
                            "threshold": detector.threshold,
                            "scores": row_scores,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            done = index + 1
            if done % 100 == 0 or done == len(test_records):
                print(f"Robustness evaluated {done:,}/{len(test_records):,}", flush=True)
    finally:
        if score_stream is not None:
            score_stream.close()

    reports: dict[str, dict] = {}
    for name in suite:
        metrics = metrics_at_threshold(labels.tolist(), scores[name].tolist(), detector.threshold)
        reports[name] = {
            **metrics,
            "mean_score": float(np.mean(scores[name], dtype=np.float64)),
            "mean_recording_inference_ms": latency_sum_ms[name] / len(test_records),
        }

    clean = reports["clean"]
    degradation: dict[str, dict] = {}
    for name, metrics in reports.items():
        if name == "clean":
            continue
        degradation[name] = {
            "delta_accuracy": _delta(metrics["accuracy"], clean["accuracy"]),
            "delta_f1": _delta(metrics["f1"], clean["f1"]),
            "delta_roc_auc": _delta(metrics["roc_auc"], clean["roc_auc"]),
            "delta_pr_auc": _delta(metrics["pr_auc"], clean["pr_auc"]),
            "delta_eer": _delta(metrics["eer"], clean["eer"]),
            "delta_false_positive_rate": _delta(metrics["false_positive_rate"], clean["false_positive_rate"]),
            "delta_false_negative_rate": _delta(metrics["false_negative_rate"], clean["false_negative_rate"]),
            "delta_mean_score": _delta(metrics["mean_score"], clean["mean_score"]),
        }

    report = {
        "scenario": scenario,
        "checkpoint": str(checkpoint),
        "detector": detector.info.as_dict(),
        "manifest": str(manifest),
        "manifest_fingerprint": manifest_fingerprint(records),
        "test_records": len(test_records),
        "threshold_source": "frozen checkpoint; unchanged for every robustness condition",
        "conditions": reports,
        "degradation_vs_clean": degradation,
        "audit_counts": audit.counts,
        "audit_warnings": list(audit.warnings),
        "score_file_written": bool(save_scores),
        "notice": (
            "Controlled stress transforms measure robustness of a frozen detector. "
            "White noise is synthetic and does not replace later real-noise evaluation."
        ),
    }
    (output / "robustness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scenario", choices=("standard", "unseen-generator", "cross-dataset"), default="standard")
    parser.add_argument("--evaluation-dataset", action="append", default=[])
    parser.add_argument("--save-scores", action="store_true")
    args = parser.parse_args()

    report = evaluate_robustness(
        args.manifest,
        args.checkpoint,
        args.output,
        device=args.device,
        scenario=args.scenario,
        evaluation_datasets=set(args.evaluation_dataset),
        save_scores=args.save_scores,
    )
    print(json.dumps(report["conditions"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
