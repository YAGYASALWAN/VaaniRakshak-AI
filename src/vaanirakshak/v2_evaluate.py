"""Evaluate a frozen VaaniRakshak V2 checkpoint without test-set tuning.

The checkpoint's saved development threshold is used exactly as exported. This
command supports ordinary, unseen-generator and cross-dataset audit modes and
writes both aggregate metrics and per-record scores for later error analysis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE
from vaanirakshak.v2_data_contract import (
    audit_cross_dataset,
    audit_manifest,
    audit_unseen_generator,
    load_jsonl,
    manifest_fingerprint,
)
from vaanirakshak.v2_detectors import CheckpointDetector
from vaanirakshak.v2_metrics import metrics_at_threshold


WINDOW_SAMPLES = 4 * MODEL_SAMPLE_RATE
HOP_SAMPLES = 2 * MODEL_SAMPLE_RATE
MIN_TAIL_SAMPLES = 2 * MODEL_SAMPLE_RATE


def _resolve_audio_ref(audio_ref: str, manifest_dir: Path) -> Path:
    path = Path(audio_ref).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _read_16khz(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    with sf.SoundFile(path) as stream:
        wave = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        rate = int(stream.samplerate)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    tensor = torch.from_numpy(wave.astype(np.float32, copy=False))
    if rate != MODEL_SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, rate, MODEL_SAMPLE_RATE)
    return tensor.float().clamp(-1.0, 1.0).cpu().numpy()


def _windows(wave: np.ndarray) -> list[np.ndarray]:
    if len(wave) <= WINDOW_SAMPLES:
        return [wave]
    result: list[np.ndarray] = []
    start = 0
    while start + WINDOW_SAMPLES <= len(wave):
        result.append(wave[start : start + WINDOW_SAMPLES])
        start += HOP_SAMPLES
    last_end = (start - HOP_SAMPLES) + WINDOW_SAMPLES if result else 0
    if len(wave) - last_end >= MIN_TAIL_SAMPLES:
        result.append(wave[start:])
    return result


def _score_recording(detector: CheckpointDetector, wave: np.ndarray) -> tuple[float, list[float], float]:
    started = time.perf_counter()
    scores = [detector.score(window, MODEL_SAMPLE_RATE) for window in _windows(wave)]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # Median is intentionally robust to one isolated anomalous window. The live
    # product has richer temporal aggregation; this recording-level benchmark
    # keeps the base detector evaluation simple and reproducible.
    return float(statistics.median(scores)), scores, elapsed_ms


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate(
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    device: str = "cpu",
    scenario: str = "standard",
    evaluation_datasets: set[str] | None = None,
) -> dict:
    manifest = manifest.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifest)
    if scenario == "unseen-generator":
        audit = audit_unseen_generator(records)
    elif scenario == "cross-dataset":
        audit = audit_cross_dataset(records, evaluation_datasets=evaluation_datasets or set())
    elif scenario == "standard":
        audit = audit_manifest(records)
    else:
        raise ValueError(f"Unknown evaluation scenario: {scenario}")
    audit.raise_for_errors()

    test_records = [record for record in records if record.split == "test"]
    if not test_records:
        raise ValueError("Manifest contains no test records")
    if {record.label for record in test_records} != {"bonafide", "spoof"}:
        raise ValueError("Binary evaluation requires both bonafide and spoof test records")

    detector = CheckpointDetector(checkpoint, device=device)
    rows: list[dict] = []
    labels: list[int] = []
    scores: list[float] = []

    for index, record in enumerate(test_records, 1):
        wave = _read_16khz(_resolve_audio_ref(record.audio_ref, manifest.parent))
        score, window_scores, elapsed_ms = _score_recording(detector, wave)
        label = 1 if record.label == "spoof" else 0
        labels.append(label)
        scores.append(score)
        rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset,
                "label": record.label,
                "generator_id": record.generator_id,
                "score": score,
                "threshold": detector.threshold,
                "predicted_spoof": score >= detector.threshold,
                "window_scores": window_scores,
                "window_count": len(window_scores),
                "elapsed_ms": elapsed_ms,
            }
        )
        if index % 100 == 0 or index == len(test_records):
            print(f"Evaluated {index:,}/{len(test_records):,}", flush=True)

    overall = metrics_at_threshold(labels, scores, detector.threshold)

    by_dataset: dict[str, dict] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        subset_labels = [1 if row["label"] == "spoof" else 0 for row in subset]
        if len(set(subset_labels)) == 2:
            by_dataset[dataset] = metrics_at_threshold(
                subset_labels,
                [float(row["score"]) for row in subset],
                detector.threshold,
            )
        else:
            by_dataset[dataset] = {
                "n": len(subset),
                "labels": sorted({row["label"] for row in subset}),
                "mean_score": statistics.fmean(float(row["score"]) for row in subset),
            }

    generator_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["generator_id"]:
            generator_rows[str(row["generator_id"])].append(row)
    by_generator = {
        generator: {
            "n": len(group),
            "mean_score": statistics.fmean(float(row["score"]) for row in group),
            "detection_rate": sum(bool(row["predicted_spoof"]) for row in group) / len(group),
        }
        for generator, group in sorted(generator_rows.items())
    }

    report = {
        "scenario": scenario,
        "checkpoint": str(checkpoint),
        "detector": detector.info.as_dict(),
        "manifest": str(manifest),
        "manifest_fingerprint": manifest_fingerprint(records),
        "test_records": len(test_records),
        "threshold_source": "frozen checkpoint; no test-set tuning",
        "overall": overall,
        "by_dataset": by_dataset,
        "by_generator": by_generator,
        "mean_recording_inference_ms": statistics.fmean(float(row["elapsed_ms"]) for row in rows),
        "audit_counts": audit.counts,
        "audit_warnings": list(audit.warnings),
    }
    _write_jsonl(output / "scores.jsonl", rows)
    _write_json(output / "evaluation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scenario", choices=("standard", "unseen-generator", "cross-dataset"), default="standard")
    parser.add_argument("--evaluation-dataset", action="append", default=[])
    args = parser.parse_args()

    report = evaluate(
        args.manifest,
        args.checkpoint,
        args.output,
        device=args.device,
        scenario=args.scenario,
        evaluation_datasets=set(args.evaluation_dataset),
    )
    print(json.dumps(report["overall"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
