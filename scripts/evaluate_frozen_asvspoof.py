#!/usr/bin/env python3
"""Attack a frozen VaaniRakshak checkpoint with ASVspoof 2019 LA.

No training or parameter updates are performed. The goal is to measure how a
model trained on MLAAD + M-AILABS generalizes to a completely different corpus.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaanirakshak.baseline_rules import binary_metrics
from vaanirakshak.english_training import CACHE_VERSION, EnglishCNN, Frontend, stream_split, waveform_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Frozen MLAAD best.pt checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="Optional smoke test cap per class; 0 evaluates the full split")
    parser.add_argument("--output", type=Path,
                        help="JSON report path; defaults beside the checkpoint")
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be between 1 and 256")
    if args.limit_per_class < 0:
        parser.error("--limit-per-class cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if state.get("schema") != "english-cnn-v1" or state.get("frontend") != CACHE_VERSION:
        raise ValueError("Checkpoint is not compatible with the English CNN frontend")

    threshold = float(state.get("threshold", 0.5))
    frontend = Frontend().to(args.device).eval()
    model = EnglishCNN().to(args.device).eval()
    model.load_state_dict(state["model"])

    labels = []
    scores = []
    attacks = []
    ids = []
    accepted = Counter()
    pending_waves = []
    pending_meta = []
    started = time.monotonic()

    def flush():
        if not pending_waves:
            return
        waves = torch.from_numpy(np.stack(pending_waves)).to(args.device)
        with torch.no_grad(), torch.autocast(device_type=args.device, enabled=args.device == "cuda"):
            features = frontend(waves)
            batch_scores = model(features).sigmoid().float().cpu().tolist()
        for (label, attack, recording_id), score in zip(pending_meta, batch_scores):
            labels.append(label)
            scores.append(float(score))
            attacks.append(attack)
            ids.append(recording_id)
        pending_waves.clear()
        pending_meta.clear()

    print(f"Frozen checkpoint: {args.checkpoint}", flush=True)
    print(f"External corpus: ASVspoof 2019 LA / {args.split}", flush=True)
    print("NO TRAINING: checkpoint weights will not be modified.", flush=True)

    for source in stream_split(args.split):
        label = source.get("key")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Unexpected ASVspoof label: {label!r}")
        if args.limit_per_class and accepted[label] >= args.limit_per_class:
            if accepted[0] >= args.limit_per_class and accepted[1] >= args.limit_per_class:
                break
            continue

        audio = source.get("audio")
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        wave, _ = waveform_window(raw)
        pending_waves.append(wave)
        pending_meta.append((label, str(source.get("system_id", "-")),
                             str(source.get("audio_file_name", "unknown"))))
        accepted[label] += 1

        if len(pending_waves) >= args.batch_size:
            flush()
            if len(labels) % 1000 == 0:
                print(f"Scored {len(labels):,} recordings...", flush=True)

    flush()
    if set(labels) != {0, 1}:
        raise ValueError("Evaluation did not collect both classes")

    metrics = binary_metrics(labels, scores, threshold)
    by_attack = defaultdict(list)
    for label, score, attack in zip(labels, scores, attacks):
        if label == 1:
            by_attack[attack].append(score)

    attack_report = {}
    for attack, group in sorted(by_attack.items()):
        detected = sum(score >= threshold for score in group)
        attack_report[attack] = {
            "n": len(group),
            "synthetic_detected": detected,
            "recall": detected / len(group),
            "mean_synthetic_score": sum(group) / len(group),
        }

    human_scores = [score for label, score in zip(labels, scores) if label == 0]
    report = {
        "experiment": "frozen_cross_dataset_evaluation",
        "trained_on": "MLAAD synthetic + matching M-AILABS genuine",
        "tested_on": f"ASVspoof 2019 LA {args.split}",
        "no_parameter_updates": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": state.get("epoch"),
        "threshold": threshold,
        "counts": {"bonafide": accepted[0], "spoof": accepted[1]},
        "metrics": metrics,
        "human_false_positive_rate": sum(score >= threshold for score in human_scores) / len(human_scores),
        "human_mean_synthetic_score": sum(human_scores) / len(human_scores),
        "spoof_recall_by_attack": attack_report,
        "seconds": time.monotonic() - started,
        "interpretation": (
            "This is a source-shift test. A large drop from the in-domain MLAAD/M-AILABS result "
            "indicates dataset shortcuts or limited cross-corpus/generalization ability; it does not "
            "by itself prove classic parameter overfitting."
        ),
    }

    output = args.output or args.checkpoint.parent / f"cross_asvspoof_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print(f"Saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
