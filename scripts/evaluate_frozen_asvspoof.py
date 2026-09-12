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

# Official ASVspoof 2019 LA attack families represented in the development and
# evaluation partitions. Stratified mode uses these quotas so an ordered stream
# cannot accidentally produce a one-attack-only smoke test.
ATTACKS_BY_SPLIT = {
    "validation": tuple(f"A{i:02d}" for i in range(1, 7)),
    "test": tuple(f"A{i:02d}" for i in range(7, 20)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Frozen MLAAD best.pt checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit-per-class", type=int, default=0,
                       help="Optional ordered smoke-test cap per class; 0 evaluates the full split")
    group.add_argument("--per-attack", type=int, default=0,
                       help="Stratified mode: score this many spoof recordings from every official attack family; "
                            "bonafide count is matched to the total spoof count")
    parser.add_argument("--output", type=Path,
                        help="JSON report path; defaults beside the checkpoint")
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be between 1 and 256")
    if args.limit_per_class < 0 or args.per_attack < 0:
        parser.error("sampling limits cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    expected_attacks = ATTACKS_BY_SPLIT[args.split]
    human_target = args.per_attack * len(expected_attacks) if args.per_attack else None

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
    accepted_attacks = Counter()
    pending_waves = []
    pending_meta = []
    started = time.monotonic()

    def flush():
        if not pending_waves:
            return
        waves = torch.from_numpy(np.stack(pending_waves)).to(args.device)
        # Deliberately evaluate in float32. External-corpus feature distributions can
        # be more extreme than the training corpus, and AMP/float16 can overflow even
        # when the same model is numerically stable in-domain.
        with torch.no_grad():
            features = frontend(waves).float()
            if not torch.isfinite(features).all():
                bad = [meta[2] for meta, ok in zip(pending_meta, torch.isfinite(features).flatten(1).all(1).tolist()) if not ok]
                raise ValueError(f"Non-finite frontend features for recordings: {bad[:10]}")
            logits = model(features).float()
            if not torch.isfinite(logits).all():
                bad = [meta[2] for meta, ok in zip(pending_meta, torch.isfinite(logits).tolist()) if not ok]
                raise ValueError(f"Non-finite model logits for recordings: {bad[:10]}")
            batch_scores = logits.sigmoid().cpu().tolist()
        for (label, attack, recording_id), score in zip(pending_meta, batch_scores):
            score = float(score)
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid model score for {recording_id}: {score!r}")
            labels.append(label)
            scores.append(score)
            attacks.append(attack)
            ids.append(recording_id)
        pending_waves.clear()
        pending_meta.clear()

    def stratified_complete():
        return (
            args.per_attack
            and accepted[0] >= human_target
            and all(accepted_attacks[a] >= args.per_attack for a in expected_attacks)
        )

    print(f"Frozen checkpoint: {args.checkpoint}", flush=True)
    print(f"External corpus: ASVspoof 2019 LA / {args.split}", flush=True)
    print("NO TRAINING: checkpoint weights will not be modified.", flush=True)
    print("Numerics: float32 inference for robust cross-corpus evaluation.", flush=True)
    if args.per_attack:
        print(f"Stratified attack test: {args.per_attack} per attack across {', '.join(expected_attacks)}; "
              f"{human_target} matched bonafide recordings.", flush=True)

    for source in stream_split(args.split):
        label = source.get("key")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Unexpected ASVspoof label: {label!r}")
        attack = str(source.get("system_id", "-"))

        if args.per_attack:
            if label == 0:
                if accepted[0] >= human_target:
                    continue
            else:
                if attack not in expected_attacks or accepted_attacks[attack] >= args.per_attack:
                    continue
            if stratified_complete():
                break
        elif args.limit_per_class and accepted[label] >= args.limit_per_class:
            if accepted[0] >= args.limit_per_class and accepted[1] >= args.limit_per_class:
                break
            continue

        audio = source.get("audio")
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        wave, _ = waveform_window(raw)
        pending_waves.append(wave)
        pending_meta.append((label, attack, str(source.get("audio_file_name", "unknown"))))
        accepted[label] += 1
        if label == 1:
            accepted_attacks[attack] += 1

        if len(pending_waves) >= args.batch_size:
            flush()
            if len(labels) and len(labels) % 250 == 0:
                print(f"Scored {len(labels):,} selected recordings; attacks={dict(sorted(accepted_attacks.items()))}", flush=True)

        if stratified_complete():
            break

    flush()
    if set(labels) != {0, 1}:
        raise ValueError("Evaluation did not collect both classes")
    if args.per_attack:
        missing = {a: args.per_attack - accepted_attacks[a]
                   for a in expected_attacks if accepted_attacks[a] < args.per_attack}
        if accepted[0] < human_target or missing:
            raise ValueError(f"Incomplete stratified sample: bonafide={accepted[0]}/{human_target}, missing attacks={missing}")

    metrics = binary_metrics(labels, scores, threshold)
    by_attack = defaultdict(list)
    for label, score, attack in zip(labels, scores, attacks):
        if label == 1:
            by_attack[attack].append(score)

    human_scores = [score for label, score in zip(labels, scores) if label == 0]
    attack_report = {}
    for attack, attack_scores in sorted(by_attack.items()):
        detected = sum(score >= threshold for score in attack_scores)
        pair_labels = [0] * len(human_scores) + [1] * len(attack_scores)
        pair_scores = human_scores + attack_scores
        attack_report[attack] = {
            "n": len(attack_scores),
            "synthetic_detected": detected,
            "recall_at_checkpoint_threshold": detected / len(attack_scores),
            "mean_synthetic_score": sum(attack_scores) / len(attack_scores),
            "roc_auc_vs_selected_humans": binary_metrics(pair_labels, pair_scores, threshold)["roc_auc"],
        }

    mode = "stratified_by_attack" if args.per_attack else ("ordered_class_cap" if args.limit_per_class else "full_split")
    report = {
        "experiment": "frozen_cross_dataset_evaluation",
        "sampling_mode": mode,
        "trained_on": "MLAAD synthetic + matching M-AILABS genuine",
        "tested_on": f"ASVspoof 2019 LA {args.split}",
        "no_parameter_updates": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": state.get("epoch"),
        "threshold": threshold,
        "counts": {"bonafide": accepted[0], "spoof": accepted[1]},
        "selected_spoof_counts_by_attack": dict(sorted(accepted_attacks.items())),
        "metrics": metrics,
        "human_false_positive_rate": sum(score >= threshold for score in human_scores) / len(human_scores),
        "human_mean_synthetic_score": sum(human_scores) / len(human_scores),
        "spoof_recall_by_attack": attack_report,
        "seconds": time.monotonic() - started,
        "interpretation": (
            "This is a source-shift test. A large drop from the in-domain MLAAD/M-AILABS result "
            "indicates dataset shortcuts or limited cross-corpus/generalization ability; it does not "
            "by itself prove classic parameter overfitting. Per-attack AUC compares each spoof family "
            "against the same selected bonafide recordings and is threshold-independent."
        ),
    }

    suffix = f"cross_asvspoof_{args.split}_stratified.json" if args.per_attack else f"cross_asvspoof_{args.split}.json"
    output = args.output or args.checkpoint.parent / suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print(f"Saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
