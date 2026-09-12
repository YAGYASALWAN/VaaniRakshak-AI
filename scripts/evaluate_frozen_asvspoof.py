#!/usr/bin/env python3
"""Attack a frozen VaaniRakshak checkpoint with ASVspoof 2019 LA.

No training or parameter updates are performed. The goal is to measure how a
checkpoint trained on one corpus generalizes to a completely different one.

The corpus that produced the checkpoint is read from the checkpoint itself, not
assumed here: both the ASVspoof and the MLAAD pipelines write the same
``english-cnn-v1`` schema, so a hardcoded description would silently mislabel an
in-domain run as a cross-corpus one.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaanirakshak.baseline_rules import binary_metrics
from vaanirakshak.english_rules import COUNTS, REPOSITORY, REVISION
from vaanirakshak.english_training import CACHE_VERSION, EnglishCNN, Frontend, score_features, stream_split, waveform_window

# Official ASVspoof 2019 LA attack families represented in the development and
# evaluation partitions. Stratified mode uses these quotas so an ordered stream
# cannot accidentally produce a one-attack-only smoke test.
ATTACKS_BY_SPLIT = {
    "validation": tuple(f"A{i:02d}" for i in range(1, 7)),
    "test": tuple(f"A{i:02d}" for i in range(7, 20)),
}
# Let roughly this many recordings through the sampling gate for every one the
# quota needs, so an attack with a below-average population is not starved.
GATE_HEADROOM = 1.5


def sampling_stride(population, wanted):
    """Keep one recording in N, so a quota is drawn from across the whole stream."""
    if not wanted:
        return 1
    return max(1, int(population / (wanted * GATE_HEADROOM)))


def gate(recording_id, stride, seed):
    """Deterministic, order-independent membership test for the sampled subset."""
    if stride <= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{recording_id}".encode()).hexdigest()[:8]
    return int(digest, 16) % stride == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Frozen best.pt checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit-per-class", type=int, default=0,
                       help="Optional ordered smoke-test cap per class; 0 evaluates the full split")
    group.add_argument("--per-attack", type=int, default=0,
                       help="Stratified mode: score this many spoof recordings from every official attack family; "
                            "bonafide count is matched to the total spoof count")
    parser.add_argument("--sample-stride", type=int, default=0,
                        help="Stratified mode: keep one recording in N, spreading the sample across the "
                             "stream instead of taking its head. 0 derives a stride from the split size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the sampling gate")
    parser.add_argument("--output", type=Path,
                        help="JSON report path; defaults beside the checkpoint")
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be between 1 and 256")
    if args.limit_per_class < 0 or args.per_attack < 0 or args.sample_stride < 0:
        parser.error("sampling limits cannot be negative")
    if args.sample_stride and not args.per_attack:
        parser.error("--sample-stride applies to stratified mode only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    expected_attacks = ATTACKS_BY_SPLIT[args.split]
    human_target = args.per_attack * len(expected_attacks) if args.per_attack else None
    spoof_stride = human_stride = 1
    if args.per_attack:
        spoof_stride = args.sample_stride or sampling_stride(
            COUNTS[args.split][1] / len(expected_attacks), args.per_attack)
        human_stride = args.sample_stride or sampling_stride(COUNTS[args.split][0], human_target)

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
    skipped = []
    pending_waves = []
    pending_meta = []
    printed = 0
    started = time.monotonic()

    def flush():
        """Score the pending batch, dropping any recording the model cannot score."""
        if not pending_waves:
            return
        waves = torch.from_numpy(np.stack(pending_waves)).to(args.device)
        batch_scores, usable = score_features(frontend, model, waves)
        for (label, attack, recording_id), score, ok in zip(pending_meta, batch_scores, usable):
            score = float(score)
            # Out-of-domain audio producing a non-finite or out-of-range score is a
            # finding, not a crash condition. Record it and keep the run alive.
            if not ok or not np.isfinite(score) or not 0.0 <= score <= 1.0:
                skipped.append({"id": recording_id, "label": label, "attack": attack})
                accepted[label] -= 1
                if label == 1:
                    accepted_attacks[attack] -= 1
                continue
            labels.append(label)
            scores.append(score)
            attacks.append(attack)
            ids.append(recording_id)
        pending_waves.clear()
        pending_meta.clear()

    def stratified_complete():
        return bool(
            args.per_attack
            and accepted[0] >= human_target
            and all(accepted_attacks[a] >= args.per_attack for a in expected_attacks)
        )

    print(f"Frozen checkpoint: {args.checkpoint}", flush=True)
    print(f"Trained on: {state.get('notice', 'unrecorded')}", flush=True)
    print(f"External corpus: ASVspoof 2019 LA / {args.split} @ {REVISION[:12]}", flush=True)
    print("NO TRAINING: checkpoint weights will not be modified.", flush=True)
    print("Numerics: float32 inference for robust cross-corpus evaluation.", flush=True)
    if args.per_attack:
        print(f"Stratified attack test: {args.per_attack} per attack across {', '.join(expected_attacks)}; "
              f"{human_target} matched bonafide recordings.", flush=True)
        print(f"Sampling gate: 1 in {spoof_stride} spoof, 1 in {human_stride} bonafide, seed {args.seed}.", flush=True)

    for source in stream_split(args.split):
        label = source.get("key")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Unexpected ASVspoof label: {label!r}")
        attack = str(source.get("system_id", "-"))
        recording_id = str(source.get("audio_file_name", "unknown"))

        if args.per_attack:
            if label == 0:
                if accepted[0] >= human_target or not gate(recording_id, human_stride, args.seed):
                    continue
            else:
                if attack not in expected_attacks or accepted_attacks[attack] >= args.per_attack:
                    continue
                if not gate(recording_id, spoof_stride, args.seed):
                    continue
        elif args.limit_per_class and accepted[label] >= args.limit_per_class:
            if accepted[0] >= args.limit_per_class and accepted[1] >= args.limit_per_class:
                break
            continue

        audio = source.get("audio")
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        wave, _ = waveform_window(raw)
        pending_waves.append(wave)
        pending_meta.append((label, attack, recording_id))
        accepted[label] += 1
        if label == 1:
            accepted_attacks[attack] += 1

        if len(pending_waves) >= args.batch_size:
            flush()
            # Compare against a running marker: len(labels) is always a multiple of
            # the batch size here, so a bare modulo almost never fires.
            if len(labels) // 250 > printed:
                printed = len(labels) // 250
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
            raise ValueError(f"Incomplete stratified sample: bonafide={accepted[0]}/{human_target}, "
                             f"missing attacks={missing}. Lower --per-attack or --sample-stride.")

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
        pair_metrics = binary_metrics(pair_labels, pair_scores, threshold)
        attack_report[attack] = {
            "n": len(attack_scores),
            "synthetic_detected": detected,
            "recall_at_checkpoint_threshold": detected / len(attack_scores),
            "mean_synthetic_score": sum(attack_scores) / len(attack_scores),
            "roc_auc_vs_selected_humans": pair_metrics["roc_auc"],
            "eer_vs_selected_humans": pair_metrics["eer"],
        }

    mode = "stratified_by_attack" if args.per_attack else ("ordered_class_cap" if args.limit_per_class else "full_split")
    report = {
        "experiment": "frozen_cross_dataset_evaluation",
        "sampling_mode": mode,
        # Provenance comes from the checkpoint. Both training pipelines write the same
        # schema, so a literal here could describe the wrong corpus entirely.
        "trained_on": state.get("notice"),
        "training_data_identity": state.get("data_identity"),
        "tested_on": f"ASVspoof 2019 LA {args.split}",
        "test_repository": REPOSITORY,
        "test_revision": REVISION,
        "no_parameter_updates": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": state.get("epoch"),
        "threshold": threshold,
        "counts": {"bonafide": accepted[0], "spoof": accepted[1]},
        "selected_spoof_counts_by_attack": dict(sorted(accepted_attacks.items())),
        "sampling": {"seed": args.seed, "spoof_stride": spoof_stride, "bonafide_stride": human_stride},
        "skipped_recordings": skipped,
        "metrics": metrics,
        "human_false_positive_rate": sum(score >= threshold for score in human_scores) / len(human_scores),
        "human_mean_synthetic_score": sum(human_scores) / len(human_scores),
        "spoof_recall_by_attack": attack_report,
        "seconds": time.monotonic() - started,
        "interpretation": (
            "This is a source-shift test. A large drop from the in-domain result indicates dataset "
            "shortcuts or limited cross-corpus/generalization ability; it does not by itself prove "
            "classic parameter overfitting. Read the EER and per-attack AUC rather than the "
            "threshold-dependent figures: the checkpoint threshold was fitted in-domain, and "
            "threshold shift is one of the things cross-corpus transfer breaks. In stratified mode "
            "the sample is drawn by a deterministic hash gate spread across the stream, but a quota "
            "that fills early still favours the earlier part of it."
        ),
    }

    suffix = f"cross_asvspoof_{args.split}_stratified.json" if args.per_attack else f"cross_asvspoof_{args.split}.json"
    output = args.output or args.checkpoint.parent / suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print(f"Saved report: {output}", flush=True)
    if skipped:
        print(f"WARNING: {len(skipped)} recordings were unscorable and excluded; see skipped_recordings.", flush=True)


if __name__ == "__main__":
    main()
