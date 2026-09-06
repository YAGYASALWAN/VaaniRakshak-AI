"""Transparent heuristic shortcut warnings; not a statistical certification."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .inspection import common_languages
from .sampling_config import BiasThresholds


def audit(frame: pd.DataFrame, thresholds: BiasThresholds | None = None) -> dict[str, Any]:
    """Compare classes using known values and separately disclose missingness.

    TV distance = half the absolute difference between categorical proportions.
    Both clip-weighted and duration-weighted distributions are checked. Dominance
    uses the larger of observed clip and time shares. LOW does not mean safe.
    """
    cfg = thresholds or BiasThresholds()
    findings: list[dict[str, Any]] = []
    groups = {label: frame[frame["label"] == label] for label in ("bonafide", "spoof")}

    def add(code: str, value: float, medium: float, high: float,
            evidence: Any, message: str) -> None:
        findings.append({"code": code, "severity": "HIGH" if value >= high else
                         "MEDIUM" if value >= medium else "LOW",
                         "value": float(value), "medium_threshold": medium,
                         "high_threshold": high, "evidence": evidence, "message": message})

    if any(g.empty for g in groups.values()):
        return {"status": "insufficient_data", "thresholds": asdict(cfg), "findings": [],
                "reason": "Both bonafide and spoof rows are required"}
    for column in ("duration", "sample_rate", "codec", "file_format", "language",
                   "gender", "speaker_id", "generator", "rms"):
        relevant = {"spoof": groups["spoof"]} if column == "generator" else groups
        missing = {label: float(g[column].isna().mean()) if column in g else 1.0
                   for label, g in relevant.items()}
        add(f"missing_{column}", max(missing.values()), cfg.missing_medium, cfg.missing_high,
            missing, "Missing metadata limits this audit; absence is not evidence of balance.")
    for column in ("sample_rate", "channels", "codec", "file_format", "language", "gender"):
        if column not in frame:
            continue
        distances = {}
        for weighting in ("clips", "duration"):
            distributions = []
            for g in groups.values():
                known = g.dropna(subset=[column])
                if weighting == "duration":
                    known = known.dropna(subset=["duration"])
                    counts = known.groupby(column)["duration"].sum()
                else:
                    counts = known[column].value_counts()
                distributions.append(counts / counts.sum() if counts.sum() else counts)
            p, q = distributions
            if len(p) and len(q) and p.sum() and q.sum():
                keys = p.index.union(q.index)
                distances[weighting] = float(
                    (p.reindex(keys, fill_value=0) - q.reindex(keys, fill_value=0)).abs().sum() / 2)
        if distances:
            add(column, max(distances.values()), cfg.categorical_medium, cfg.categorical_high,
                distances, f"{column} distributions may reveal class (total variation distance).")
    languages = common_languages(groups["bonafide"], groups["spoof"])
    if languages["only_indicvoices"] or languages["only_indicsynth"]:
        add("one_class_language", 1.0, cfg.categorical_medium, cfg.categorical_high,
            languages, "Some observed languages occur in only one class.")
    means = {label: pd.to_numeric(g["duration"]).mean() for label, g in groups.items()}
    if all(pd.notna(v) for v in means.values()):
        low, high = min(means.values()), max(means.values())
        ratio = high / low if low > 0 else (1.0 if high == 0 else None)
        add("duration", ratio if ratio is not None else cfg.duration_ratio_high,
            cfg.duration_ratio_medium, cfg.duration_ratio_high,
            {"means_seconds": means, "ratio": ratio}, "Mean duration ratio may reveal class.")
    for column, relevant, prefix in (
        ("speaker_id", groups, "speaker_share"),
        ("generator", {"spoof": groups["spoof"]}, "generator_share"),
    ):
        for label, group in relevant.items():
            known = group.dropna(subset=[column])
            shares = {}
            if len(known):
                shares["clips"] = float(known[column].value_counts(normalize=True).max())
                duration = known.groupby(column)["duration"].sum()
                if duration.sum() > 0:
                    shares["duration"] = float(duration.max() / duration.sum())
                add(f"{column}_dominance_{label}", max(shares.values()),
                    getattr(cfg, prefix + "_medium"), getattr(cfg, prefix + "_high"),
                    shares, f"A single {column} contributes a large share of observed {label} data.")
    if "rms" in frame:
        rms = {k: pd.to_numeric(g["rms"]).mean() for k, g in groups.items()}
        if all(pd.notna(v) for v in rms.values()):
            add("loudness_rms", abs(rms["bonafide"] - rms["spoof"]),
                cfg.rms_gap_medium, cfg.rms_gap_high, rms,
                "Mean RMS differs by class; RMS is an amplitude proxy, not LUFS.")
    # Dataset and class are confounded even if observed nuisance factors match.
    per_dataset = frame.groupby("dataset")["label"].nunique()
    if len(per_dataset) > 1 and (per_dataset == 1).all():
        add("dataset_class_confounding", 1.0, cfg.categorical_medium, cfg.categorical_high,
            {"datasets": sorted(per_dataset.index.tolist())},
            "Each source contains one class; unmeasured dataset fingerprints remain a risk.")
    return {"status": "audited", "scope": "supplied metadata only",
            "thresholds": asdict(cfg), "findings": findings,
            "limitations": ["Heuristic warnings, not proof of causation or fairness.",
                           "No model trained; environment, source reuse and acoustic artifacts need separate checks."]}
