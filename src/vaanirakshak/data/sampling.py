"""Deterministic duration-budgeted selection, with auditable feasibility checks."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .adapters import normalize_language
from .inspection import balance_table, common_languages, table_records
from .sampling_config import SamplingConfig


@dataclass
class SamplingResult:
    selected: pd.DataFrame
    report: dict[str, Any]


def plan_sample(frame: pd.DataFrame, config: SamplingConfig) -> SamplingResult:
    """Select original clips without downloading, segmenting, or assigning splits.

    Greedy heuristic: balance language progress, diversify speakers, then balance
    observed generator duration. Seeded identity hashes break ties independently
    of input row order. Budgets are hard upper limits; whole clips are indivisible.
    Infeasibility is reported, not solved by weakening caps or repeating clips.
    """
    work = frame.copy()
    if work["sample_id"].isna().any():
        raise ValueError("Every candidate requires a stable sample identity")
    if work["processed_file"].notna().any() or work["segment_start"].notna().any():
        raise ValueError("Plan from original source clips, not processed/overlapping windows")
    if work.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Duplicate identities would double-count hours")
    if work.dropna(subset=["source_file"]).duplicated(["dataset", "source_file"]).any():
        raise ValueError("Repeated source files would double-count original audio")
    if work["label"].isna().any() or not set(work["label"]).issubset({"bonafide", "spoof"}):
        raise ValueError("Invalid labels")
    # Whitelist prevents aliases/external data from entering the primary pool.
    primary = work["dataset"].isin(["indicvoices", "indicsynth"])
    expected = work["dataset"].map({"indicvoices": "bonafide", "indicsynth": "spoof"})
    if (primary & (work["label"] != expected)).any():
        raise ValueError("Primary source has an unexpected class")
    if "role" in work:
        primary &= work["role"].eq("train_dev")
    work = work[primary].copy()
    work["language"] = work["language"].map(normalize_language)
    language_sets = common_languages(work[work.label == "bonafide"], work[work.label == "spoof"])
    languages = list(config.languages) or language_sets["common"]
    if not languages or set(languages) - set(language_sets["common"]):
        raise ValueError("Selected languages must be observed in both primary classes")
    if set(config.target_hours_per_language) - set(languages):
        raise ValueError("Language targets refer to unselected languages")
    reasons: Counter[str] = Counter()
    reasons["external_or_nonprimary"] = int((~primary).sum())
    eligible = []
    for idx, row in work.iterrows():
        reason = None
        if row["language"] not in languages:
            reason = "unselected_language"
        elif pd.isna(row["duration"]) or float(row["duration"]) <= 0:
            reason = "missing_or_zero_duration"
        elif pd.isna(row["speaker_id"]) or not str(row["speaker_id"]).strip():
            reason = "missing_speaker"
        elif row["generator"] in config.holdout_generators:
            reason = "reserved_generator"
        elif row["label"] == "spoof" and pd.isna(row["generator"]):
            reason = "missing_generator"
        if reason:
            reasons[reason] += 1
        else:
            eligible.append(idx)
    work = work.loc[eligible].copy().reset_index(drop=True)
    if not work.empty:
        work["duration"] = pd.to_numeric(work["duration"], errors="raise")
        import numpy as np
        if not np.isfinite(work["duration"]).all():
            raise ValueError("Duration must be finite")
    class_budget = config.class_seconds()
    # Per-language values mean hours per class; unspecified languages divide remainder.
    budgets = {}
    for label, total in class_budget.items():
        explicit = sum(config.target_hours_per_language.values()) * 3600
        remainder_languages = [x for x in languages if x not in config.target_hours_per_language]
        if explicit > total + 1e-9:
            raise ValueError("Per-language hours exceed a class budget")
        for language in languages:
            budgets[(label, language)] = (
                config.target_hours_per_language[language] * 3600
                if language in config.target_hours_per_language else
                (total - explicit) / len(remainder_languages))
    seconds: dict[tuple[str, str], float] = defaultdict(float)
    speaker_clips: Counter[str] = Counter()
    speakers: set[str] = set()
    stratum_speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    generator_seconds: dict[str, float] = defaultdict(float)
    rows = work.to_dict(orient="index")
    rank = {
        i: hashlib.sha256(f"{config.seed}|{r.dataset}|{r.sample_id}".encode()).hexdigest()
        for i, r in work.iterrows()
    }
    remaining = set(work.index)
    selected: list[int] = []
    while remaining:
        candidates = []
        for i in remaining:
            row = rows[i]
            key = (row["label"], row["language"])
            speaker = row["speaker_id"]
            duration = float(row["duration"])
            if seconds[key] + duration > budgets[key] + 1e-9:
                continue
            if speaker_clips[speaker] >= config.max_clips_per_speaker:
                continue
            if speaker not in speakers and len(speakers) >= config.max_speakers:
                continue
            if row["label"] == "spoof":
                cap = class_budget["spoof"] * config.max_generator_share
                if generator_seconds[row["generator"]] + duration > cap + 1e-9:
                    continue
            needs_speaker = len(stratum_speakers[key]) < config.min_speakers_per_language
            candidates.append((
                0 if needs_speaker and speaker not in stratum_speakers[key] else 1,
                seconds[key] / budgets[key] if budgets[key] else 1,
                speaker_clips[speaker],
                generator_seconds[row["generator"]] if config.generator_balancing and row["label"] == "spoof" else 0,
                rank[i], i,
            ))
        if not candidates:
            break
        i = min(candidates)[-1]
        row = rows[i]
        key = (row["label"], row["language"])
        speaker = row["speaker_id"]
        selected.append(i)
        remaining.remove(i)
        seconds[key] += float(row["duration"])
        speaker_clips[speaker] += 1
        speakers.add(speaker)
        stratum_speakers[key].add(speaker)
        if row["label"] == "spoof":
            generator_seconds[row["generator"]] += float(row["duration"])
    chosen = work.loc[selected].copy()
    # Enforce actual selected-duration share, not just share of requested hours.
    # Remove excess from dominant generators. This can make the plan underfilled.
    removed = 0
    while len(chosen):
        spoof = chosen[chosen.label == "spoof"]
        amounts = spoof.groupby("generator")["duration"].sum()
        if amounts.empty or amounts.sum() <= 0:
            break
        dominant = amounts.idxmax()
        if amounts.max() <= config.max_generator_share * amounts.sum() + 1e-9:
            break
        drop = next(i for i in reversed(selected) if i in chosen.index and
                    chosen.at[i, "label"] == "spoof" and chosen.at[i, "generator"] == dominant)
        chosen = chosen.drop(index=drop)
        removed += 1
    chosen = chosen.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    problems = []
    strata = []
    for key, target in sorted(budgets.items()):
        group = chosen[(chosen.label == key[0]) & (chosen.language == key[1])]
        actual = float(group.duration.sum())
        count = int(group.speaker_id.nunique())
        strata.append({"label": key[0], "language": key[1], "target_seconds": target,
                       "selected_seconds": actual, "shortfall_seconds": max(0, target - actual),
                       "speakers": count})
        if target > 0 and actual < target * (1 - config.balance_tolerance) - 1e-9:
            problems.append(f"Underfilled {key[0]}/{key[1]}")
        if target > 0 and count < config.min_speakers_per_language:
            problems.append(f"Too few speakers in {key[0]}/{key[1]}")
    totals = {label: float(chosen.loc[chosen.label == label, "duration"].sum())
              for label in class_budget}
    gap = abs(totals["bonafide"] - totals["spoof"]) / max(max(totals.values()), 1e-12)
    if gap > config.balance_tolerance or not all(totals.values()):
        problems.append("Class audio durations are not balanced")
    return SamplingResult(chosen, {
        "status": "feasible" if not problems else "infeasible",
        "algorithm": "greedy diversity-first whole-clip selection; not a global optimizer",
        "config": asdict(config), "input_rows": len(frame), "eligible_rows": len(work),
        "excluded": dict(sorted(reasons.items())), "generator_cap_removals": removed,
        "class_balance": table_records(balance_table(chosen, ["label"])),
        "strata": strata, "class_duration_relative_gap": gap, "problems": problems,
        "languages": language_sets,
        "acquisition_authorized": False,
        "identity_warning": "Namespace and source/target-reference overlap require verification before splits.",
    })
