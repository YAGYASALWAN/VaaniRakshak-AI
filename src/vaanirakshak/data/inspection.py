"""Descriptive metadata statistics with explicit missing-value denominators."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapters import normalize_language

CATEGORIES = ("sample_rate", "channels", "file_format", "codec", "language",
              "gender", "attack_type", "generator")


def numeric_stats(values: pd.Series) -> dict[str, Any]:
    """Summarize finite observed measurements; an entirely unknown total is null."""
    numeric = pd.to_numeric(values, errors="raise").dropna().astype(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite measurement")
    return {
        "known": len(numeric), "missing": len(values) - len(numeric),
        "total": float(numeric.sum()) if len(numeric) else None,
        "mean": float(numeric.mean()) if len(numeric) else None,
        "median": float(numeric.median()) if len(numeric) else None,
        "min": float(numeric.min()) if len(numeric) else None,
        "max": float(numeric.max()) if len(numeric) else None,
    }


def balance_table(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Counts and observed duration per group, including missing group values."""
    work = frame.copy()
    work["duration"] = pd.to_numeric(work["duration"], errors="raise")
    table = work.groupby(fields, dropna=False, sort=True).agg(
        clips=("sample_id", "size"), known_duration_clips=("duration", "count"),
        duration_seconds=("duration", lambda x: x.sum(min_count=1)),
        unique_speakers=("speaker_id", "nunique"),
    ).reset_index()
    table["hours"] = table["duration_seconds"] / 3600
    table["missing_duration_clips"] = table["clips"] - table["known_duration_clips"]
    return table


def table_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe primitive records, with missing values encoded as null."""
    import json
    return json.loads(table.to_json(orient="records", double_precision=12))


def common_languages(real: pd.DataFrame, spoof: pd.DataFrame) -> dict[str, list[str]]:
    """Compare observed languages; nulls never count as a common language."""
    a = {normalize_language(x) for x in real["language"]}
    b = {normalize_language(x) for x in spoof["language"]}
    a.discard(None)
    b.discard(None)
    return {"common": sorted(a & b), "only_indicvoices": sorted(a - b),
            "only_indicsynth": sorted(b - a)}


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize supplied rows only (not a claim about the complete remote corpus).

    Duration is source audio seconds, never overlapping processed-window hours.
    No audio is accessed. Unknown duration makes total hours a partial subtotal.
    """
    duration = numeric_stats(frame["duration"])
    if duration["min"] is not None and duration["min"] < 0:
        raise ValueError("Negative duration")
    distributions = {}
    for field in CATEGORIES:
        series = frame[field] if field in frame else pd.Series([None] * len(frame))
        distributions[field] = {
            "missing": int(series.isna().sum()),
            "counts": {str(k): int(v) for k, v in series.dropna().astype(str).value_counts().sort_index().items()},
        }
    result: dict[str, Any] = {
        "scope": "supplied metadata rows only", "total_rows": len(frame),
        "known_source_files": int(frame["source_file"].notna().sum()),
        "duration_seconds": duration,
        "total_hours": duration["total"] / 3600 if duration["total"] is not None else None,
        "duration_complete": duration["missing"] == 0,
        "unique_speakers": int(frame["speaker_id"].nunique()),
        "missing_speaker_rows": int(frame["speaker_id"].isna().sum()),
        "distributions": distributions,
        "class_balance": table_records(balance_table(frame, ["label"])),
        "languages": table_records(balance_table(frame, ["label", "language"])),
        "speakers": table_records(balance_table(frame, ["label", "speaker_id"])),
    }
    synthetic = frame[frame["label"] == "spoof"]
    result["generators"] = table_records(balance_table(synthetic, ["generator"]))
    result["language_generator"] = table_records(balance_table(synthetic, ["language", "generator"]))
    result["languages_per_generator"] = {
        str(g): sorted(group["language"].dropna().unique().tolist())
        for g, group in synthetic.dropna(subset=["generator"]).groupby("generator", sort=True)
    }
    if "rms" in frame:
        result["rms"] = numeric_stats(frame["rms"])
    if "almost_silent" in frame:
        known = frame["almost_silent"].dropna()
        result["almost_silent"] = {
            "measured": len(known), "missing": len(frame) - len(known),
            "percent": 100 * float(known.astype(bool).mean()) if len(known) else None,
        }
    return result


def inspect_local_audio(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Measure explicitly supplied local raw audio under root; no URLs or downloads.

    Reuses Milestone 1 loader and silence validation. Decode failures are reported
    per row; missing or remote files remain unknown. RMS is not LUFS.
    """
    from vaanirakshak.config import ValidationConfig
    from .audio_io import load_audio
    from .validation import validate_audio_data
    from vaanirakshak.exceptions import AudioLoadError

    work = frame.copy()
    work["audio_error"] = None
    base = root.resolve()
    for index, row in work.iterrows():
        source = row["source_file"]
        if pd.isna(source) or "://" in str(source):
            continue
        path = (base / str(source)).resolve()
        if not path.is_relative_to(base):
            work.at[index, "audio_error"] = "outside audio root"
            continue
        try:
            audio = load_audio(path)
            report = validate_audio_data(audio, config=ValidationConfig())
        except AudioLoadError as exc:
            work.at[index, "audio_error"] = str(exc)
            continue
        checks = {check.name: check for check in report.checks}
        if not checks["finite_samples"].passed:
            work.at[index, "audio_error"] = "non-finite waveform"
            continue
        for field, value in {
            "duration": audio.duration_seconds, "sample_rate": audio.sample_rate,
            "channels": audio.num_channels, "codec": audio.subtype,
            "file_format": audio.format.lower() if audio.format else None,
            "rms": checks["not_completely_silent"].value,
            "almost_silent": not checks["not_almost_silent"].passed,
        }.items():
            work.at[index, field] = value
    return work
