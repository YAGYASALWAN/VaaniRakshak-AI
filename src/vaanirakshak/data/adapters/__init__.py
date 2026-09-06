"""Metadata-only adapters. No downloads, audio decoding, or inferred generators."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from vaanirakshak.data.metadata import MANIFEST_COLUMNS, validate_manifest_frame

EXTRA_COLUMNS = (
    "channels", "file_format", "age_group", "transcript", "language_raw",
    "speaker_id_raw", "source_speaker_id", "source_reference", "target_reference",
    "source_split", "role", "metadata_locator", "rms", "almost_silent",
)
INSPECTION_COLUMNS = (*MANIFEST_COLUMNS, *EXTRA_COLUMNS)
LANGUAGE_GROUPS = (
    ("as", "asm", "assamese"), ("bn", "ben", "bengali", "bangla"),
    ("brx", "bodo"), ("doi", "dogri"), ("gu", "guj", "gujarati"),
    ("hi", "hin", "hindi"), ("kn", "kan", "kannada"), ("ks", "kas", "kashmiri"),
    ("kok", "gom", "konkani"), ("mai", "maithili"), ("ml", "mal", "malayalam"),
    ("mr", "mar", "marathi"), ("mni", "manipuri", "meitei"),
    ("ne", "nep", "nepali"), ("or", "ori", "ory", "odia", "oriya"),
    ("pa", "pan", "punjabi", "panjabi"), ("sa", "san", "sanskrit"),
    ("sat", "santali"), ("sd", "snd", "sindhi"), ("ta", "tam", "tamil"),
    ("te", "tel", "telugu"), ("ur", "urd", "urdu"), ("en", "eng", "english"),
)
LANGUAGES = {alias: group[0] for group in LANGUAGE_GROUPS for alias in group}


def scalar(value: Any) -> str | None:
    """Normalize a scalar metadata value; structured values are not stringified."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        raise ValueError("Expected scalar metadata, received a nested value")
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text and text.lower() not in {"null", "nan", "none", "<na>"} else None


def normalize_language(value: Any) -> str | None:
    """Map known names/codes; preserve unknown normalized tokens for explicit mapping."""
    text = scalar(value)
    if text is None:
        return None
    token = text.lower().replace("_", "-")
    return LANGUAGES.get(token, LANGUAGES.get(token.split("-")[0], token))


def lookup(row: Mapping[str, Any], path: str) -> Any:
    """Read a literal key first, or a dotted nested key (e.g. audio.path)."""
    if path in row:
        return row[path]
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def number(value: Any, field: str, *, integer: bool = False) -> float | int | None:
    """Accept missing numeric values, reject non-finite/negative measurements."""
    text = scalar(value)
    if text is None:
        return None
    result = float(text)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    if integer and (result <= 0 or not result.is_integer()):
        raise ValueError(f"{field} must be a positive integer")
    return int(result) if integer else result


COMMON_FIELDS = {
    "sample_id": ("sample_id", "id", "utterance_id"),
    "source_file": ("source_file", "file_name", "file", "path", "audio.path"),
    "speaker_id": ("speaker_id",),
    "language": ("language", "lang"),
    "gender": ("gender",),
    "duration": ("duration", "duration_seconds"),
    "sample_rate": ("sample_rate", "sampling_rate", "audio.sampling_rate"),
    "channels": ("channels", "num_channels"),
    "codec": ("codec",),
    "file_format": ("file_format",),
    "generator": ("generator",),
    "attack_type": ("attack_type",),
    "transcript": ("transcript", "text"),
    "age_group": ("age_group",),
    "processed_file": ("processed_file",),
    "source_split": ("split",),
}


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    label: str | None,
    fields: Mapping[str, tuple[str, ...]],
    field_map: Mapping[str, str] | None = None,
    language: str | None = None,
    namespace: str | None = None,
    locator_prefix: str | None = None,
) -> pd.DataFrame:
    """Translate source metadata into manifest-compatible pre-acquisition records.

    Explicit field_map entries override aliases. Missing identity needs a stable
    locator_prefix (revision/config/split/export) plus original row position.
    Durations are seconds. Speaker namespaces must reflect identity domains, not
    guesses about shared speakers. Source paths are opaque locators, never fetched.
    """
    mapping = dict(field_map or {})
    allowed = set(INSPECTION_COLUMNS) | {"label"}
    if set(mapping) - allowed:
        raise ValueError(f"Unknown canonical fields: {sorted(set(mapping) - allowed)}")
    result = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Row {index} must be an object")
        record: dict[str, Any] = dict.fromkeys(INSPECTION_COLUMNS)
        for dest, aliases in {**COMMON_FIELDS, **fields}.items():
            candidates = (mapping[dest],) if dest in mapping else aliases
            values = [lookup(raw, key) for key in candidates]
            record[dest] = next((v for v in values if scalar(v) is not None), None)
        for dest, key in mapping.items():
            record[dest] = lookup(raw, key)
        record["label"] = label if label is not None else scalar(record.get("label"))
        record["dataset"] = dataset
        record["language_raw"] = scalar(record["language"]) or scalar(language)
        record["language"] = normalize_language(record["language_raw"])
        record["speaker_id_raw"] = scalar(record["speaker_id"])
        for field in ("speaker_id", "source_speaker_id"):
            value = scalar(record[field])
            record[field] = f"{namespace or dataset}::{value}" if value is not None else None
        for field in ("duration", "sample_rate", "channels"):
            record[field] = number(record[field], field, integer=field != "duration")
        for field in INSPECTION_COLUMNS:
            if field not in {"duration", "sample_rate", "channels"}:
                record[field] = scalar(record[field])
        if record["gender"]:
            record["gender"] = {"m": "male", "f": "female"}.get(
                record["gender"].lower(), record["gender"].lower())
        if record["file_format"]:
            record["file_format"] = record["file_format"].lower().lstrip(".")
        elif record["source_file"]:
            suffix = PurePosixPath(urlsplit(record["source_file"].replace("\\", "/")).path).suffix
            record["file_format"] = suffix.lower().lstrip(".") or None
        record["metadata_locator"] = f"{locator_prefix}#row={index}" if locator_prefix else None
        identity = record["sample_id"] or record["source_file"] or record["metadata_locator"]
        if identity is None:
            raise ValueError(f"Row {index}: provide sample_id, source_file, or locator_prefix")
        if record["sample_id"] is None:
            record["sample_id"] = dataset + "__" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        record["split"] = "external" if dataset == "asvspoof" else "unassigned"
        record["role"] = "external_benchmark" if dataset == "asvspoof" else "train_dev"
        if record["label"] == "bonafide":
            record["generator"] = None
            record["attack_type"] = None
        result.append(record)
    frame = pd.DataFrame(result, columns=INSPECTION_COLUMNS)
    validate_manifest_frame(frame)
    if frame["label"].isna().any():
        raise ValueError("Every row requires a verified bonafide/spoof label")
    if frame.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Duplicate sample identities: use full locators or explicit stable IDs")
    return frame


def adapt(rows: Iterable[Mapping[str, Any]], dataset: str, **kwargs: Any) -> pd.DataFrame:
    """Dispatch a known source adapter without importing audio dependencies."""
    from .indicvoices import normalize as indicvoices
    from .indicsynth import normalize as indicsynth
    from .asvspoof import normalize as asvspoof

    adapters = {"indicvoices": indicvoices, "indicsynth": indicsynth, "asvspoof": asvspoof}
    if dataset not in adapters:
        raise ValueError(f"Unknown dataset: {dataset}")
    return adapters[dataset](rows, **kwargs)
