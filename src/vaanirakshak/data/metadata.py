"""Unified manifest schema for every dataset source.

Labels are always the strings ``bonafide`` and ``spoof``. Numeric 0/1 mapping
belongs in the training dataloader later, not in dataset engineering.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from vaanirakshak.exceptions import ManifestSchemaError

logger = logging.getLogger(__name__)

ALLOWED_LABELS = frozenset({"bonafide", "spoof"})

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_file",
    "processed_file",
    "dataset",
    "label",
    "speaker_id",
    "language",
    "gender",
    "duration",
    "sample_rate",
    "segment_start",
    "segment_end",
    "generator",
    "attack_type",
    "codec",
    "split",
)


@dataclass
class ManifestRecord:
    sample_id: str
    source_file: str
    processed_file: str
    dataset: str
    label: str
    speaker_id: str | None
    language: str | None
    gender: str | None
    duration: float
    sample_rate: int
    segment_start: float
    segment_end: float
    generator: str | None
    attack_type: str | None
    codec: str | None
    split: str

    def __post_init__(self) -> None:
        if self.label not in ALLOWED_LABELS:
            raise ManifestSchemaError(
                f"label must be one of {sorted(ALLOWED_LABELS)}, got {self.label!r}"
            )
        if not self.sample_id:
            raise ManifestSchemaError("sample_id must be non-empty")
        if not self.dataset:
            raise ManifestSchemaError("dataset must be non-empty")
        if self.sample_rate <= 0:
            raise ManifestSchemaError("sample_rate must be positive")
        if self.duration < 0:
            raise ManifestSchemaError("duration must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_sample_id(
    dataset: str,
    source_stem: str,
    segment_start: float,
    segment_end: float,
) -> str:
    """Stable id from dataset, filename stem, and window times (not a random UUID)."""
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source_stem)
    return f"{dataset}__{safe_stem}__{segment_start:.3f}_{segment_end:.3f}"


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=list(MANIFEST_COLUMNS))


def records_to_frame(records: list[ManifestRecord]) -> pd.DataFrame:
    if not records:
        return empty_manifest()
    frame = pd.DataFrame([record.to_dict() for record in records])
    return frame.loc[:, list(MANIFEST_COLUMNS)]


def validate_manifest_frame(frame: pd.DataFrame) -> None:
    """Raise if required columns or allowed labels are missing.

    Nullable fields may be NA. ``label`` and ``split`` may not.
    """
    missing = [col for col in MANIFEST_COLUMNS if col not in frame.columns]
    if missing:
        raise ManifestSchemaError(f"Manifest missing columns: {missing}")
    if frame.empty:
        return
    bad_labels = set(frame["label"].dropna().unique()) - ALLOWED_LABELS
    if bad_labels:
        raise ManifestSchemaError(
            f"Illegal labels {sorted(bad_labels)}; use only {sorted(ALLOWED_LABELS)}"
        )


def append_manifest(path: str | Path, records: list[ManifestRecord]) -> Path:
    """Append rows to a CSV, creating a header if the file is new.

    Inputs
        path: destination CSV under ``data/manifests`` (not raw audio).
        records: one row per processed window.
    Outputs
        Path written.
    Edge cases
        Empty record list is a no-op. Existing files are concatenated.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.info("No manifest rows to write for %s", dest)
        return dest

    new_frame = records_to_frame(records)
    validate_manifest_frame(new_frame)
    if dest.is_file():
        existing = pd.read_csv(dest)
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    validate_manifest_frame(combined)
    combined.to_csv(dest, index=False)
    logger.info("Wrote %s manifest rows to %s", len(records), dest)
    return dest
