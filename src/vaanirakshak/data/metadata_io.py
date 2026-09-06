"""Local bounded metadata readers. No dataset library or network side effects."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import adapt

MAX_METADATA_BYTES = 50 * 1024 * 1024
MAX_METADATA_ROWS = 100_000


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a small CSV/JSON/JSONL metadata export with a 50 MiB/100k-row guard.

    CSV strings retain leading zeros in identities. No audio columns are decoded.
    Larger exports require an explicit future scalable acquisition design.
    """
    source = Path(path)
    if source.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Metadata file exceeds 50 MiB safety bound")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        if source.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
        elif source.suffix.lower() == ".jsonl":
            reader = (json.loads(line) for line in handle if line.strip())
        elif source.suffix.lower() == ".json":
            reader = json.load(handle)
            if not isinstance(reader, list):
                raise ValueError("JSON export must contain an array of objects")
        else:
            raise ValueError("Use .csv, .json, or .jsonl metadata")
        rows = []
        for row in reader:
            if len(rows) >= MAX_METADATA_ROWS:
                raise ValueError("Metadata exceeds 100,000-row safety bound")
            if not isinstance(row, dict):
                raise ValueError("Metadata row must be an object")
            rows.append(row)
    return rows


def load_inputs(inputs: dict[str, Any], base: Path) -> pd.DataFrame:
    """Normalize exports configured per dataset; input paths resolve relative to YAML."""
    frames = []
    for dataset, specs in inputs.items():
        specs = specs if isinstance(specs, list) else [specs]
        for spec in specs:
            options = dict(spec)
            path = base / options.pop("path")
            frames.append(adapt(read_rows(path), dataset, **options))
    if not frames:
        raise ValueError("Configure at least one local metadata input")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Duplicate identities across metadata exports")
    return combined
