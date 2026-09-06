"""ASVspoof2021_DF mirror adapter, permanently marked external."""
import json
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from . import normalize_rows


def normalize(
    rows: Iterable[Mapping[str, Any]], *,
    label_map: Mapping[str, str] | None = None, **kwargs: Any,
) -> pd.DataFrame:
    """Parse notes JSON; numeric labels require an explicit verified label_map.

    Attack ID is retained as attack_type; vocoder family is retained as generator,
    not presented as an exact synthesis-system identity.
    """
    expanded = []
    for raw in rows:
        row = dict(raw)
        notes = row.get("notes")
        if isinstance(notes, str):
            notes = json.loads(notes)
        if notes is not None and not isinstance(notes, Mapping):
            raise ValueError("ASVspoof notes must be a JSON object")
        row["notes"] = notes or {}
        if label_map and str(row.get("label")) in label_map:
            row["label"] = label_map[str(row["label"])]
        expanded.append(row)
    return normalize_rows(
        expanded, dataset="asvspoof", label=None,
        fields={"sample_id": ("sample_id", "notes.utterance_id"),
                "label": ("label",), "speaker_id": ("speaker_id", "notes.speaker_id"),
                "codec": ("codec", "notes.codec"),
                "attack_type": ("attack_id", "notes.attack_id"),
                "generator": ("generator", "notes.vocoder")}, **kwargs,
    )
