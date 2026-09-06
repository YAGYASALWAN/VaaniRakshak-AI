"""IndicSynth publisher-preview field mapping; generator values stay verbatim."""
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from . import normalize_rows

FIELDS = {
    "speaker_id": ("Target Speaker ID", "target_speaker", "target_speaker_id", "speaker_id"),
    "source_speaker_id": ("Source Speaker_ID", "source_speaker_id"),
    "source_reference": ("Source Reference Audio", "source_reference"),
    "target_reference": ("Target Reference Audio", "target_reference"),
    "generator": ("Generative Model", "generator", "synthesis_system"),
    "gender": ("Gender", "gender"),
    "transcript": ("TTS Transcript", "transcript"),
}


def normalize(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> pd.DataFrame:
    """Use target identity for speaker_id; never mistake reference audio for spoof audio."""
    return normalize_rows(rows, dataset="indicsynth", label="spoof", fields=FIELDS, **kwargs)
