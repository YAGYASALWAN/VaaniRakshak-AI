"""IndicVoices-R metadata: verified lang, speaker_id, gender, age_group, duration."""
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from . import normalize_rows


def normalize(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> pd.DataFrame:
    """Preserve source measurements; do not assume the card's 48 kHz per row."""
    return normalize_rows(rows, dataset="indicvoices", label="bonafide", fields={}, **kwargs)
