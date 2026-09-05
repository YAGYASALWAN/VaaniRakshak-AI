"""Leakage-aware split helpers.

Randomly shuffling *segments* would put the same speaker (and often the same
utterance, via overlapping windows) into train and test. These helpers split
on a grouping key first, then assign every row that shares that key together.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from voiceshield.config import SplitConfig
from voiceshield.data.metadata import MANIFEST_COLUMNS, validate_manifest_frame
from voiceshield.exceptions import SplitError

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "dev", "test")


def _group_key(frame: pd.DataFrame, speaker_column: str, fallback_column: str) -> pd.Series:
    speakers = frame[speaker_column]
    fallback = frame[fallback_column].astype(str)
    missing = speakers.isna() | (speakers.astype(str).str.strip() == "") | (
        speakers.astype(str) == "nan"
    )
    keys = speakers.astype(str).copy()
    keys[missing] = fallback[missing]
    return keys


def _assign_groups_to_splits(
    groups: Sequence[str],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique = list(dict.fromkeys(groups))
    if not unique:
        raise SplitError("No groups available to split")

    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    rng.shuffle(order)
    shuffled = [unique[i] for i in order]

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_dev = int(round(n * dev_ratio))
    n_test = n - n_train - n_dev
    if n_test < 0:
        n_dev += n_test
        n_test = 0
    # Guarantee each requested non-zero ratio gets at least one group when possible.
    counts = [n_train, n_dev, n_test]
    for i, ratio in enumerate((train_ratio, dev_ratio, test_ratio)):
        if ratio > 0 and n >= 3 and counts[i] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[i] += 1
    n_train, n_dev, n_test = counts
    if n_train + n_dev + n_test != n:
        n_test = n - n_train - n_dev

    assignment: dict[str, str] = {}
    idx = 0
    for split, count in zip(SPLIT_NAMES, (n_train, n_dev, n_test)):
        for group in shuffled[idx : idx + count]:
            assignment[group] = split
        idx += count
    return assignment


def speaker_disjoint_split(
    frame: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
    speaker_column: str = "speaker_id",
    fallback_column: str = "source_file",
    seed: int | None = None,
) -> pd.DataFrame:
    """Assign train/dev/test so no speaker key appears in two splits.

    Formally, after assignment:
        train_speakers ∩ dev_speakers = ∅
        train_speakers ∩ test_speakers = ∅
        dev_speakers ∩ test_speakers = ∅

    Inputs
        frame: manifest-like table.
        speaker_column: grouping key. Empty speaker ids fall back to
        ``source_file`` so overlapping segments of one file stay together.
        seed: overrides config seed; same seed => same assignment.
    Outputs
        Copy of ``frame`` with a ``split`` column.
    Edge cases
        Fewer groups than splits: some splits may be empty (logged).
        Missing speaker_id is not silently treated as one giant speaker.
    """
    validate_manifest_frame(frame)
    cfg = config or SplitConfig()
    use_seed = cfg.seed if seed is None else seed
    work = frame.copy()
    keys = _group_key(work, speaker_column, fallback_column)
    assignment = _assign_groups_to_splits(
        keys.tolist(),
        train_ratio=cfg.train_ratio,
        dev_ratio=cfg.dev_ratio,
        test_ratio=cfg.test_ratio,
        seed=use_seed,
    )
    work["split"] = keys.map(assignment)

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        a = set(keys[work["split"] == left])
        b = set(keys[work["split"] == right])
        overlap = a & b
        if overlap:
            raise SplitError(f"Speaker overlap between {left} and {right}: {sorted(overlap)[:5]}")

    logger.info(
        "Speaker-disjoint split (seed=%s): train=%s dev=%s test=%s groups=%s",
        use_seed,
        int((work["split"] == "train").sum()),
        int((work["split"] == "dev").sum()),
        int((work["split"] == "test").sum()),
        len(assignment),
    )
    return work.loc[:, list(MANIFEST_COLUMNS)]


def generator_disjoint_split(
    frame: pd.DataFrame,
    *,
    train_generators: Iterable[str],
    test_generators: Iterable[str],
    dev_generators: Iterable[str] | None = None,
    generator_column: str = "generator",
) -> pd.DataFrame:
    """Put named synthesis systems into designated splits (no random mix).

    Example: train on XTTS+VITS, test on FreeVC. Bona fide rows have a null
    generator; they are left unchanged unless you filter them separately.

    Inputs
        train_generators / test_generators / optional dev_generators.
    Outputs
        Copy with ``split`` overwritten for rows whose generator is listed.
    Edge cases
        A generator listed in two splits raises SplitError.
        Unknown generators are left as-is and logged.
    """
    validate_manifest_frame(frame)
    train_set = {str(g) for g in train_generators}
    test_set = {str(g) for g in test_generators}
    dev_set = {str(g) for g in (dev_generators or [])}
    if train_set & test_set or train_set & dev_set or test_set & dev_set:
        raise SplitError("A generator cannot be assigned to more than one split")

    work = frame.copy()
    gen = work[generator_column].astype("string")
    work.loc[gen.isin(train_set), "split"] = "train"
    work.loc[gen.isin(dev_set), "split"] = "dev"
    work.loc[gen.isin(test_set), "split"] = "test"

    seen = set(gen.dropna().astype(str))
    assigned = train_set | test_set | dev_set
    leftover = seen - assigned
    if leftover:
        logger.warning("Generators not assigned to a split: %s", sorted(leftover))
    return work.loc[:, list(MANIFEST_COLUMNS)]
