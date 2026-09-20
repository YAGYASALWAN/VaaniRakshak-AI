"""Manifest manipulation tools for VaaniRakshak V2 experiments.

The most important helper builds an unseen-generator scenario without pretending
that generator-disjointness alone is sufficient. Records connected by known source
utterance or speaker identity are treated as one leakage component. If a held-out
component also contains spoof audio from a training generator, that conflicting
spoof variant is dropped rather than allowed to leak across the test boundary.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Iterable

from vaanirakshak.v2_data_contract import (
    AudioRecord,
    audit_manifest,
    audit_unseen_generator,
    load_jsonl,
    write_jsonl,
)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def identity_components(records: Iterable[AudioRecord]) -> list[list[AudioRecord]]:
    """Connect records sharing known speaker OR source utterance identity."""
    items = list(records)
    union = _UnionFind(len(items))
    owners: dict[tuple[str, str], int] = {}
    for index, item in enumerate(items):
        for field, value in (("speaker", item.speaker_id), ("source", item.source_utterance_id)):
            if not value:
                continue
            key = (field, value)
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index

    groups: dict[int, list[AudioRecord]] = {}
    for index, item in enumerate(items):
        groups.setdefault(union.find(index), []).append(item)
    return sorted(groups.values(), key=lambda group: min(item.record_id for item in group))


def _pool_audit(records: list[AudioRecord]) -> None:
    # Existing split labels are irrelevant while deriving a new scenario. Collapse
    # all records into one temporary split so the base audit still checks IDs,
    # hashes and required generator metadata without treating the old split as a
    # constraint on the new experimental split.
    view = [replace(item, split="train") for item in records]
    audit_manifest(view).raise_for_errors()


def _component_key(group: list[AudioRecord], seed: int) -> int:
    identity = "|".join(sorted(item.record_id for item in group))
    digest = hashlib.blake2s(f"{seed}:{identity}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _rebase_audio_ref(record: AudioRecord, source_root: Path, output_root: Path) -> AudioRecord:
    path = Path(record.audio_ref)
    if path.is_absolute():
        return record
    absolute = (source_root / path).resolve()
    relative = os.path.relpath(absolute, output_root.resolve())
    return replace(record, audio_ref=Path(relative).as_posix())


def build_unseen_generator_scenario(
    records: Iterable[AudioRecord],
    *,
    heldout_generators: set[str],
    dev_fraction: float = 0.10,
    test_bonafide_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[list[AudioRecord], dict]:
    """Create train/dev/test with spoof generators absent from train/dev.

    Held-out spoof records define test leakage components. Bonafide records in
    those components are retained in test. Non-held-out spoof variants in the same
    component are DROPPED: moving them to train would leak identity/content, while
    keeping them in test would violate a strict generator-disjoint evaluation.

    Additional bonafide-only components are moved to test until approximately the
    requested bonafide/spoof ratio is reached. Remaining identity components are
    deterministically assigned to train/dev.
    """
    items = list(records)
    if not items:
        raise ValueError("Cannot build scenario from an empty manifest")
    if not heldout_generators:
        raise ValueError("At least one held-out generator is required")
    if not 0.01 <= dev_fraction <= 0.40:
        raise ValueError("dev_fraction must be between 0.01 and 0.40")
    if not 0.0 <= test_bonafide_ratio <= 5.0:
        raise ValueError("test_bonafide_ratio must be between 0 and 5")

    _pool_audit(items)
    available = {item.generator_id for item in items if item.generator_id}
    missing = sorted(heldout_generators - available)
    if missing:
        raise ValueError(f"Held-out generators not present in manifest: {missing}")

    components = identity_components(items)
    heldout_components: set[int] = set()
    for index, group in enumerate(components):
        if any(item.label == "spoof" and item.generator_id in heldout_generators for item in group):
            heldout_components.add(index)

    test: list[AudioRecord] = []
    dropped: list[AudioRecord] = []
    remaining_components: list[list[AudioRecord]] = []
    for index, group in enumerate(components):
        if index not in heldout_components:
            remaining_components.append(group)
            continue
        for item in group:
            if item.label == "bonafide" or item.generator_id in heldout_generators:
                test.append(replace(item, split="test"))
            else:
                dropped.append(item)

    test_spoof = sum(item.label == "spoof" for item in test)
    if not test_spoof:
        raise ValueError("Held-out generators produced no spoof test records")

    target_bonafide = int(round(test_spoof * test_bonafide_ratio))
    current_bonafide = sum(item.label == "bonafide" for item in test)
    # Prefer bonafide-only components so adding genuine controls does not discard
    # otherwise useful spoof training examples. If insufficient, permit mixed
    # components but drop their spoof variants to keep generator separation strict.
    candidates = sorted(
        enumerate(remaining_components),
        key=lambda pair: (
            any(item.label == "spoof" for item in pair[1]),
            _component_key(pair[1], seed),
        ),
    )
    consumed_component_indices: set[int] = set()
    for component_index, group in candidates:
        if current_bonafide >= target_bonafide:
            break
        genuine = [item for item in group if item.label == "bonafide"]
        if not genuine:
            continue
        consumed_component_indices.add(component_index)
        for item in genuine:
            test.append(replace(item, split="test"))
        for item in group:
            if item.label == "spoof":
                dropped.append(item)
        current_bonafide += len(genuine)

    train_dev_components = [
        group for index, group in enumerate(remaining_components) if index not in consumed_component_indices
    ]
    train: list[AudioRecord] = []
    dev: list[AudioRecord] = []
    threshold = int(dev_fraction * 10_000)
    for group in train_dev_components:
        bucket = _component_key(group, seed) % 10_000
        destination = dev if bucket < threshold else train
        split = "dev" if destination is dev else "train"
        destination.extend(replace(item, split=split) for item in group)

    def labels(group: list[AudioRecord]) -> set[str]:
        return {item.label for item in group}

    # Large corpora naturally populate both labels in dev, but make the constraint
    # deterministic for smaller experiments by moving whole identity components.
    for required_label in ("bonafide", "spoof"):
        if required_label in labels(dev):
            continue
        movable = [
            group for group in train_dev_components
            if any(item.label == required_label for item in group)
            and all(any(candidate.record_id == item.record_id for candidate in train) for item in group)
        ]
        if not movable:
            raise ValueError(f"Unable to create development examples for label {required_label!r}")
        chosen = min(movable, key=lambda group: _component_key(group, seed ^ 0xA5A5))
        chosen_ids = {item.record_id for item in chosen}
        train = [item for item in train if item.record_id not in chosen_ids]
        dev.extend(replace(item, split="dev") for item in chosen)

    scenario = sorted(train + dev + test, key=lambda item: (item.split, item.record_id))
    audit = audit_unseen_generator(scenario)
    audit.raise_for_errors()
    if {item.label for item in test} != {"bonafide", "spoof"}:
        raise ValueError("Unseen-generator test could not retain both labels")

    report = {
        "heldout_generators": sorted(heldout_generators),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "test_bonafide_ratio": test_bonafide_ratio,
        "records": len(scenario),
        "dropped_records": len(dropped),
        "dropped_by_label": {
            "bonafide": sum(item.label == "bonafide" for item in dropped),
            "spoof": sum(item.label == "spoof" for item in dropped),
        },
        "counts": audit.counts,
        "reason_for_drops": "prevent source/speaker identity leakage or seen-generator contamination in strict test",
    }
    return scenario, report


def build_unseen_generator_file(
    source_manifest: Path,
    output_manifest: Path,
    *,
    heldout_generators: set[str],
    dev_fraction: float = 0.10,
    test_bonafide_ratio: float = 1.0,
    seed: int = 42,
) -> dict:
    source_manifest = source_manifest.resolve()
    output_manifest = output_manifest.resolve()
    records = load_jsonl(source_manifest)
    scenario, report = build_unseen_generator_scenario(
        records,
        heldout_generators=heldout_generators,
        dev_fraction=dev_fraction,
        test_bonafide_ratio=test_bonafide_ratio,
        seed=seed,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rebased = [
        _rebase_audio_ref(item, source_manifest.parent, output_manifest.parent)
        for item in scenario
    ]
    write_jsonl(output_manifest, rebased)
    return report
