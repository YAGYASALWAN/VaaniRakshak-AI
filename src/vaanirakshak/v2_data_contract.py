"""Dataset/evaluation contracts for VaaniRakshak V2.

V1 taught us that a detector can obtain attractive metrics while learning dataset
provenance rather than synthetic-speech artefacts. V2 therefore makes dataset
identity, generator identity and split intent explicit before training begins.

The manifest format is JSONL: one JSON object per audio item. This module contains
no downloader and does not assume any particular dataset provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal


Label = Literal["bonafide", "spoof"]
Split = Literal["train", "dev", "test"]


@dataclass(frozen=True)
class AudioRecord:
    record_id: str
    label: Label
    dataset: str
    split: Split
    audio_ref: str
    language: str | None = None
    speaker_id: str | None = None
    generator_id: str | None = None
    source_utterance_id: str | None = None
    content_sha256: str | None = None
    codec: str | None = None
    sample_rate: int | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "AudioRecord":
        required = {"record_id", "label", "dataset", "split", "audio_ref"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Manifest record missing required fields: {missing}")

        label = str(value["label"]).strip().lower()
        split = str(value["split"]).strip().lower()
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"Unsupported label: {label!r}")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")

        record_id = str(value["record_id"]).strip()
        dataset = str(value["dataset"]).strip()
        audio_ref = str(value["audio_ref"]).strip()
        if not record_id or not dataset or not audio_ref:
            raise ValueError("record_id, dataset and audio_ref must be nonempty")

        generator_id = _optional_text(value.get("generator_id"))
        if label == "bonafide" and generator_id is not None:
            raise ValueError("Bonafide records must not carry generator_id")

        sample_rate = value.get("sample_rate")
        if sample_rate is not None:
            sample_rate = int(sample_rate)
            if not 4_000 <= sample_rate <= 192_000:
                raise ValueError("sample_rate is outside a plausible audio range")

        content_sha256 = _optional_text(value.get("content_sha256"))
        if content_sha256 is not None:
            if len(content_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in content_sha256.lower()):
                raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
            content_sha256 = content_sha256.lower()

        return cls(
            record_id=record_id,
            label=label,  # type: ignore[arg-type]
            dataset=dataset,
            split=split,  # type: ignore[arg-type]
            audio_ref=audio_ref,
            language=_optional_text(value.get("language")),
            speaker_id=_optional_text(value.get("speaker_id")),
            generator_id=generator_id,
            source_utterance_id=_optional_text(value.get("source_utterance_id")),
            content_sha256=content_sha256,
            codec=_optional_text(value.get("codec")),
            sample_rate=sample_rate,
        )

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "label": self.label,
            "dataset": self.dataset,
            "split": self.split,
            "audio_ref": self.audio_ref,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "generator_id": self.generator_id,
            "source_utterance_id": self.source_utterance_id,
            "content_sha256": self.content_sha256,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
        }


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_jsonl(path: str | Path) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} must contain a JSON object")
            try:
                records.append(AudioRecord.from_dict(value))
            except ValueError as exc:
                raise ValueError(f"Manifest line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Manifest is empty")
    return records


def write_jsonl(path: str | Path, records: Iterable[AudioRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")


def manifest_fingerprint(records: Iterable[AudioRecord]) -> str:
    payload = "\n".join(
        json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: item.record_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Dataset contract failed:\n- " + "\n- ".join(self.errors))


def audit_manifest(
    records: Iterable[AudioRecord],
    *,
    require_generator_for_spoof: bool = True,
    require_content_hash: bool = True,
) -> AuditResult:
    items = list(records)
    errors: list[str] = []
    warnings: list[str] = []

    if not items:
        errors.append("manifest contains no records")
        return AuditResult(tuple(errors), tuple(warnings), {})

    ids: dict[str, AudioRecord] = {}
    audio_refs: dict[str, AudioRecord] = {}
    hashes: dict[str, list[AudioRecord]] = {}
    source_ids: dict[str, list[AudioRecord]] = {}
    speakers: dict[str, list[AudioRecord]] = {}

    for item in items:
        if item.record_id in ids:
            errors.append(f"duplicate record_id {item.record_id!r}")
        ids[item.record_id] = item

        if item.audio_ref in audio_refs:
            other = audio_refs[item.audio_ref]
            errors.append(f"audio_ref reused by {other.record_id!r} and {item.record_id!r}")
        audio_refs[item.audio_ref] = item

        if require_generator_for_spoof and item.label == "spoof" and not item.generator_id:
            errors.append(f"spoof record {item.record_id!r} lacks generator_id")

        if require_content_hash and not item.content_sha256:
            errors.append(f"record {item.record_id!r} lacks content_sha256")

        if item.content_sha256:
            hashes.setdefault(item.content_sha256, []).append(item)
        if item.source_utterance_id:
            source_ids.setdefault(item.source_utterance_id, []).append(item)
        if item.speaker_id:
            speakers.setdefault(item.speaker_id, []).append(item)

    for digest, group in hashes.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            errors.append(f"identical audio hash {digest[:12]}… appears across splits {sorted(splits)}")

    # A source utterance may legitimately have one human and multiple generated
    # versions, but those related variants must stay in one split to prevent
    # transcript/content leakage into evaluation.
    for source_id, group in source_ids.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            errors.append(f"source_utterance_id {source_id!r} crosses splits {sorted(splits)}")

    # When a source supplies a speaker/voice identity, keep it split-disjoint. We
    # do not invent IDs where upstream metadata is absent, but known identities
    # must never be knowingly shared between training and evaluation.
    for speaker_id, group in speakers.items():
        splits = {item.split for item in group}
        if len(splits) > 1:
            errors.append(f"speaker_id {speaker_id!r} crosses splits {sorted(splits)}")

    for split in ("train", "dev", "test"):
        subset = [item for item in items if item.split == split]
        if not subset:
            warnings.append(f"split {split!r} is empty")
            continue
        labels = {item.label for item in subset}
        if labels != {"bonafide", "spoof"}:
            warnings.append(f"split {split!r} does not contain both labels: {sorted(labels)}")

    counts = {
        "records": len(items),
        "by_split": {split: sum(item.split == split for item in items) for split in ("train", "dev", "test")},
        "by_label": {label: sum(item.label == label for item in items) for label in ("bonafide", "spoof")},
        "datasets": sorted({item.dataset for item in items}),
        "spoof_generators": sorted({item.generator_id for item in items if item.generator_id}),
        "speaker_ids_present": sum(bool(item.speaker_id) for item in items),
        "source_utterance_ids_present": sum(bool(item.source_utterance_id) for item in items),
    }
    return AuditResult(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), counts)


def audit_unseen_generator(records: Iterable[AudioRecord]) -> AuditResult:
    items = list(records)
    base = audit_manifest(items)
    errors = list(base.errors)
    warnings = list(base.warnings)

    train_generators = {
        item.generator_id
        for item in items
        if item.split in {"train", "dev"} and item.label == "spoof" and item.generator_id
    }
    test_generators = {
        item.generator_id
        for item in items
        if item.split == "test" and item.label == "spoof" and item.generator_id
    }
    overlap = sorted(train_generators & test_generators)
    if overlap:
        errors.append(f"unseen-generator test is contaminated by train/dev generators: {overlap}")
    if not test_generators:
        errors.append("unseen-generator test has no spoof generator identities")

    return AuditResult(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), base.counts)


def audit_cross_dataset(
    records: Iterable[AudioRecord],
    *,
    evaluation_datasets: set[str],
) -> AuditResult:
    items = list(records)
    base = audit_manifest(items)
    errors = list(base.errors)
    warnings = list(base.warnings)

    if not evaluation_datasets:
        errors.append("cross-dataset audit requires at least one evaluation dataset")
        return AuditResult(tuple(errors), tuple(warnings), base.counts)

    train_datasets = {item.dataset for item in items if item.split in {"train", "dev"}}
    test_datasets = {item.dataset for item in items if item.split == "test"}

    leaked = sorted(train_datasets & evaluation_datasets)
    if leaked:
        errors.append(f"evaluation datasets also appear in train/dev: {leaked}")

    missing = sorted(evaluation_datasets - test_datasets)
    if missing:
        errors.append(f"declared evaluation datasets missing from test split: {missing}")

    unexpected = sorted(test_datasets - evaluation_datasets)
    if unexpected:
        warnings.append(f"test split also contains undeclared datasets: {unexpected}")

    return AuditResult(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), base.counts)
