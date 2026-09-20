"""Prepare a leakage-audited English MLAAD-tiny manifest for VaaniRakshak V2.

MLAAD-tiny already contains local audio, so this command does not download or copy
large files. It verifies the English bona-fide and spoof trees, reads every spoof
``meta.csv``, hashes the audio, derives generator/source provenance, then creates a
deterministic leakage-safe train/dev/test manifest.

Important: MLAAD-tiny is a sampled subset. A spoof row's ``original_file`` may refer
to an M-AILABS source utterance whose genuine WAV is not included in tiny's sampled
``original/en`` tree. That is valid provenance, not an error. We still use the
reference as a source-utterance identity so related generated variants stay in the
same split; when the corresponding genuine WAV is present, it naturally joins the
same identity component as well.

MLAAD generator folders may expose slightly different metadata columns. Only
``path``, ``original_file`` and ``language`` are mandatory here. ``model_name`` is
optional because the generator directory is a reliable fallback, and
``reference_speaker`` is optional provenance rather than a dataset requirement.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from vaanirakshak.v2_data_contract import AudioRecord, audit_manifest, write_jsonl
from vaanirakshak.v2_manifest_tools import identity_components


DATASET = "MLAAD-tiny"
SCHEMA = "vaanirakshak-v2-mlaad-tiny-v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _clean_relative(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe MLAAD relative path: {value!r}")
    return path.as_posix()


def _normalize_original_reference(value: str) -> str:
    """Return the canonical path under ``original/en`` used for source identity."""
    relative = _clean_relative(value)
    prefix = "original/en/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    if not relative:
        raise ValueError(f"Empty MLAAD original-file reference: {value!r}")
    return _clean_relative(relative)


def _resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"MLAAD path escapes dataset root: {relative!r}") from exc
    return path


def _source_id(relative_original: str) -> str:
    return f"MLAAD-tiny:{_normalize_original_reference(relative_original)}"


def _record_id(label: str, relative: str) -> str:
    digest = hashlib.blake2s(relative.encode("utf-8"), digest_size=12).hexdigest()
    return f"MLAAD-tiny:{label}:{digest}"


def _manifest_audio_ref(audio_path: Path, output_root: Path) -> str:
    return Path(os.path.relpath(audio_path.resolve(), output_root.resolve())).as_posix()


def _generator(row: dict, meta_path: Path) -> str:
    value = str(row.get("model_name") or "").strip()
    if not value:
        value = meta_path.parent.name.strip()
    if not value:
        raise ValueError(f"Spoof row in {meta_path} has no generator identity")
    return f"MLAAD:{value}"


def _speaker(row: dict) -> str | None:
    value = str(row.get("reference_speaker") or "").strip()
    if not value or value.lower() in {"none", "null", "unknown", "nan"}:
        return None
    return f"MLAAD:reference:{value}"


def _validate_metadata_schema(fieldnames: list[str] | None, meta_path: Path) -> set[str]:
    if fieldnames is None:
        raise ValueError(f"MLAAD metadata has no header: {meta_path}")
    fields = {str(value).strip() for value in fieldnames if str(value).strip()}
    required = {"path", "original_file", "language"}
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"MLAAD metadata in {meta_path} is missing required columns: {missing}; fields={sorted(fields)}")
    return fields


def _component_bucket(group: list[AudioRecord], seed: int) -> int:
    identity = "|".join(sorted(item.record_id for item in group))
    digest = hashlib.blake2s(f"{seed}:{identity}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 10_000


def split_records(
    records: list[AudioRecord],
    *,
    dev_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> list[AudioRecord]:
    """Assign identity-connected components to deterministic train/dev/test splits."""
    if not 0.01 <= dev_fraction <= 0.40:
        raise ValueError("dev_fraction must be between 0.01 and 0.40")
    if not 0.01 <= test_fraction <= 0.40:
        raise ValueError("test_fraction must be between 0.01 and 0.40")
    if dev_fraction + test_fraction >= 0.80:
        raise ValueError("dev_fraction + test_fraction must leave at least 20% for train")

    provisional = [replace(item, split="train") for item in records]
    components = identity_components(provisional)
    dev_cut = int(dev_fraction * 10_000)
    test_cut = dev_cut + int(test_fraction * 10_000)

    result: list[AudioRecord] = []
    for group in components:
        bucket = _component_bucket(group, seed)
        if bucket < dev_cut:
            split = "dev"
        elif bucket < test_cut:
            split = "test"
        else:
            split = "train"
        result.extend(replace(item, split=split) for item in group)

    result.sort(key=lambda item: (item.split, item.record_id))
    audit = audit_manifest(result)
    audit.raise_for_errors()
    missing = []
    for split in ("train", "dev", "test"):
        labels = {item.label for item in result if item.split == split}
        if labels != {"bonafide", "spoof"}:
            missing.append(f"{split}={sorted(labels)}")
    if missing:
        raise ValueError("Deterministic MLAAD-tiny split did not retain both labels: " + ", ".join(missing))
    return result


def prepare(
    source_root: Path,
    output_root: Path,
    *,
    dev_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    original_root = source_root / "original" / "en"
    fake_root = source_root / "fake" / "en"
    if not original_root.is_dir() or not fake_root.is_dir():
        raise ValueError("Expected MLAAD-tiny English trees at original/en and fake/en")

    output_root.mkdir(parents=True, exist_ok=True)

    originals = sorted(path for path in original_root.rglob("*.wav") if path.is_file())
    meta_files = sorted(path for path in fake_root.rglob("meta.csv") if path.is_file())
    fake_files = sorted(path for path in fake_root.rglob("*.wav") if path.is_file())
    if not originals:
        raise ValueError("MLAAD-tiny contains no English bona-fide WAV files")
    if not fake_files or not meta_files:
        raise ValueError("MLAAD-tiny contains no English spoof WAV/metadata files")

    print(
        f"MLAAD-tiny scan: bona-fide={len(originals)}, spoof={len(fake_files)}, metadata_files={len(meta_files)}",
        flush=True,
    )

    records: list[AudioRecord] = []
    known_originals: dict[str, Path] = {}
    for ordinal, path in enumerate(originals, 1):
        dataset_relative = _clean_relative(path.relative_to(source_root).as_posix())
        source_key = _normalize_original_reference(path.relative_to(original_root).as_posix())
        if source_key in known_originals:
            raise ValueError(f"Duplicate MLAAD original identity: {source_key}")
        known_originals[source_key] = path
        records.append(
            AudioRecord(
                record_id=_record_id("bonafide", dataset_relative),
                label="bonafide",
                dataset=DATASET,
                split="train",
                audio_ref=_manifest_audio_ref(path, output_root),
                language="en",
                speaker_id=None,
                generator_id=None,
                source_utterance_id=_source_id(source_key),
                content_sha256=_sha256(path),
                codec="WAV",
                sample_rate=None,
            )
        )
        if ordinal % 1000 == 0 or ordinal == len(originals):
            print(f"Hashed bona-fide audio {ordinal}/{len(originals)}", flush=True)

    metadata_audio: set[Path] = set()
    metadata_rows = 0
    spoof_hashed = 0
    spoof_with_local_original = 0
    spoof_without_local_original = 0
    metadata_with_reference_speaker = 0
    metadata_without_reference_speaker = 0
    metadata_with_model_name = 0
    metadata_without_model_name = 0

    for meta_ordinal, meta_path in enumerate(meta_files, 1):
        with meta_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="|")
            fields = _validate_metadata_schema(reader.fieldnames, meta_path)
            if "reference_speaker" in fields:
                metadata_with_reference_speaker += 1
            else:
                metadata_without_reference_speaker += 1
            if "model_name" in fields:
                metadata_with_model_name += 1
            else:
                metadata_without_model_name += 1

            for row in reader:
                metadata_rows += 1
                if str(row.get("language") or "").strip().lower() != "en":
                    raise ValueError(f"Non-English row found under MLAAD-tiny fake/en: {meta_path}")

                fake_relative = _clean_relative(row["path"])
                original_key = _normalize_original_reference(row["original_file"])
                fake_path = _resolve_under(source_root, fake_relative)
                if not fake_path.is_file():
                    raise FileNotFoundError(f"MLAAD spoof file referenced by metadata is missing: {fake_path}")
                if fake_path in metadata_audio:
                    raise ValueError(f"MLAAD spoof audio appears more than once in metadata: {fake_relative}")
                metadata_audio.add(fake_path)

                if original_key in known_originals:
                    spoof_with_local_original += 1
                else:
                    spoof_without_local_original += 1

                records.append(
                    AudioRecord(
                        record_id=_record_id("spoof", fake_relative),
                        label="spoof",
                        dataset=DATASET,
                        split="train",
                        audio_ref=_manifest_audio_ref(fake_path, output_root),
                        language="en",
                        speaker_id=_speaker(row),
                        generator_id=_generator(row, meta_path),
                        source_utterance_id=_source_id(original_key),
                        content_sha256=_sha256(fake_path),
                        codec="WAV",
                        sample_rate=None,
                    )
                )
                spoof_hashed += 1
                if spoof_hashed % 1000 == 0 or spoof_hashed == len(fake_files):
                    print(f"Hashed spoof audio {spoof_hashed}/{len(fake_files)}", flush=True)
        if meta_ordinal % 10 == 0 or meta_ordinal == len(meta_files):
            print(f"Read spoof metadata {meta_ordinal}/{len(meta_files)} files", flush=True)

    discovered_fake = {path.resolve() for path in fake_files}
    if metadata_audio != discovered_fake:
        untracked = sorted(path.as_posix() for path in discovered_fake - metadata_audio)
        missing = sorted(path.as_posix() for path in metadata_audio - discovered_fake)
        raise ValueError(
            "MLAAD spoof metadata/audio coverage mismatch: "
            f"untracked_audio={len(untracked)}, missing_audio={len(missing)}"
        )

    records = split_records(
        records,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    audit = audit_manifest(records)
    audit.raise_for_errors()

    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    report = {
        "schema": SCHEMA,
        "source_root": str(source_root),
        "records": len(records),
        "original_wav_files": len(originals),
        "spoof_wav_files": len(fake_files),
        "metadata_files": len(meta_files),
        "metadata_rows": metadata_rows,
        "metadata_files_with_reference_speaker": metadata_with_reference_speaker,
        "metadata_files_without_reference_speaker": metadata_without_reference_speaker,
        "metadata_files_with_model_name": metadata_with_model_name,
        "metadata_files_without_model_name": metadata_without_model_name,
        "spoof_rows_with_local_original": spoof_with_local_original,
        "spoof_rows_without_local_original": spoof_without_local_original,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "test_fraction": test_fraction,
        "counts": audit.counts,
        "warnings": list(audit.warnings),
        "split_policy": "identity components sharing known source utterance or available reference speaker stay in one split",
        "source_utterance_policy": (
            "fake original_file is retained as source identity even when MLAAD-tiny does not include the corresponding "
            "sampled genuine WAV; when present, genuine and generated variants are joined automatically"
        ),
        "generator_policy": "model_name when available, otherwise generator-directory fallback",
        "speaker_policy": "reference_speaker is used when supplied by that generator metadata; absence is not invented",
        "audio_storage": "existing MLAAD-tiny WAV files are referenced in place; no duplicate audio copy is created",
        "license_note": "Respect MLAAD-tiny upstream license and non-commercial restrictions for the SIH prototype.",
    }
    (output_root / "mlaad_tiny_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "Prepared MLAAD-tiny: "
        f"train={audit.counts['by_split']['train']}, "
        f"dev={audit.counts['by_split']['dev']}, "
        f"test={audit.counts['by_split']['test']}, "
        f"spoof_sources_present={spoof_with_local_original}, "
        f"spoof_sources_absent={spoof_without_local_original}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Root of the local MLAAD-tiny Git/LFS clone")
    parser.add_argument("--output", type=Path, required=True, help="Directory for manifest and audit JSON")
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = prepare(
        args.source,
        args.output,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"V2 MLAAD-tiny manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
