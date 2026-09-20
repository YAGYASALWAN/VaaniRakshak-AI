"""Materialize a pinned public anti-spoof benchmark into the V2 manifest format.

Supported benchmark configs are intentionally evaluation-only. They are never
merged into V2 training by this command. Audio is read with Hugging Face Audio
``decode=False`` so no extra decoder stack is required; the benchmark packages
already contain canonical 16 kHz mono FLAC.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from vaanirakshak.baseline_download import save_json
from vaanirakshak.v2_data_contract import AudioRecord, audit_manifest, write_jsonl


def _label(value) -> str:
    if value in (0, "0", "bonafide", "bona_fide", "real", "genuine"):
        return "bonafide"
    if value in (1, "1", "spoof", "fake", "synthetic"):
        return "spoof"
    raise ValueError(f"Unknown benchmark label: {value!r}")


def _notes(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Benchmark notes must be a JSON object or JSON object string")


def _audio_bytes(value) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes) and raw:
            return raw
        path = value.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    raise ValueError("Benchmark row did not expose audio bytes/path with decode=False")


def _generator_id(dataset_name: str, label: str, notes: dict) -> str | None:
    if label == "bonafide":
        return None
    attack = notes.get("attack_id") or notes.get("attack_condition")
    if attack is None or not str(attack).strip():
        raise ValueError("Spoof benchmark row lacks attack/generator identity")
    return f"{dataset_name}:{str(attack).strip()}"


def _source_identity(dataset_name: str, notes: dict) -> str | None:
    value = notes.get("source_id") or notes.get("utterance_id")
    if value is None or not str(value).strip():
        return None
    return f"{dataset_name}:{str(value).strip()}"


def _speaker_identity(dataset_name: str, notes: dict) -> str | None:
    value = notes.get("speaker_id")
    if value is None or not str(value).strip():
        return None
    return f"{dataset_name}:{str(value).strip()}"


def _safe_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not clean:
        clean = "record"
    if len(clean) > 120:
        clean = clean[:96] + "-" + hashlib.blake2s(value.encode(), digest_size=8).hexdigest()
    return clean


def prepare_benchmark(
    output: Path,
    config: dict,
    *,
    max_per_label: int | None = None,
) -> Path:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("Benchmark preparation requires the optional 'datasets' package") from exc

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = {"repository", "revision", "split", "dataset_name"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Benchmark config missing fields: {missing}")
    if max_per_label is not None and max_per_label < 1:
        raise ValueError("max_per_label must be positive or omitted for the full benchmark")

    repository = str(config["repository"])
    revision = str(config["revision"])
    split = str(config["split"])
    dataset_name = str(config["dataset_name"])

    stream = load_dataset(repository, split=split, revision=revision, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    audio_root = output / "audio" / dataset_name / "test"
    audio_root.mkdir(parents=True, exist_ok=True)
    records: list[AudioRecord] = []
    counts = Counter()
    seen_record_ids: set[str] = set()
    metadata_path = output / "source_metadata.jsonl"
    metadata_tmp = metadata_path.with_suffix(".jsonl.tmp")

    with metadata_tmp.open("w", encoding="utf-8", newline="\n") as metadata_stream:
        for row_number, row in enumerate(stream, 1):
            label = _label(row.get("label"))
            if max_per_label is not None and counts[label] >= max_per_label:
                if all(counts[name] >= max_per_label for name in ("bonafide", "spoof")):
                    break
                continue

            notes = _notes(row.get("notes"))
            path_value = str(row.get("path") or notes.get("utterance_id") or f"row-{row_number}")
            record_key = str(notes.get("utterance_id") or Path(path_value).stem or f"row-{row_number}")
            record_id = f"{dataset_name}:{record_key}"
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate benchmark record id: {record_id}")
            seen_record_ids.add(record_id)

            raw = _audio_bytes(row.get("audio"))
            digest = hashlib.sha256(raw).hexdigest()
            extension = Path(path_value).suffix.lower()
            if extension != ".flac":
                raise ValueError(f"Pinned benchmark unexpectedly contains non-FLAC audio: {path_value}")

            filename = _safe_filename(record_key) + ".flac"
            relative = Path("audio") / dataset_name / "test" / filename
            target = output / relative
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Existing benchmark audio differs for {record_id}")
            else:
                temporary = target.with_suffix(".flac.tmp")
                temporary.write_bytes(raw)
                temporary.replace(target)

            record = AudioRecord(
                record_id=record_id,
                label=label,  # type: ignore[arg-type]
                dataset=dataset_name,
                split="test",
                audio_ref=relative.as_posix(),
                language="en",
                speaker_id=_speaker_identity(dataset_name, notes),
                generator_id=_generator_id(dataset_name, label, notes),
                source_utterance_id=_source_identity(dataset_name, notes),
                content_sha256=digest,
                codec=str(notes.get("codec") or "FLAC"),
                sample_rate=16_000,
            )
            records.append(record)
            metadata_stream.write(
                json.dumps(
                    {
                        "record_id": record_id,
                        "source_path": path_value,
                        "notes": notes,
                        "sha256": digest,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            counts[label] += 1
            if len(records) % 1000 == 0:
                print(f"Materialized {len(records):,} {dataset_name} records", flush=True)
            if max_per_label is not None and all(counts[name] >= max_per_label for name in ("bonafide", "spoof")):
                break
    metadata_tmp.replace(metadata_path)

    if not records:
        raise ValueError("Benchmark materialization produced no records")
    if set(counts) != {"bonafide", "spoof"}:
        raise ValueError(f"Materialized benchmark lacks both labels: {dict(counts)}")

    audit = audit_manifest(records)
    audit.raise_for_errors()
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, records)
    save_json(
        output / "benchmark_receipt.json",
        {
            "repository": repository,
            "revision": revision,
            "split": split,
            "dataset_name": dataset_name,
            "license": config.get("license"),
            "role": config.get("role"),
            "records": len(records),
            "counts": dict(counts),
            "max_per_label": max_per_label,
            "sampling_notice": (
                "FULL PINNED BENCHMARK" if max_per_label is None
                else "BALANCED PREFIX SMOKE SUBSET ONLY; do not report as full benchmark performance"
            ),
            "audit_counts": audit.counts,
            "source_metadata": metadata_path.name,
        },
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_external_benchmarks.json"))
    parser.add_argument("--max-per-label", type=int)
    args = parser.parse_args()

    configs = json.loads(args.config.read_text(encoding="utf-8"))
    if args.benchmark not in configs:
        raise SystemExit(f"Unknown benchmark {args.benchmark!r}; choices: {', '.join(sorted(configs))}")
    manifest = prepare_benchmark(args.output, configs[args.benchmark], max_per_label=args.max_per_label)
    print(f"V2 benchmark manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
