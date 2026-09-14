"""Prepare a leakage-audited raw-audio SEA-Spoof subset for VaaniRakshak V2.

This module reuses the pinned SEA-Spoof source and byte-budgeted range reader from
V1, but does NOT create log-mel features. Selected English audio is decoded and
canonicalized to mono 16 kHz FLAC, then described by the V2 JSONL manifest.

Canonicalizing storage removes trivial source codec/sample-rate differences before
WavLM training while retaining source provenance in a sidecar receipt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
import torch
import torchaudio

from vaanirakshak.baseline_download import save_json
from vaanirakshak.sea_training import select_groups
from vaanirakshak.sea_transfer import MAX_BYTES, REPOSITORY, REVISION, RangeFile, Transfer
from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE
from vaanirakshak.v2_data_contract import AudioRecord, audit_manifest, write_jsonl


SCHEMA = "vaanirakshak-v2-sea-raw-v1"
SOURCE_SPLITS = {"train": "train", "validation": "dev", "evaluation": "test"}


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_id(value: str) -> str:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=10).hexdigest()
    return digest


def _canonical_flac(raw: bytes) -> tuple[bytes, dict]:
    with sf.SoundFile(io.BytesIO(raw)) as stream:
        if not 4_000 <= stream.samplerate <= 192_000:
            raise ValueError("SEA audio sample rate outside supported range")
        if not 1 <= stream.channels <= 8:
            raise ValueError("SEA audio channel count outside supported range")
        wave = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        source_rate = int(stream.samplerate)
        source_channels = int(stream.channels)
        source_format = str(stream.format)
        source_subtype = str(stream.subtype)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("SEA audio contains no finite samples")

    tensor = torch.from_numpy(wave.astype(np.float32, copy=False))
    if source_rate != MODEL_SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, source_rate, MODEL_SAMPLE_RATE)
    canonical = tensor.float().clamp(-1.0, 1.0).cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, canonical, MODEL_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    encoded = buffer.getvalue()
    if not encoded:
        raise ValueError("Failed to encode canonical FLAC")
    return encoded, {
        "source_sample_rate": source_rate,
        "source_channels": source_channels,
        "source_format": source_format,
        "source_subtype": source_subtype,
        "canonical_sample_rate": MODEL_SAMPLE_RATE,
        "canonical_channels": 1,
        "canonical_format": "FLAC/PCM_16",
        "canonical_frames": int(len(canonical)),
    }


def _scan_groups(root: Path, source: dict, client: Transfer, budget: int) -> dict:
    import pyarrow.parquet as pq

    path = root / "v2_sea_plan.json"
    identity = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": source["files"],
        "budget": budget,
    }
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("identity") != identity:
            raise ValueError("V2 SEA preparation plan changed; use a separate output directory")
        return plan

    candidates = []
    metadata_root = root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for ordinal, item in enumerate(source["files"], 1):
        cache = metadata_root / (_safe_id(item["path"]) + ".json")
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("source") != item:
                raise ValueError("Cached SEA metadata source differs from pinned source")
            candidates.extend(saved["groups"])
            continue

        print(f"Scanning SEA metadata {ordinal}/{len(source['files'])}: {item['path']}", flush=True)
        groups = []
        with RangeFile(client, item) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            required = {"language", "label", "audio", "row_id", "split"}
            if not required.issubset(parquet.schema_arrow.names):
                raise ValueError("SEA-Spoof schema differs from the verified release")
            for row_group in range(parquet.num_row_groups):
                labels = Counter()
                for batch in parquet.iter_batches(
                    row_groups=[row_group], columns=["language", "label"], batch_size=1024, use_threads=False
                ):
                    for row in batch.to_pylist():
                        if row["language"] == "en":
                            if row["label"] not in {"bonafide", "spoof"}:
                                raise ValueError("Unexpected English SEA label")
                            labels[row["label"]] += 1
                if not labels:
                    continue
                metadata = parquet.metadata.row_group(row_group)
                estimated = sum(metadata.column(i).total_compressed_size for i in range(metadata.num_columns)) + 1024**2
                groups.append(
                    {
                        "unit": _safe_id(f"{item['path']}:{row_group}"),
                        "path": item["path"],
                        "split": item["split"],
                        "row_group": row_group,
                        "english_rows": sum(labels.values()),
                        "labels": dict(labels),
                        "estimated_bytes": estimated,
                    }
                )
            parquet.close()
        save_json(cache, {"source": item, "groups": groups})
        candidates.extend(groups)

    # Preserve the old allocation strategy across official splits, but use a V2
    # specific budget. Transfer's global 30 GB hard ceiling remains authoritative.
    available = min(int(budget), MAX_BYTES - client.used - 1_000_000_000)
    if available <= 0:
        raise ValueError("No SEA transfer budget remains")
    selected = select_groups(candidates, budget=available)
    plan = {
        "identity": identity,
        "groups": selected,
        "estimated_payload_bytes": sum(group["estimated_bytes"] for group in selected),
        "english_rows_before_audit": sum(group["english_rows"] for group in selected),
    }
    save_json(path, plan)
    return plan


def _source_utterance(row: dict) -> str | None:
    value = row.get("utterance_id")
    if value is None or not str(value).strip():
        return None
    source_dataset = str(row.get("source_dataset") or "unknown").strip()
    return f"SEA:{source_dataset}:{str(value).strip()}"


def _speaker(row: dict, label: str) -> str | None:
    for field in ("speaker_id", "speaker_or_voice"):
        value = row.get(field)
        if value is not None and str(value).strip():
            prefix = "human" if label == "bonafide" else "voice"
            return f"SEA:{prefix}:{str(value).strip()}"
    return None


def _generator(row: dict, label: str) -> str | None:
    if label == "bonafide":
        return None
    value = row.get("source_model")
    if value is None or not str(value).strip():
        # V2 contract requires generator identity for unseen-generator auditing.
        # Do not invent a unique generator from row_id because that would make the
        # unseen-generator test meaningless.
        return None
    return f"SEA:{str(value).strip()}"


def prepare(
    root: Path,
    source: dict,
    client: Transfer,
    *,
    budget: int = 26_000_000_000,
) -> Path:
    import pyarrow.parquet as pq

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise ValueError("SEA source configuration does not match pinned V2 source")
    if not 1_000_000_000 <= budget <= MAX_BYTES:
        raise ValueError("budget must be between 1 GB and the 30 GB hard ceiling")

    plan = _scan_groups(root, source, client, budget)
    sources = {item["path"]: item for item in source["files"]}
    audio_root = root / "audio" / "SEA-Spoof"
    receipts_root = root / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)

    records: list[AudioRecord] = []
    receipts: list[dict] = []
    exact_hash_owner: dict[str, str] = {}
    canonical_hash_owner: dict[str, str] = {}
    duplicate_counts = Counter()

    for ordinal, group in enumerate(plan["groups"], 1):
        receipt_path = receipts_root / f"{group['unit']}.json"
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            if saved.get("group") != group:
                raise ValueError("Completed V2 SEA group identity changed")
            for value in saved.get("records", []):
                record = AudioRecord.from_dict(value["manifest"])
                records.append(record)
                receipts.append(value["receipt"])
                exact_hash_owner[value["receipt"]["source_audio_sha256"]] = record.split
                canonical_hash_owner[record.content_sha256] = record.split
            print(f"Using completed V2 SEA group {ordinal}/{len(plan['groups'])}", flush=True)
            continue

        if shutil.disk_usage(root).free < 2_000_000_000:
            raise ValueError("Less than 2 GB free disk space remains; preparation stopped safely")

        print(f"Materializing V2 SEA group {ordinal}/{len(plan['groups'])}", flush=True)
        completed = []
        with RangeFile(client, sources[group["path"]]) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            for batch in parquet.iter_batches(row_groups=[group["row_group"]], batch_size=16, use_threads=False):
                for row in batch.to_pylist():
                    if row.get("language") != "en":
                        continue
                    if row.get("split") != group["split"] or row.get("label") not in {"bonafide", "spoof"}:
                        raise ValueError("SEA row violates selected group contract")
                    rid = row.get("row_id")
                    audio = row.get("audio")
                    raw = audio.get("bytes") if isinstance(audio, dict) else None
                    if not isinstance(rid, str) or not rid or not isinstance(raw, bytes) or not raw:
                        raise ValueError("SEA row is missing row_id or embedded audio bytes")

                    split = SOURCE_SPLITS[group["split"]]
                    label = str(row["label"])
                    source_hash = _hash_bytes(raw)
                    if source_hash in exact_hash_owner:
                        duplicate_counts[f"exact_{split}"] += 1
                        continue

                    encoded, audio_meta = _canonical_flac(raw)
                    canonical_hash = _hash_bytes(encoded)
                    if canonical_hash in canonical_hash_owner:
                        duplicate_counts[f"canonical_{split}"] += 1
                        continue

                    generator_id = _generator(row, label)
                    if label == "spoof" and generator_id is None:
                        # A spoof with unknown generator is still valuable for the
                        # ordinary benchmark, but it cannot satisfy the V2 manifest
                        # contract used for unseen-generator claims. Exclude rather
                        # than silently inventing provenance.
                        duplicate_counts[f"missing_generator_{split}"] += 1
                        continue

                    relative = Path("audio") / "SEA-Spoof" / split / f"{rid}.flac"
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".flac.tmp")
                    temporary.write_bytes(encoded)
                    temporary.replace(target)

                    record = AudioRecord(
                        record_id=f"SEA:{rid}",
                        label=label,  # type: ignore[arg-type]
                        dataset="SEA-Spoof",
                        split=split,  # type: ignore[arg-type]
                        audio_ref=str(relative.as_posix()),
                        language="en",
                        speaker_id=_speaker(row, label),
                        generator_id=generator_id,
                        source_utterance_id=_source_utterance(row),
                        content_sha256=canonical_hash,
                        codec="FLAC/PCM_16",
                        sample_rate=MODEL_SAMPLE_RATE,
                    )
                    receipt = {
                        "source_repository": REPOSITORY,
                        "source_revision": REVISION,
                        "source_parquet": group["path"],
                        "source_row_group": group["row_group"],
                        "source_row_id": rid,
                        "source_split": group["split"],
                        "source_audio_sha256": source_hash,
                        "source_model": row.get("source_model"),
                        "source_dataset": row.get("source_dataset"),
                        "speaker_or_voice": row.get("speaker_or_voice"),
                        "utterance_id": row.get("utterance_id"),
                        **audio_meta,
                    }
                    exact_hash_owner[source_hash] = split
                    canonical_hash_owner[canonical_hash] = split
                    records.append(record)
                    receipts.append(receipt)
                    completed.append({"manifest": record.as_dict(), "receipt": receipt})
            parquet.close()
        save_json(receipt_path, {"group": group, "records": completed})

    manifest_path = root / "manifest.jsonl"
    audit = audit_manifest(records)
    audit.raise_for_errors()
    write_jsonl(manifest_path, records)
    save_json(
        root / "v2_sea_audit.json",
        {
            "schema": SCHEMA,
            "repository": REPOSITORY,
            "revision": REVISION,
            "records": len(records),
            "audit_counts": audit.counts,
            "audit_warnings": list(audit.warnings),
            "filtered": dict(duplicate_counts),
            "transfer_reserved_bytes": client.used,
            "canonical_audio": "mono 16 kHz FLAC PCM_16",
            "limitations": [
                "Official SEA source split roles are retained as train/dev/test.",
                "Generator identity comes from source_model and may be absent for some spoof rows; those rows are excluded.",
                "speaker_or_voice semantics depend on the upstream source and are retained only as audit metadata.",
                "This preparation does not by itself establish unseen-generator separation; use v2_data_contract audits for that scenario.",
            ],
        },
    )
    return manifest_path


def main() -> None:
    import argparse
    from huggingface_hub import get_token

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=Path("configs/seaspoof_source.json"))
    parser.add_argument("--budget-gb", type=float, default=26.0)
    args = parser.parse_args()

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    client = Transfer(args.output, get_token())
    manifest = prepare(args.output, source, client, budget=int(args.budget_gb * 1_000_000_000))
    print(f"V2 SEA manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
