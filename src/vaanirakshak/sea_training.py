"""English-only SEA-Spoof preparation under a cumulative 30 GB payload ceiling."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from vaanirakshak.baseline_download import file_sha, save_json
from vaanirakshak.sea_transfer import MAX_BYTES, REPOSITORY, REVISION, RangeFile

SCHEMA = "sea30-en-logmel-v1"
SOURCE_SPLITS = {"train": "train", "validation": "validation", "evaluation": "test"}
NOTICE = ("English-only SEA-Spoof subset experiment. Original train/validation/evaluation roles retained. "
          "Known duplicate audio is removed across partitions. Speaker and generator independence are not established. "
          "One centre four-second window per clip; scores are not calibrated probabilities or proof of authenticity.")


def select_groups(candidates, budget=26_500_000_000):
    """Select a usable English subset without assuming Parquet row groups fit fixed split quotas.

    SEA Parquet row groups are indivisible transfer units. A validation/evaluation row
    group can therefore be larger than a nominal 10% split allocation even when the
    complete train/dev/test selection comfortably fits the global budget. Reserve the
    minimum class coverage for every official split first, then spend the remainder
    toward the 80/10/10 target and finally use any otherwise stranded global budget.
    """
    splits = (("train", .8), ("validation", .1), ("evaluation", .1))
    labels = ("bonafide", "spoof")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("SEA selection budget must be positive")

    pools = {
        split: [x for x in candidates if x["split"] == split and x.get("english_rows", 0)]
        for split, _ in splits
    }
    counts = {split: Counter() for split, _ in splits}
    split_bytes = Counter()
    selected = []
    spent = 0

    def size(group):
        value = int(group["estimated_bytes"])
        if value <= 0:
            raise ValueError("SEA candidate has a non-positive estimated size")
        return value

    def add(split, group):
        nonlocal spent
        selected.append(group)
        pools[split].remove(group)
        counts[split].update(group.get("labels", {}))
        amount = size(group)
        split_bytes[split] += amount
        spent += amount

    def minimum_priority(split, group):
        deficit = {label: max(0, 100 - counts[split][label]) for label in labels}
        marginal = sum(min(int(group.get("labels", {}).get(label, 0)), deficit[label]) for label in labels)
        amount = size(group)
        english_rows = int(group.get("english_rows", 0))
        return (marginal / amount, marginal, english_rows / amount, english_rows, str(group["unit"]))

    def fill_priority(split, group):
        amount = size(group)
        coverage = sum(
            int(group.get("labels", {}).get(label, 0)) / (counts[split][label] + 100)
            for label in labels
        )
        english_rows = int(group.get("english_rows", 0))
        return (coverage / amount, english_rows / amount, english_rows, str(group["unit"]))

    for split, _ in splits:
        if not pools[split]:
            raise ValueError(f"No English SEA candidates are available for {split}")
        while any(counts[split][label] < 100 for label in labels):
            remaining = budget - spent
            eligible = []
            for group in pools[split]:
                if size(group) > remaining:
                    continue
                deficits = {label: max(0, 100 - counts[split][label]) for label in labels}
                marginal = sum(
                    min(int(group.get("labels", {}).get(label, 0)), deficits[label])
                    for label in labels
                )
                if marginal > 0:
                    eligible.append(group)
            if not eligible:
                raise ValueError(
                    f"Cannot form a usable English {split} subset within the global byte budget; "
                    f"coverage so far is bonafide={counts[split]['bonafide']}, spoof={counts[split]['spoof']}"
                )
            add(split, max(eligible, key=lambda group: minimum_priority(split, group)))

    for split, fraction in splits:
        target = int(budget * fraction)
        while pools[split] and spent < budget:
            allowance = min(budget - spent, max(0, target - split_bytes[split]))
            if allowance <= 0:
                break
            eligible = [group for group in pools[split] if size(group) <= allowance]
            if not eligible:
                break
            add(split, max(eligible, key=lambda group: fill_priority(split, group)))

    split_fractions = dict(splits)
    while spent < budget:
        remaining = budget - spent
        eligible = [
            (split, group)
            for split, _ in splits
            for group in pools[split]
            if size(group) <= remaining
        ]
        if not eligible:
            break

        def global_priority(item):
            split, group = item
            amount = size(group)
            english_rows = int(group.get("english_rows", 0))
            target = int(budget * split_fractions[split])
            under_target = max(0, target - split_bytes[split])
            return (under_target > 0, english_rows / amount, english_rows, str(group["unit"]))

        split, group = max(eligible, key=global_priority)
        add(split, group)

    if spent > budget:
        raise AssertionError("SEA group selection exceeded the global budget")
    return selected


def scan(root, source, client):
    import pyarrow.parquet as pq
    path = Path(root) / "plan.json"
    identity = {"schema": SCHEMA, "repository": REPOSITORY, "revision": REVISION, "files": source["files"]}
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["identity"] != identity:
            raise ValueError("Preparation plan changed; use a separate data directory")
        return plan
    candidates = []
    for index, item in enumerate(source["files"], 1):
        print(f"Inspecting language metadata {index}/{len(source['files'])}: {item['path']}", flush=True)
        scan_path = Path(root) / "metadata" / (hashlib.sha256(item["path"].encode()).hexdigest()[:20] + ".json")
        if scan_path.exists():
            saved = json.loads(scan_path.read_text())
            if saved["source"] != item:
                raise ValueError("Cached metadata source differs")
            candidates.extend(saved["groups"])
            continue
        if client.used > 1_500_000_000:
            raise ValueError("Metadata inspection exceeded its 1.5 GB allowance; audio loading has not started")
        groups = []
        with RangeFile(client, item) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            if not {"language", "label", "audio", "row_id", "split"}.issubset(parquet.schema_arrow.names):
                raise ValueError("SEA-Spoof schema differs from the verified release")
            for number in range(parquet.num_row_groups):
                labels = Counter()
                for batch in parquet.iter_batches(row_groups=[number], columns=["language", "label"], batch_size=512, use_threads=False):
                    for row in batch.to_pylist():
                        if row["language"] == "en":
                            if row["label"] not in ("bonafide", "spoof"):
                                raise ValueError("Unexpected English label")
                            labels[row["label"]] += 1
                metadata = parquet.metadata.row_group(number)
                estimated = sum(metadata.column(i).total_compressed_size for i in range(metadata.num_columns)) + 1024**2
                if labels:
                    unit = hashlib.sha256(f"{item['path']}:{number}".encode()).hexdigest()[:24]
                    groups.append(dict(unit=unit, path=item["path"], split=item["split"], row_group=number,
                                       english_rows=sum(labels.values()), labels=dict(labels), estimated_bytes=estimated))
            parquet.close()
        save_json(scan_path, {"source": item, "groups": groups})
        candidates.extend(groups)
    selected = select_groups(candidates, min(26_500_000_000, MAX_BYTES - client.used - 2_000_000_000))
    plan = {"identity": identity, "groups": selected, "estimated_payload_bytes": sum(x["estimated_bytes"] for x in selected),
            "english_rows_before_audit": sum(x["english_rows"] for x in selected), "max_payload_bytes": MAX_BYTES}
    save_json(path, plan)
    print(f"Selected {plan['english_rows_before_audit']:,} English recordings before duplicate audit; "
          f"estimated additional reads {plan['estimated_payload_bytes']/1e9:.2f} GB.", flush=True)
    return plan


def filter_duplicates(rows):
    """Prefer evaluation over validation over training when exact audio is repeated."""
    seen, result, rejected = set(), [], Counter()
    ids = set()
    for split in ("test", "validation", "train"):
        for row in (r for r in rows if r["split"] == split):
            if row["id"] in ids:
                raise ValueError("Duplicate global row_id in selected source")
            ids.add(row["id"])
            keys = {"audio:" + row["audio_sha256"], "window:" + row["window_sha256"]}
            if seen.intersection(keys):
                rejected[split] += 1
                continue
            seen.update(keys)
            result.append(row)
    return result, dict(rejected)


def check_partitions(partitions):
    for field in ("id", "audio_sha256", "window_sha256"):
        owners = {}
        for split, rows in partitions.items():
            for row in rows:
                if row["language"] != "en":
                    raise ValueError("Non-English recording reached training")
                value = row[field]
                if value in owners and owners[value] != split:
                    raise ValueError("Source identity or audio crosses partitions")
                owners[value] = split


def prepare(root, source, client, device):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from vaanirakshak.english_training import Frontend, FEATURE_SHAPE, waveform_window

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = scan(root, source, client)
    sources = {x["path"]: x for x in source["files"]}
    frontend = Frontend().to(device).eval()
    all_rows = []
    for ordinal, group in enumerate(plan["groups"], 1):
        folder = root / "features"
        folder.mkdir(exist_ok=True)
        path, receipt = folder / (group["unit"] + ".npy"), folder / (group["unit"] + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["group"] != group or file_sha(path) != saved["sha256"]:
                raise ValueError("Completed feature checksum mismatch")
            all_rows.extend(saved["rows"])
            print(f"Using feature group {ordinal}/{len(plan['groups'])}", flush=True)
            continue
        required = group["english_rows"] * 64 * 401 * 2
        if shutil.disk_usage(root).free < required + 1_000_000_000:
            raise ValueError("Insufficient feature cache space; free space without deleting this run's cache")
        print(f"Preparing English group {ordinal}/{len(plan['groups'])}; "
              f"transfer reserved {client.used/1e9:.2f}/30 GB", flush=True)
        features = np.lib.format.open_memmap(path, mode="w+", dtype="float16", shape=(group["english_rows"], *FEATURE_SHAPE))
        records, pending, waves = [], [], []

        def flush():
            if not pending:
                return
            with torch.no_grad():
                values = frontend(torch.from_numpy(np.stack(waves)).to(device)).cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError("Invalid audio features")
            features[len(records):len(records) + len(pending)] = values
            records.extend(pending)
            pending.clear()
            waves.clear()

        with RangeFile(client, sources[group["path"]]) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            for batch in parquet.iter_batches(row_groups=[group["row_group"]], batch_size=32, use_threads=False):
                for row in batch.to_pylist():
                    if row["language"] != "en":
                        continue
                    if row["split"] != group["split"] or row["label"] not in ("bonafide", "spoof") or not isinstance(row["row_id"], str):
                        raise ValueError("English row violates the split/label contract")
                    raw = row["audio"]["bytes"]
                    wave, properties = waveform_window(raw)
                    record = dict(id=row["row_id"], language="en", label=int(row["label"] == "spoof"),
                                  split=SOURCE_SPLITS[group["split"]], unit=group["unit"], index=len(records) + len(pending),
                                  audio_sha256=hashlib.sha256(raw).hexdigest(), window_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                                  source_model=row.get("source_model"), source_dataset=row.get("source_dataset"),
                                  speaker_or_voice=row.get("speaker_or_voice"), **properties)
                    pending.append(record)
                    waves.append(wave)
                    if len(pending) == 32:
                        flush()
            parquet.close()
        flush()
        if len(records) != group["english_rows"] or dict(Counter("spoof" if r["label"] else "bonafide" for r in records)) != group["labels"]:
            raise ValueError("Prepared English coverage differs from metadata scan")
        features.flush()
        del features
        save_json(receipt, {"group": group, "sha256": file_sha(path), "rows": records})
        all_rows.extend(records)
    rows, rejections = filter_duplicates(all_rows)
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    check_partitions(partitions)
    counts = {s: dict(Counter(r["label"] for r in group)) for s, group in partitions.items()}
    for split, count in counts.items():
        if min(count.get(k, 0) for k in (0, 1)) < 100:
            raise ValueError(f"Too few independent English examples in {split} after duplicate removal")
    voices = defaultdict(set)
    for row in rows:
        if row.get("speaker_or_voice"):
            voices[(row.get("source_dataset"), row["speaker_or_voice"])].add(row["split"])
    audit = dict(counts=counts, duplicate_rejections=rejections,
                 cross_split_speaker_or_voice_keys=sum(len(v) > 1 for v in voices.values()),
                 speaker_independence="unverified; source field may identify a synthetic voice rather than a human speaker",
                 source_generator_counts=dict(Counter(str(r.get("source_model")) for r in rows)),
                 transfer_reserved_bytes=client.used, notice=NOTICE)
    save_json(root / "audit.json", audit)
    save_json(root / "manifest.json", {"schema": SCHEMA, "rows": rows, "counts": counts})
    return rows, counts


def profile(root, rows, counts):
    import numpy as np
    arrays = {}

    class Features:
        def __init__(self, group):
            self.group = group

        def __getitem__(self, index):
            row = self.group[index]
            if row["unit"] not in arrays:
                arrays[row["unit"]] = np.load(Path(root) / "features" / (row["unit"] + ".npy"), mmap_mode="r", allow_pickle=False)
            return arrays[row["unit"]][row["index"]]

    def load(root_arg, split, device):
        group = [r for r in rows if r["split"] == split]
        return Features(group), group

    return dict(repository=REPOSITORY, revision=REVISION, notice=NOTICE, counts=counts,
                cache_split=load, check_disjoint=check_partitions, run_prefix="sea30_")


def create_detector(checkpoint=None, device="cpu"):
    from vaanirakshak.demo_server import Detector, ROOT

    class SEADetector(Detector):
        def checkpoint(self):
            if self.requested:
                return super().checkpoint()
            paths = list((ROOT / "models").glob("sea30_*/best.pt"))
            return max(paths, key=lambda p: p.stat().st_mtime_ns) if paths else None

    return SEADetector(checkpoint, device)