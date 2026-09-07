"""Audit SEA-Spoof metadata without downloading or redistributing audio."""
from collections import Counter, defaultdict
import hashlib
import re

LANGUAGES = {"en", "hi", "id", "ms", "ta", "th", "vi"}
SPLITS = {"train", "validation", "evaluation"}
LABELS = {"bonafide": 0, "spoof": 1}


def audit(rows, language="en"):
    if language not in LANGUAGES | {"all"}:
        raise ValueError("Unknown language")
    counts = Counter()
    owners = {field: defaultdict(set) for field in ("row_id", "utterance_id", "speaker_id", "audio_sha256", "exact_text")}
    coverage = Counter()
    invalid = Counter()
    unique_ids = set()
    for row in rows:
        if row.get("language") not in LANGUAGES:
            invalid["unknown_language"] += 1
            continue
        if language != "all" and row["language"] != language:
            continue
        split, label, rid = row.get("split"), row.get("label"), row.get("row_id")
        if split not in SPLITS or label not in LABELS or not isinstance(rid, str) or not rid:
            invalid["invalid_split_label_or_row_id"] += 1
            continue
        if rid in unique_ids:
            invalid["duplicate_row_id"] += 1
        unique_ids.add(rid)
        counts[split, row["language"], label] += 1
        for field in ("row_id", "utterance_id", "speaker_id", "audio_sha256"):
            value = row.get(field)
            if value is not None and str(value).strip():
                if field == "audio_sha256" and not re.fullmatch(r"[0-9a-f]{64}", str(value)):
                    invalid["invalid_audio_sha256"] += 1
                    continue
                owners[field][str(value)].add(split)
                coverage[field] += 1
        if row.get("is_text_exact") in (True, "true", "True") and row.get("text"):
            text = " ".join(str(row["text"]).lower().split())
            if len(text) >= 20:
                owners["exact_text"][hashlib.sha256(text.encode()).hexdigest()].add(split)
    overlaps = {field: sum(len(splits) > 1 for splits in groups.values()) for field, groups in owners.items()}
    missing = []
    included_languages = LANGUAGES if language == "all" else {language}
    for split in sorted(SPLITS):
        for lang in sorted(included_languages):
            for label in LABELS:
                if not counts[split, lang, label]:
                    missing.append(f"{split}/{lang}/{label}")
    total = sum(counts.values())
    hard_failure = bool(invalid or missing or overlaps["row_id"] or overlaps["audio_sha256"] or overlaps["speaker_id"])
    return dict(status="issues_found" if hard_failure else "metadata_checks_passed", language=language,
                records=total, label_mapping=LABELS, counts={"/".join(k): v for k, v in sorted(counts.items())},
                invalid_records=dict(invalid), missing_cells=missing, cross_split_overlaps=overlaps,
                speaker_coverage=coverage["speaker_id"], audio_hash_coverage=coverage["audio_sha256"],
                speaker_separation="checked_for_supplied_ids" if total and coverage["speaker_id"] == total else "unverified",
                audio_duplicates="checked_for_supplied_hashes" if total and coverage["audio_sha256"] == total else "unverified",
                limitations=["SEA-Spoof row_id is the unique key; utterance_id can collide across source subsets.",
                             "Repeated utterance IDs or exact transcripts are review flags, not automatically proof of shared speakers.",
                             "Missing speaker IDs cannot be repaired by inventing IDs from row numbers.",
                             "This metadata audit does not measure detector accuracy or inspect audio quality."])
