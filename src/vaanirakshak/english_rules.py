"""Dataset contract for the full ASVspoof 2019 LA English experiment."""
from collections import Counter
import re

REPOSITORY = "Bisher/ASVspoof_2019_LA"
REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
COUNTS = {"train": {0: 2580, 1: 22800}, "validation": {0: 2548, 1: 22296},
          "test": {0: 7355, 1: 63882}}
PREFIX = {"train": "LA_T_", "validation": "LA_D_", "test": "LA_E_"}
NOTICE = ("English ASVspoof 2019 LA experiment using a third-party mirror. "
          "The official partitions are preserved. One centre four-second window is scored per recording. "
          "Results do not establish performance on modern Qwen3 voices, telephone calls, or other languages.")


def record_metadata(row, split):
    if split not in COUNTS:
        raise ValueError("Unknown official split")
    label = row.get("key")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Expected verified labels 0=bonafide and 1=spoof")
    rid, speaker, attack = row.get("audio_file_name"), row.get("speaker_id"), row.get("system_id")
    if not isinstance(rid, str) or not re.fullmatch(PREFIX[split] + r"\d{7}", rid):
        raise ValueError("Audio ID does not match the official split")
    if not isinstance(speaker, str) or not re.fullmatch(r"LA_\d{4}", speaker):
        raise ValueError("Missing or invalid source speaker ID")
    if label == 0 and attack != "-":
        raise ValueError("Genuine label disagrees with attack ID")
    if label == 1 and (not isinstance(attack, str) or not re.fullmatch(r"A\d{2}", attack)):
        raise ValueError("Synthetic label lacks an attack ID")
    return {"id": rid, "speaker": speaker, "label": label, "attack": attack}


def validate_partition(rows, split):
    observed = Counter(row["label"] for row in rows)
    if dict(observed) != COUNTS[split]:
        raise ValueError(f"Incomplete {split} partition: {dict(observed)}; expected {COUNTS[split]}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source recording IDs")
    for row in rows:
        record_metadata({"key": row["label"], "audio_file_name": row["id"],
                         "speaker_id": row["speaker"], "system_id": row["attack"]}, split)


def check_disjoint(partitions):
    """No known target speaker, recording ID or exact original audio crosses splits."""
    for field in ("id", "speaker", "audio_sha256"):
        owners = {}
        for split, rows in partitions.items():
            for row in rows:
                value = row[field]
                if value in owners and owners[value] != split:
                    raise ValueError(f"Cross-split {field} overlap between {owners[value]} and {split}")
                owners[value] = split
