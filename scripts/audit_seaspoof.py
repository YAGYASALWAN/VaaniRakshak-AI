"""Audit an existing CSV or JSONL metadata export; no audio download."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vaanirakshak.seaspoof_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--language", default="en", choices=("en", "hi", "id", "ms", "ta", "th", "vi", "all"))
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source) if args.metadata.suffix.lower() == ".csv" else (json.loads(line) for line in source if line.strip())
        result = audit(rows, args.language)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result["status"] == "issues_found")


if __name__ == "__main__":
    raise SystemExit(main())
