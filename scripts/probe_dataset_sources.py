#!/usr/bin/env python3
"""Milestone 3 objective 1: inspect bounded public JSON metadata, never audio."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from vaanirakshak.data.source_probe import run_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="New output directory")
    args = parser.parse_args()
    output = args.output or Path("data/metadata") / (
        "source_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        report = run_probe(output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Probe failed: {exc}\n")
    compact = {
        "summary_file": str(output / "probe_summary.json"), "audio_downloaded": False,
        "payload_bytes_read": report["payload_bytes_read"],
        "sources": {
            alias: {
                "errors": value["errors"],
                "subsets": {language: item["status"] for language, item in value["subsets"].items()},
            } for alias, value in report["sources"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    errors = any(source["errors"] or any(item["status"] != "inspected"
                 for item in source["subsets"].values()) for source in report["sources"].values())
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
