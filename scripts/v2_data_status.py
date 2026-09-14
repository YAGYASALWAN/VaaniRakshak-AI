"""Inspect VaaniRakshak V2 SEA preparation state without network access.

This command is safe to run while diagnosing or resuming a large SEA-Spoof
preparation directory. It never contacts Hugging Face and never mutates the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaanirakshak.sea_transfer import MAX_BYTES
from vaanirakshak.v2_data_contract import audit_manifest, load_jsonl, manifest_fingerprint


SCHEMA = "vaanirakshak-v2-data-status-v1"


def _read_json(path: Path, *, label: str, issues: list[str]):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"{label} is unreadable or invalid JSON: {exc}")
        return None
    return value


def _directory_bytes(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    files = 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                files += 1
                total += item.stat().st_size
        except OSError:
            continue
    return files, total


def inspect(data_root: Path, *, scan_audio: bool = False) -> dict:
    data_root = data_root.expanduser().resolve()
    issues: list[str] = []
    warnings: list[str] = []

    probe = data_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    disk = shutil.disk_usage(probe)

    ledger_path = data_root / "transfer.json"
    reserved_bytes = 0
    ledger_present = ledger_path.is_file()
    if ledger_present:
        ledger = _read_json(ledger_path, label="transfer.json", issues=issues)
        if not isinstance(ledger, dict):
            issues.append("transfer.json must contain a JSON object")
        else:
            value = ledger.get("reserved_bytes")
            if isinstance(value, bool) or not isinstance(value, int):
                issues.append("transfer.json reserved_bytes must be an integer")
            elif not 0 <= value <= MAX_BYTES:
                issues.append("transfer.json reserved_bytes is outside the 0..30 GB hard ceiling")
            else:
                reserved_bytes = value

    plan_path = data_root / "v2_sea_plan.json"
    planned_groups: set[str] = set()
    plan_summary = None
    if plan_path.is_file():
        plan = _read_json(plan_path, label="v2_sea_plan.json", issues=issues)
        if isinstance(plan, dict) and isinstance(plan.get("groups"), list):
            groups = plan["groups"]
            for group in groups:
                if not isinstance(group, dict) or not str(group.get("unit") or "").strip():
                    issues.append("v2_sea_plan.json contains a group without a valid unit id")
                    continue
                planned_groups.add(str(group["unit"]))
            plan_summary = {
                "groups": len(groups),
                "unique_group_ids": len(planned_groups),
                "estimated_payload_bytes": plan.get("estimated_payload_bytes"),
                "english_rows_before_audit": plan.get("english_rows_before_audit"),
            }
            if len(planned_groups) != len(groups):
                issues.append("v2_sea_plan.json contains duplicate/invalid group ids")
        elif plan is not None:
            issues.append("v2_sea_plan.json does not contain a groups list")

    metadata_root = data_root / "metadata"
    metadata_files = list(metadata_root.glob("*.json")) if metadata_root.is_dir() else []

    receipts_root = data_root / "receipts"
    completed_groups: set[str] = set()
    receipt_records = 0
    invalid_receipts = 0
    if receipts_root.is_dir():
        for path in receipts_root.glob("*.json"):
            value = _read_json(path, label=f"receipt {path.name}", issues=issues)
            if not isinstance(value, dict) or not isinstance(value.get("group"), dict):
                invalid_receipts += 1
                continue
            unit = str(value["group"].get("unit") or "").strip()
            records = value.get("records")
            if not unit or not isinstance(records, list):
                invalid_receipts += 1
                issues.append(f"receipt {path.name} is missing group unit or records")
                continue
            if planned_groups and unit not in planned_groups:
                warnings.append(f"receipt {path.name} belongs to a group not present in the current plan")
            completed_groups.add(unit)
            receipt_records += len(records)

    cache_root = data_root / "range_cache"
    cache_scopes = 0
    if cache_root.is_dir():
        cache_scopes = sum(1 for item in cache_root.iterdir() if item.is_dir())
    cache_files, cache_bytes = _directory_bytes(cache_root)

    manifest_path = data_root / "manifest.jsonl"
    manifest_summary = None
    manifest_complete = False
    if manifest_path.is_file():
        try:
            records = load_jsonl(manifest_path)
            audit = audit_manifest(records)
            manifest_summary = {
                "records": len(records),
                "fingerprint": manifest_fingerprint(records),
                "audit_ok": audit.ok,
                "audit_errors": list(audit.errors),
                "audit_warnings": list(audit.warnings),
                "audit_counts": audit.counts,
            }
            if audit.errors:
                issues.extend(f"manifest audit: {message}" for message in audit.errors)
            manifest_complete = audit.ok
        except (OSError, ValueError) as exc:
            issues.append(f"manifest.jsonl cannot be loaded/audited: {exc}")

    audio_summary = {
        "scanned": bool(scan_audio),
        "flac_files": None,
        "bytes": None,
    }
    if scan_audio:
        audio_files, audio_bytes = _directory_bytes(data_root / "audio")
        audio_summary.update({"flac_files": audio_files, "bytes": audio_bytes})

    planned_count = len(planned_groups)
    completed_count = len(completed_groups)
    if issues:
        state = "blocked"
    elif manifest_complete:
        state = "complete"
    elif planned_count:
        state = "finalizing" if completed_count >= planned_count else "materializing"
    elif metadata_files or ledger_present:
        state = "planning"
    else:
        state = "not_started"

    if state == "complete":
        next_action = "Run/inspect the manifest audit, then begin the frozen training workflow."
    elif state in {"planning", "materializing", "finalizing"}:
        next_action = "Rerun the same v2_prepare_sea command with this same output directory."
    elif state == "blocked":
        next_action = "Do not resume automatically; resolve the reported integrity/safety issues first."
    else:
        next_action = "Run V2 preflight, then start v2_prepare_sea in this directory."

    return {
        "schema": SCHEMA,
        "data_root": str(data_root),
        "state": state,
        "safe_to_resume": state != "blocked",
        "network_used": False,
        "issues": issues,
        "warnings": warnings,
        "disk": {
            "probe_path": str(probe),
            "free_bytes": int(disk.free),
            "free_gib": round(disk.free / (1024 ** 3), 2),
        },
        "transfer": {
            "ledger_present": ledger_present,
            "reserved_bytes": reserved_bytes,
            "hard_ceiling_bytes": MAX_BYTES,
            "remaining_ceiling_bytes": MAX_BYTES - reserved_bytes,
        },
        "plan": plan_summary,
        "metadata_files": len(metadata_files),
        "receipts": {
            "completed_groups": completed_count,
            "planned_groups": planned_count,
            "receipt_records": receipt_records,
            "invalid_receipts": invalid_receipts,
        },
        "retry_cache": {
            "scopes": cache_scopes,
            "files": cache_files,
            "bytes": cache_bytes,
        },
        "manifest": manifest_summary,
        "audio": audio_summary,
        "next_action": next_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "v2_sea_en")
    parser.add_argument("--scan-audio", action="store_true", help="Count files/bytes under audio/ (can be slow on a large corpus).")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Exit code 2 when integrity/safety issues are present.")
    args = parser.parse_args()

    report = inspect(args.data_root, scan_audio=args.scan_audio)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    if args.strict and not report["safe_to_resume"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
