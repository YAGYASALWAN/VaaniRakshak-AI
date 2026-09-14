"""Audit VaaniRakshak V2 manifests or derive strict evaluation scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaanirakshak.v2_data_contract import (  # noqa: E402
    audit_cross_dataset,
    audit_manifest,
    audit_unseen_generator,
    load_jsonl,
    manifest_fingerprint,
)
from vaanirakshak.v2_manifest_tools import build_unseen_generator_file  # noqa: E402


def audit_command(args) -> int:
    records = load_jsonl(args.manifest)
    if args.scenario == "standard":
        result = audit_manifest(records)
    elif args.scenario == "unseen-generator":
        result = audit_unseen_generator(records)
    else:
        result = audit_cross_dataset(records, evaluation_datasets=set(args.evaluation_dataset))

    payload = {
        "ok": result.ok,
        "scenario": args.scenario,
        "manifest": str(args.manifest.resolve()),
        "fingerprint": manifest_fingerprint(records),
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "counts": result.counts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 2


def unseen_command(args) -> int:
    report = build_unseen_generator_file(
        args.manifest,
        args.output,
        heldout_generators=set(args.holdout_generator),
        dev_fraction=args.dev_fraction,
        test_bonafide_ratio=args.test_bonafide_ratio,
        seed=args.seed,
    )
    report_path = args.output.with_suffix(args.output.suffix + ".scenario.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote strict unseen-generator manifest: {args.output.resolve()}")
    print(f"Scenario report: {report_path.resolve()}")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit a V2 JSONL manifest")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--scenario", choices=("standard", "unseen-generator", "cross-dataset"), default="standard")
    audit.add_argument("--evaluation-dataset", action="append", default=[])
    audit.set_defaults(func=audit_command)

    unseen = sub.add_parser("unseen-generator", help="Derive a strict generator-held-out scenario")
    unseen.add_argument("--manifest", type=Path, required=True, help="Source/pool manifest")
    unseen.add_argument("--output", type=Path, required=True)
    unseen.add_argument("--holdout-generator", action="append", required=True)
    unseen.add_argument("--dev-fraction", type=float, default=0.10)
    unseen.add_argument("--test-bonafide-ratio", type=float, default=1.0)
    unseen.add_argument("--seed", type=int, default=42)
    unseen.set_defaults(func=unseen_command)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
