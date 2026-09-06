#!/usr/bin/env python3
"""Audit local metadata exports; downloads and training are never performed."""
import argparse
import json
from pathlib import Path

from vaanirakshak.data.audit import audit
from vaanirakshak.data.metadata_io import load_inputs
from vaanirakshak.data.reporting import write_reports
from vaanirakshak.data.sampling_config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sampling_config.yaml"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    _, thresholds, raw = load_settings(args.config)
    frame = load_inputs(raw.get("inputs", {}), args.config.resolve().parent)
    # ASVspoof must not influence primary acquisition decisions.
    frame = frame[frame.dataset.isin(["indicvoices", "indicsynth"])].copy()
    report = audit(frame, thresholds)
    output = args.config.resolve().parent / raw.get("output", "../data/reports")
    write_reports(frame, output, bias=report, plots=args.plots)
    print(json.dumps({"output": str(output.resolve()), "status": report["status"],
                      "findings": len(report["findings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
