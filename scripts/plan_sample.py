#!/usr/bin/env python3
"""Create a deterministic metadata-only candidate plan, never acquire audio."""
import argparse
import json
from pathlib import Path

from vaanirakshak.data.metadata_io import load_inputs
from vaanirakshak.data.reporting import write_json
from vaanirakshak.data.sampling import plan_sample
from vaanirakshak.data.sampling_config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sampling_config.yaml"))
    args = parser.parse_args()
    config, _, raw = load_settings(args.config)
    frame = load_inputs(raw.get("inputs", {}), args.config.resolve().parent)
    result = plan_sample(frame, config)
    output = args.config.resolve().parent / raw.get("output", "../data/reports")
    write_json(output / "sampling_plan.json", result.report)
    result.selected.to_csv(output / "candidate_plan.csv", index=False)
    print(json.dumps(result.report, indent=2, allow_nan=False))
    return 0 if result.report["status"] == "feasible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
