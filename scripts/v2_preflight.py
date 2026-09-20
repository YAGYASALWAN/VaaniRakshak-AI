"""Preflight VaaniRakshak V2 before expensive data preparation or SSL training.

Checks local disk/GPU/runtime configuration and, unless --offline is supplied,
verifies that the current Hugging Face account can access the pinned SEA-Spoof
repository. This command downloads no dataset audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaanirakshak.sea_transfer import MAX_BYTES, REPOSITORY, REVISION


def _gib(value: int) -> float:
    return value / (1024 ** 3)


def run(*, data_root: Path, budget_gb: float, require_cuda: bool, offline: bool) -> dict:
    if not 1.0 <= budget_gb <= MAX_BYTES / 1_000_000_000:
        raise ValueError("budget_gb must be between 1 and the SEA 30 GB hard ceiling")

    source_path = ROOT / "configs" / "seaspoof_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise ValueError("configs/seaspoof_source.json does not match the pinned SEA source contract")
    if not isinstance(source.get("files"), list) or not source["files"]:
        raise ValueError("Pinned SEA source config has no files")

    data_root = data_root.expanduser().resolve()
    probe = data_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if not probe.exists():
        raise ValueError(f"Could not locate a filesystem parent for {data_root}")
    disk = shutil.disk_usage(probe)
    requested_payload = int(budget_gb * 1_000_000_000)
    # Preparation stores canonical FLAC in addition to remote transfer. Require
    # payload budget plus a conservative 8 GiB safety margin before beginning.
    required_free = requested_payload + 8 * 1024 ** 3
    disk_ok = disk.free >= required_free

    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_memory_bytes = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        cuda_memory_bytes = int(properties.total_memory)
    if require_cuda and not cuda_available:
        raise ValueError("CUDA is required for the planned V2 training run but torch.cuda.is_available() is false")

    import transformers
    import torchaudio

    hf_access = None
    hf_user = None
    if not offline:
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if not token:
            raise ValueError("No Hugging Face login found. Run `hf auth login` with the account approved for SEA-Spoof.")
        api = HfApi(token=token)
        try:
            who = api.whoami()
            hf_user = who.get("name") if isinstance(who, dict) else None
            api.auth_check(REPOSITORY, repo_type="dataset")
            # A pinned metadata request ensures the approved account can resolve
            # the exact revision without fetching any audio payload.
            api.list_repo_files(REPOSITORY, repo_type="dataset", revision=REVISION)
            hf_access = True
        except Exception as exc:
            raise ValueError(
                "Hugging Face login exists but the pinned SEA-Spoof revision is not accessible to this account"
            ) from exc

    report = {
        "ok": bool(disk_ok and (cuda_available or not require_cuda) and (offline or hf_access)),
        "data_root": str(data_root),
        "source": {"repository": REPOSITORY, "revision": REVISION, "configured_files": len(source["files"])},
        "budget_gb_decimal": budget_gb,
        "disk": {
            "probe_path": str(probe),
            "free_gib": round(_gib(disk.free), 2),
            "required_free_gib": round(_gib(required_free), 2),
            "ok": disk_ok,
        },
        "cuda": {
            "required": require_cuda,
            "available": cuda_available,
            "device": cuda_device,
            "total_memory_gib": None if cuda_memory_bytes is None else round(_gib(cuda_memory_bytes), 2),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torchaudio": str(torchaudio.__version__),
            "transformers": str(transformers.__version__),
        },
        "huggingface": {
            "checked": not offline,
            "user": hf_user,
            "pinned_sea_access": hf_access,
        },
        "notes": [
            "Preflight downloads no dataset audio.",
            "Disk check is conservative: transfer budget plus 8 GiB safety margin.",
            "A successful preflight does not predict model accuracy or exact training VRAM usage.",
        ],
    }
    if not disk_ok:
        raise ValueError(
            f"Insufficient free disk: {report['disk']['free_gib']} GiB available, "
            f"{report['disk']['required_free_gib']} GiB required by preflight policy"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "v2_sea_en")
    parser.add_argument("--budget-gb", type=float, default=26.0)
    parser.add_argument("--allow-cpu", action="store_true", help="Do not require CUDA (useful only for preparation/debugging).")
    parser.add_argument("--offline", action="store_true", help="Skip Hugging Face account/repository access checks.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    try:
        report = run(
            data_root=args.data_root,
            budget_gb=args.budget_gb,
            require_cuda=not args.allow_cpu,
            offline=args.offline,
        )
    except ValueError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
