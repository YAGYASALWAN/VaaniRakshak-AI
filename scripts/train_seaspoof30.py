"""Run English SEA-Spoof preparation/training with a cumulative 30 GB ceiling."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sea30_en")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 64:
        parser.error("Use 1–100 epochs and batch size 1–64")
    lock = None
    try:
        from huggingface_hub import get_token
        import torch
        from vaanirakshak.english_training import run
        from vaanirakshak.sea_transfer import Transfer, REPOSITORY, REVISION
        from vaanirakshak.sea_training import prepare, profile
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Use your working GPU Python environment.")
        torch.set_num_threads(4)
        source = json.loads((ROOT / "configs/seaspoof_source.json").read_text())
        if source["repository"] != REPOSITORY or source["revision"] != REVISION:
            raise ValueError("Dataset source configuration mismatch")
        args.data.mkdir(parents=True, exist_ok=True)
        # OS-level lock is released on process exit, including abnormal termination.
        from filelock import FileLock
        lock = FileLock(str(args.data / "run.lock"))
        lock.acquire(timeout=0)
        client = Transfer(args.data, get_token())
        print("SEA-Spoof: English only; at most 30 GB total fetched payload. No full-dataset download.", flush=True)
        print("Inspecting metadata first. The chosen English recording count will be printed before audio preparation.", flush=True)
        rows, counts = prepare(args.data, source, client, args.device)
        run(args.data, args.epochs, args.batch_size, args.resume, args.device, profile(args.data, rows, counts))
    except KeyboardInterrupt:
        print("Stopped. Completed groups and epoch checkpoints remain available for resume.")
        return 130
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        if lock is not None and lock.is_locked:
            lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
