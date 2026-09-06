#!/usr/bin/env python3
"""Download a bounded original-audio pilot, train a small CNN, or predict a clip."""
import argparse
import getpass
from pathlib import Path
import sys
import warnings

# Allow direct execution from a checkout, without reinstalling the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "prepare", "train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/baseline"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 64:
        parser.error("Epochs must be 1–100 and batch size 1–64")
    try:
        from vaanirakshak.baseline import predict, prepare, train
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; check your PyTorch installation")
        manifest = args.data / "manifest.json"
        if args.command in ("all", "prepare"):
            token = None
            if not manifest.exists():
                import pyarrow.parquet  # Check dependency before prompting or downloading.
                print("This run downloads up to 8 GiB of original shards for Hindi/Punjabi. "
                      "Allow about 12 GiB free disk space. Completed shards are reused.", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    token = getpass.getpass("Hugging Face read token (hidden): ")
            manifest = prepare(args.data, token)
        if args.command in ("all", "train"):
            train(manifest, args.epochs, args.batch_size, args.device)
        if args.command == "predict":
            if not args.checkpoint or not args.audio:
                parser.error("predict requires --checkpoint and --audio")
            predict(args.checkpoint, args.audio, args.device)
    except (getpass.GetPassWarning, EOFError):
        parser.exit(2, "Hidden input requires an interactive terminal.\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nStopped. Completed downloads and saved checkpoints remain available.\n")
    except ImportError as exc:
        parser.exit(2, f"Missing dependency: {exc.name}. Install requirements-baseline.txt.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Baseline stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
