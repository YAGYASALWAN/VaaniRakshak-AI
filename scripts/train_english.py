#!/usr/bin/env python3
"""Load full English ASVspoof 2019 LA from Hugging Face, train, then evaluate."""
import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/english_asv2019"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", type=Path, help="Existing models/english_... directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--predict", type=Path, help="Mono 16 kHz WAV/FLAC to classify")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 64:
        parser.error("Use 1–100 epochs and a batch size of 1–64")
    try:
        from vaanirakshak.english_training import predict, run
        if args.predict:
            if not args.checkpoint:
                parser.error("--predict requires --checkpoint")
            predict(args.checkpoint, args.predict, args.device)
        else:
            run(args.data, args.epochs, args.batch_size, args.resume, args.device)
    except KeyboardInterrupt:
        print("\nStopped. Completed feature batches and epoch checkpoints remain saved.", file=sys.stderr)
        return 130
    except Exception:
        # Keep the real traceback: internal PyTorch faults are not missing requirements.
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
