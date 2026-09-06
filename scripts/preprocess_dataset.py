#!/usr/bin/env python3
"""Preprocess one locally supplied audio file into 16 kHz mono windows + manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaanirakshak.config import default_config_path, load_config
from vaanirakshak.data.pipeline import process_audio_file
from vaanirakshak.exceptions import AudioLoadError, AudioValidationError, ManifestSchemaError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the VoiceShield preprocessing pipeline on a local file."
    )
    parser.add_argument("--input", required=True, type=Path, help="Original audio file (not modified)")
    parser.add_argument("--dataset", default="local", help="Dataset name stored in the manifest")
    parser.add_argument(
        "--label",
        required=True,
        choices=("bonafide", "spoof"),
        help="Canonical class label",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "dev", "test"),
        help="Destination processed/ folder. For a smoke test this is not a real speaker split.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--generator", default=None, help="TTS / VC system name if spoofed")
    parser.add_argument("--attack-type", default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config or default_config_path())

    try:
        result = process_audio_file(
            args.input,
            dataset=args.dataset,
            label=args.label,
            split=args.split,
            config=config,
            output_root=args.output_root,
            manifest_path=args.manifest,
            speaker_id=args.speaker_id,
            language=args.language,
            gender=args.gender,
            generator=args.generator,
            attack_type=args.attack_type,
            codec=args.codec,
        )
    except (AudioLoadError, AudioValidationError, ManifestSchemaError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        return 1

    if result.skipped_reason:
        logging.error("Skipped: %s", result.skipped_reason)
        return 1

    print(f"windows={len(result.records)}")
    for record, path in zip(result.records, result.processed_files):
        print(
            f"{record.sample_id}\t{path}\t"
            f"{record.segment_start:.3f}-{record.segment_end:.3f}\t{record.label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
