#!/usr/bin/env python3
"""Print structured validation reports for locally supplied audio files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaanirakshak.config import default_config_path, load_config
from vaanirakshak.exceptions import AudioLoadError

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a"}


def _iter_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES:
            files.append(child)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local audio without modifying it.")
    parser.add_argument("--input", type=Path, help="Local audio file or directory")
    parser.add_argument("--dataset", choices=("indicvoices", "indicsynth", "asvspoof"))
    parser.add_argument("--metadata", type=Path, help="Local metadata CSV/JSON/JSONL")
    parser.add_argument("--language", help="Verified language of this metadata export")
    parser.add_argument("--locator-prefix", help="Revision/config/split identifying row positions")
    parser.add_argument("--field-map", type=Path, help="JSON canonical-field to source-key mapping")
    parser.add_argument("--audio-root", type=Path, help="Explicitly inspect audio under this root")
    parser.add_argument("--output", type=Path, default=Path("data/reports/inspection"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.dataset:
        import json
        from vaanirakshak.data.adapters import adapt
        from vaanirakshak.data.metadata_io import read_rows
        from vaanirakshak.data.inspection import inspect_local_audio
        from vaanirakshak.data.reporting import write_reports
        if not args.metadata:
            parser.error("--dataset requires --metadata; no automatic downloads")
        mapping = json.loads(args.field_map.read_text()) if args.field_map else None
        frame = adapt(read_rows(args.metadata), args.dataset, language=args.language,
                      locator_prefix=args.locator_prefix, field_map=mapping)
        if args.audio_root:
            frame = inspect_local_audio(frame, args.audio_root)
        summary = write_reports(frame, args.output, plots=args.plots)
        frame.to_csv(args.output / "normalized_metadata.csv", index=False)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    if not args.input:
        parser.error("Provide --input for audio or --dataset and --metadata")
    from vaanirakshak.data.audio_io import load_audio
    from vaanirakshak.data.validation import validate_audio_data

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config or default_config_path())
    files = _iter_audio(args.input)
    if not files:
        logging.error("No audio files found at %s", args.input)
        return 1

    n_ok = 0
    n_fail = 0
    for file_path in files:
        try:
            audio = load_audio(file_path)
        except AudioLoadError as exc:
            n_fail += 1
            print(f"LOAD_FAIL\t{file_path}\t{exc}")
            continue
        report = validate_audio_data(audio, config=config.validation)
        status = "OK" if report.ok else "INVALID"
        if report.ok:
            n_ok += 1
        else:
            n_fail += 1
        print(
            f"{status}\t{file_path}\tsr={audio.sample_rate}\t"
            f"ch={audio.num_channels}\tdur={audio.duration_seconds:.4f}\t"
            f"{report.summary()}"
        )
    print(f"summary\tok={n_ok}\tinvalid={n_fail}\ttotal={len(files)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
