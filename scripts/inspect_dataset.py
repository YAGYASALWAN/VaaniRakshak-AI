#!/usr/bin/env python3
"""Print structured validation reports for locally supplied audio files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaanirakshak.config import default_config_path, load_config
from vaanirakshak.data.audio_io import load_audio
from vaanirakshak.data.validation import validate_audio_data
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
    parser.add_argument("--input", required=True, type=Path, help="File or directory")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

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
