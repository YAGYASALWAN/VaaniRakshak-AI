"""Unit tests for the dataset-engineering layer using synthetic waveforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from voiceshield.config import load_config
from voiceshield.data.audio_io import load_audio
from voiceshield.data.metadata import (
    MANIFEST_COLUMNS,
    ManifestRecord,
    append_manifest,
    records_to_frame,
    validate_manifest_frame,
)
from voiceshield.data.pipeline import process_audio_file
from voiceshield.data.preprocessing import convert_to_mono, resample_waveform, segment_waveform
from voiceshield.data.splits import generator_disjoint_split, speaker_disjoint_split
from voiceshield.data.validation import validate_waveform
from voiceshield.exceptions import ManifestSchemaError, SplitError


def _tone(duration_s: float, sr: int, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr)), dtype=np.float32) / float(sr)
    return (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_stereo_to_mono_averages_channels() -> None:
    left = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    right = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    mono = convert_to_mono(stereo)
    np.testing.assert_allclose(mono, np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_resample_to_16khz() -> None:
    orig_sr = 48000
    target_sr = 16000
    waveform = _tone(1.0, orig_sr)
    resampled = resample_waveform(waveform, orig_sr, target_sr)
    expected = int(round(len(waveform) * target_sr / orig_sr))
    assert resampled.shape[0] == expected
    assert resampled.dtype == np.float32
    # Duration is preserved: 1.0 s ± one sample.
    assert abs(resampled.shape[0] / target_sr - 1.0) < 2.0 / target_sr


def test_exact_segment_length() -> None:
    sr = 16000
    waveform = _tone(10.0, sr)
    segments = segment_waveform(
        waveform,
        sr,
        window_seconds=4.0,
        hop_seconds=2.0,
        short_clip_policy="pad",
    )
    assert all(seg.waveform.shape[0] == 64000 for seg in segments)


def test_overlapping_segmentation_matches_expected_starts() -> None:
    sr = 16000
    waveform = _tone(10.0, sr)
    segments = segment_waveform(
        waveform,
        sr,
        window_seconds=4.0,
        hop_seconds=2.0,
        short_clip_policy="skip",
    )
    starts = [round(seg.start_seconds, 5) for seg in segments]
    assert starts == [0.0, 2.0, 4.0, 6.0]
    ends = [round(seg.end_seconds, 5) for seg in segments]
    assert ends == [4.0, 6.0, 8.0, 10.0]


def test_short_sample_pad_and_skip() -> None:
    sr = 16000
    short = _tone(1.0, sr)
    padded = segment_waveform(
        short,
        sr,
        window_seconds=4.0,
        hop_seconds=2.0,
        short_clip_policy="pad",
    )
    assert len(padded) == 1
    assert padded[0].waveform.shape[0] == 64000
    assert padded[0].padded is True
    # Energy should still sit at the beginning (end-padding).
    assert float(np.max(np.abs(padded[0].waveform[:sr]))) > 0.0
    assert float(np.max(np.abs(padded[0].waveform[-sr:]))) == 0.0

    skipped = segment_waveform(
        short,
        sr,
        window_seconds=4.0,
        hop_seconds=2.0,
        short_clip_policy="skip",
    )
    assert skipped == []


def test_nan_validation_fails() -> None:
    waveform = _tone(0.5, 16000)
    waveform[10] = np.nan
    report = validate_waveform(waveform, 16000)
    assert report.ok is False
    names = {check.name: check.passed for check in report.checks}
    assert names["finite_samples"] is False


def test_silence_validation_fails() -> None:
    silent = np.zeros(16000, dtype=np.float32)
    report = validate_waveform(silent, 16000)
    assert report.ok is False
    names = {check.name: check.passed for check in report.checks}
    assert names["not_completely_silent"] is False


def test_manifest_schema_and_illegal_labels(tmp_path: Path) -> None:
    record = ManifestRecord(
        sample_id="local__x__0.000_4.000",
        source_file="in.wav",
        processed_file="out.wav",
        dataset="local",
        label="bonafide",
        speaker_id=None,
        language=None,
        gender=None,
        duration=4.0,
        sample_rate=16000,
        segment_start=0.0,
        segment_end=4.0,
        generator=None,
        attack_type=None,
        codec=None,
        split="train",
    )
    frame = records_to_frame([record])
    validate_manifest_frame(frame)
    assert list(frame.columns) == list(MANIFEST_COLUMNS)

    dest = tmp_path / "manifest.csv"
    append_manifest(dest, [record])
    loaded = pd.read_csv(dest)
    validate_manifest_frame(loaded)

    with pytest.raises(ManifestSchemaError):
        ManifestRecord(
            sample_id="bad",
            source_file="in.wav",
            processed_file="out.wav",
            dataset="local",
            label="real",
            speaker_id=None,
            language=None,
            gender=None,
            duration=4.0,
            sample_rate=16000,
            segment_start=0.0,
            segment_end=4.0,
            generator=None,
            attack_type=None,
            codec=None,
            split="train",
        )


def _toy_manifest() -> pd.DataFrame:
    rows = []
    for speaker in ("spk_a", "spk_b", "spk_c", "spk_d", "spk_e", "spk_f"):
        for idx in range(2):
            rows.append(
                ManifestRecord(
                    sample_id=f"{speaker}_{idx}",
                    source_file=f"{speaker}.wav",
                    processed_file=f"{speaker}_{idx}.wav",
                    dataset="local",
                    label="bonafide" if speaker < "spk_d" else "spoof",
                    speaker_id=speaker,
                    language="hi",
                    gender=None,
                    duration=4.0,
                    sample_rate=16000,
                    segment_start=float(idx * 2),
                    segment_end=float(idx * 2 + 4),
                    generator="XTTS" if speaker >= "spk_d" else None,
                    attack_type=None,
                    codec=None,
                    split="unassigned",
                )
            )
    return records_to_frame(rows)


def test_speaker_disjoint_split_is_deterministic() -> None:
    frame = _toy_manifest()
    first = speaker_disjoint_split(frame, seed=7)
    second = speaker_disjoint_split(frame, seed=7)
    third = speaker_disjoint_split(frame, seed=11)
    pd.testing.assert_series_equal(first["split"], second["split"], check_names=False)
    assert not first["split"].equals(third["split"]) or first["split"].nunique() == 1

    def speakers(split: str) -> set[str]:
        return set(first.loc[first["split"] == split, "speaker_id"])

    assert speakers("train").isdisjoint(speakers("dev"))
    assert speakers("train").isdisjoint(speakers("test"))
    assert speakers("dev").isdisjoint(speakers("test"))


def test_generator_disjoint_rejects_overlap() -> None:
    frame = _toy_manifest()
    with pytest.raises(SplitError):
        generator_disjoint_split(
            frame,
            train_generators=["XTTS"],
            test_generators=["XTTS"],
        )


def test_pipeline_on_synthetic_wav(tmp_path: Path) -> None:
    sr = 22050
    stereo = np.stack([_tone(5.0, sr, 180.0), _tone(5.0, sr, 240.0)], axis=1)
    source = tmp_path / "example.wav"
    sf.write(source, stereo, sr)

    processed = tmp_path / "processed"
    manifest = tmp_path / "manifest.csv"
    result = process_audio_file(
        source,
        dataset="local",
        label="bonafide",
        split="train",
        config=load_config(Path("configs/data_config.yaml")),
        output_root=processed,
        manifest_path=manifest,
        speaker_id="spk_demo",
        language="en",
    )
    assert result.skipped_reason is None
    assert result.records
    assert all(rec.sample_rate == 16000 for rec in result.records)
    assert all(rec.duration == pytest.approx(4.0) for rec in result.records)
    assert all(Path(p).is_file() for p in result.processed_files)
    reloaded = load_audio(result.processed_files[0])
    assert reloaded.num_channels == 1
    assert reloaded.sample_rate == 16000
    table = pd.read_csv(manifest)
    validate_manifest_frame(table)


def test_load_config_yaml(tmp_path: Path) -> None:
    payload = {
        "audio": {"sample_rate": 16000, "window_seconds": 4.0, "hop_seconds": 2.0},
        "segmentation": {"short_clip_policy": "pad"},
        "splits": {"train_ratio": 0.8, "dev_ratio": 0.1, "test_ratio": 0.1, "seed": 1},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.audio.sample_rate == 16000
    assert cfg.segmentation.short_clip_policy == "pad"
