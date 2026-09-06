"""End-to-end local-file pipeline: load → validate → transform → manifest.

This is the only place that orders the steps. Scripts stay thin so the same
path can be unit-tested without the real IndicVoices / IndicSynth corpora.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from vaanirakshak.config import AppConfig, load_config, default_config_path
from vaanirakshak.data.audio_io import load_audio, save_audio
from vaanirakshak.data.metadata import ManifestRecord, append_manifest, make_sample_id
from vaanirakshak.data.preprocessing import convert_to_mono, resample_waveform, segment_waveform
from vaanirakshak.data.validation import validate_audio_data, validate_waveform
from vaanirakshak.exceptions import AudioValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    records: list[ManifestRecord]
    processed_files: list[Path]
    skipped_reason: str | None = None


def process_audio_file(
    input_path: str | Path,
    *,
    dataset: str,
    label: str,
    split: str,
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    gender: str | None = None,
    generator: str | None = None,
    attack_type: str | None = None,
    codec: str | None = None,
    write_files: bool = True,
    fail_on_invalid: bool = True,
) -> PipelineResult:
    """Run the full preprocessing path on one locally supplied file.

    Inputs
        input_path: original audio (left untouched).
        dataset/label/split: manifest fields. Label must be bonafide|spoof.
        Optional metadata: speaker, language, gender, generator, attack, codec.
    Outputs
        PipelineResult with one ManifestRecord per window and saved WAV paths.
    Edge cases
        Invalid source: raise if fail_on_invalid else return skipped_reason.
        ASVspoof-style external datasets log a training-use warning.
        Silent individual windows are not written.
    """
    cfg = config or load_config(config_path or default_config_path())
    source = Path(input_path)

    if dataset in cfg.datasets:
        spec = cfg.datasets[dataset]
        if not spec.use_for_training:
            logger.warning(
                "Dataset '%s' is marked role=%s and must not be used for training or tuning",
                dataset,
                spec.role,
            )

    audio = load_audio(source)
    report = validate_audio_data(audio, config=cfg.validation)
    if not report.ok:
        message = report.summary()
        if fail_on_invalid:
            raise AudioValidationError(message)
        return PipelineResult(records=[], processed_files=[], skipped_reason=message)

    if cfg.audio.mono:
        mono = convert_to_mono(audio.waveform)
    else:
        if audio.waveform.ndim == 2 and audio.waveform.shape[1] != 1:
            raise AudioValidationError("Config requests non-mono audio but pipeline is mono-only")
        mono = convert_to_mono(audio.waveform)

    resampled = resample_waveform(mono, audio.sample_rate, cfg.audio.sample_rate)
    post = validate_waveform(
        resampled,
        cfg.audio.sample_rate,
        num_channels=1,
        config=cfg.validation,
        path=source,
    )
    if not post.ok:
        message = "post-resample " + post.summary()
        if fail_on_invalid:
            raise AudioValidationError(message)
        return PipelineResult(records=[], processed_files=[], skipped_reason=message)

    segments = segment_waveform(
        resampled,
        cfg.audio.sample_rate,
        audio_config=cfg.audio,
        segmentation_config=cfg.segmentation,
    )
    if not segments:
        reason = "no segments produced (short_clip_policy=skip or empty audio)"
        logger.info("%s: %s", source, reason)
        return PipelineResult(records=[], processed_files=[], skipped_reason=reason)

    processed_root = Path(output_root) if output_root is not None else Path(cfg.paths.processed_data)
    dest_dir = processed_root / split
    records: list[ManifestRecord] = []
    processed_files: list[Path] = []

    for segment in segments:
        window_report = validate_waveform(
            segment.waveform,
            cfg.audio.sample_rate,
            num_channels=1,
            config=cfg.validation,
            path=source,
        )
        if not window_report.ok:
            logger.warning(
                "Dropping window %.3f-%.3f of %s (%s)",
                segment.start_seconds,
                segment.end_seconds,
                source,
                window_report.summary(),
            )
            continue

        sample_id = make_sample_id(
            dataset,
            source.stem,
            segment.start_seconds,
            segment.end_seconds,
        )
        dest = dest_dir / f"{sample_id}.wav"
        if write_files:
            save_audio(
                dest,
                segment.waveform,
                cfg.audio.sample_rate,
                subtype=cfg.audio.output_subtype,
            )
            processed_files.append(dest)
        else:
            processed_files.append(dest)

        records.append(
            ManifestRecord(
                sample_id=sample_id,
                source_file=str(source.resolve()),
                processed_file=str(dest),
                dataset=dataset,
                label=label,
                speaker_id=speaker_id,
                language=language,
                gender=gender,
                duration=float(segment.waveform.shape[0]) / float(cfg.audio.sample_rate),
                sample_rate=cfg.audio.sample_rate,
                segment_start=float(segment.start_seconds),
                segment_end=float(segment.end_seconds),
                generator=generator,
                attack_type=attack_type,
                codec=codec,
                split=split,
            )
        )

    if write_files and records:
        manifest = Path(manifest_path) if manifest_path is not None else Path(cfg.paths.manifests) / "manifest.csv"
        append_manifest(manifest, records)

    logger.info("Processed %s into %s windows", source, len(records))
    return PipelineResult(records=records, processed_files=processed_files)
