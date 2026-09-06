"""Structured validation of waveforms and decoded files.

Validation never "fixes" audio. It reports which checks passed or failed so
callers can skip, quarantine, or abort. Silent failure would hide dataset bugs
that later look like model errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vaanirakshak.config import ValidationConfig
from vaanirakshak.data.audio_io import AudioData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str
    value: Any = None


@dataclass
class ValidationReport:
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    path: Path | None = None

    def failed(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        if self.ok:
            return "validation passed"
        parts = [f"{c.name}: {c.message}" for c in self.failed()]
        location = str(self.path) if self.path is not None else "<in-memory>"
        return f"validation failed for {location}: " + "; ".join(parts)


def _rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _non_silent_frame_ratio(
    mono: np.ndarray,
    sample_rate: int,
    frame_seconds: float,
    rms_threshold: float,
) -> float:
    """Fraction of short frames whose RMS exceeds ``rms_threshold``.

    We look at *frames*, not the whole-file RMS, so a recording that is silent
    except for one click is still flagged as almost completely silent.
    """
    frame_length = max(1, int(round(frame_seconds * sample_rate)))
    n_samples = int(mono.shape[0])
    if n_samples == 0:
        return 0.0

    n_frames = int(np.ceil(n_samples / frame_length))
    energies = []
    for i in range(n_frames):
        frame = mono[i * frame_length : (i + 1) * frame_length]
        energies.append(_rms(frame) > rms_threshold)
    return float(np.mean(energies)) if energies else 0.0


def _as_mono_view(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        # soundfile layout: (samples, channels)
        if array.shape[1] <= array.shape[0] or array.shape[1] <= 8:
            return np.mean(array, axis=1)
        return np.mean(array, axis=0)
    raise ValueError(f"Waveform must be 1-D or 2-D, got shape {array.shape}")


def validate_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    num_channels: int | None = None,
    config: ValidationConfig | None = None,
    path: Path | None = None,
) -> ValidationReport:
    """Run the full check list on an in-memory waveform.

    Inputs
        waveform: samples, 1-D mono or 2-D (samples, channels).
        sample_rate: claimed Hz of this array.
        num_channels: optional explicit channel count for the channels check.
        config: thresholds from YAML.
        path: used only for log / report context.
    Outputs
        ValidationReport with ``ok`` and per-check CheckResult rows.
    Edge cases
        Empty arrays, NaN/Inf from broken decoders, sr <= 0, duration below
        ``min_duration_seconds``, digital zeros, and near-silent recordings.
    """
    cfg = config or ValidationConfig()
    checks: list[CheckResult] = []
    array = np.asarray(waveform)

    if array.ndim == 0:
        checks.append(CheckResult("shape", False, "waveform is a scalar", array.shape))
        report = ValidationReport(ok=False, checks=checks, path=path)
        logger.warning("%s", report.summary())
        return report

    n_samples = int(array.shape[0]) if array.ndim >= 1 else 0
    inferred_channels = 1 if array.ndim == 1 else int(array.shape[1])
    channels = num_channels if num_channels is not None else inferred_channels

    checks.append(
        CheckResult(
            "readable_array",
            array.ndim in {1, 2},
            "waveform rank is 1 or 2" if array.ndim in {1, 2} else f"bad rank {array.ndim}",
            array.shape,
        )
    )
    checks.append(
        CheckResult(
            "sample_rate",
            int(sample_rate) > 0,
            f"sample_rate={sample_rate}",
            sample_rate,
        )
    )
    checks.append(
        CheckResult(
            "empty_waveform",
            n_samples > 0,
            f"num_samples={n_samples}",
            n_samples,
        )
    )

    finite = bool(array.size == 0 or np.isfinite(array).all())
    checks.append(
        CheckResult(
            "finite_samples",
            finite,
            "all samples finite" if finite else "NaN or Inf present",
            None if finite else "non-finite",
        )
    )

    duration = (n_samples / float(sample_rate)) if sample_rate > 0 else 0.0
    checks.append(
        CheckResult(
            "min_duration",
            duration >= cfg.min_duration_seconds,
            f"duration={duration:.6f}s threshold={cfg.min_duration_seconds}s",
            duration,
        )
    )
    checks.append(
        CheckResult(
            "channels",
            channels >= 1,
            f"num_channels={channels}",
            channels,
        )
    )

    try:
        mono = _as_mono_view(array) if array.size else np.array([], dtype=np.float32)
    except ValueError as exc:
        checks.append(CheckResult("channel_layout", False, str(exc), array.shape))
        report = ValidationReport(ok=False, checks=checks, path=path)
        logger.warning("%s", report.summary())
        return report

    rms = _rms(mono)
    checks.append(
        CheckResult(
            "not_completely_silent",
            rms >= cfg.silence_rms_threshold,
            f"rms={rms:.8f} threshold={cfg.silence_rms_threshold}",
            rms,
        )
    )

    ratio = _non_silent_frame_ratio(
        mono,
        sample_rate=max(sample_rate, 1),
        frame_seconds=cfg.silence_frame_seconds,
        rms_threshold=cfg.silence_rms_threshold,
    )
    checks.append(
        CheckResult(
            "not_almost_silent",
            ratio >= cfg.min_non_silent_ratio,
            f"non_silent_frame_ratio={ratio:.4f} threshold={cfg.min_non_silent_ratio}",
            ratio,
        )
    )

    ok = all(check.passed for check in checks)
    report = ValidationReport(ok=ok, checks=checks, path=path)
    if ok:
        logger.debug("%s (%s)", report.summary(), path)
    else:
        logger.warning("%s", report.summary())
    return report


def validate_audio_data(
    audio: AudioData,
    *,
    config: ValidationConfig | None = None,
) -> ValidationReport:
    """Validate a decoded file, including channel count from the container."""
    return validate_waveform(
        audio.waveform,
        audio.sample_rate,
        num_channels=audio.num_channels,
        config=config,
        path=audio.path,
    )
