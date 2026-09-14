"""Audio normalization and quality primitives for VaaniRakshak V2.

This milestone intentionally uses an explainable energy-based speech gate rather
than pretending a simple heuristic is a production VAD. The interface is designed
so a neural VAD can replace it later without changing the session/risk layers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from math import gcd

import numpy as np
from scipy.signal import resample_poly


MODEL_SAMPLE_RATE = 16_000
FRAME_MS = 30
MIN_WINDOW_RMS_DBFS = -52.0
SPEECH_FRAME_RMS_DBFS = -45.0
MIN_SPEECH_RATIO = 0.30
MAX_CLIPPING_RATIO = 0.10


@dataclass(frozen=True)
class AudioQuality:
    rms_dbfs: float
    peak: float
    clipping_ratio: float
    speech_ratio: float
    usable: bool
    reason: str | None

    def as_dict(self) -> dict:
        return {
            "rms_dbfs": round(self.rms_dbfs, 3),
            "peak": round(self.peak, 6),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "speech_ratio": round(self.speech_ratio, 6),
            "usable": self.usable,
            "reason": self.reason,
        }


def pcm16_to_float32(samples: list[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    return np.clip(values / 32768.0, -1.0, 1.0)


def resample_audio(wave: np.ndarray, source_rate: int, target_rate: int = MODEL_SAMPLE_RATE) -> np.ndarray:
    """Resample one inference window using polyphase filtering.

    Resampling is deliberately performed on complete overlapping inference windows,
    not independently on tiny browser packets. This avoids packet-boundary artifacts
    while still allowing streaming inference as soon as a full window is available.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    wave = np.asarray(wave, dtype=np.float32)
    if source_rate == target_rate or wave.size == 0:
        return wave.copy()
    common = gcd(source_rate, target_rate)
    up = target_rate // common
    down = source_rate // common
    output = resample_poly(wave, up, down)
    return np.asarray(output, dtype=np.float32)


def assess_audio_quality(wave: np.ndarray, sample_rate: int) -> AudioQuality:
    """Estimate whether a window contains enough usable speech for spoof inference.

    This is a conservative *speech/quality gate*, not a semantic speech recognizer.
    A later milestone can swap this function for WebRTC/Silero/neural VAD while
    preserving the rest of the product pipeline.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return AudioQuality(-120.0, 0.0, 0.0, 0.0, False, "empty_audio")
    if not np.isfinite(wave).all():
        return AudioQuality(-120.0, 0.0, 0.0, 0.0, False, "invalid_samples")

    rms = float(np.sqrt(np.mean(np.square(wave), dtype=np.float64)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
    peak = float(np.max(np.abs(wave)))
    clipping_ratio = float(np.mean(np.abs(wave) >= 0.995))

    frame_samples = max(1, int(sample_rate * FRAME_MS / 1000))
    speech_frames = 0
    frame_count = 0
    for start in range(0, len(wave), frame_samples):
        frame = wave[start : start + frame_samples]
        if len(frame) < frame_samples // 2:
            continue
        frame_rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        frame_dbfs = 20.0 * math.log10(max(frame_rms, 1e-6))
        speech_frames += int(frame_dbfs >= SPEECH_FRAME_RMS_DBFS)
        frame_count += 1
    speech_ratio = speech_frames / frame_count if frame_count else 0.0

    reason = None
    if rms_dbfs < MIN_WINDOW_RMS_DBFS:
        reason = "too_quiet"
    elif clipping_ratio > MAX_CLIPPING_RATIO:
        reason = "excessive_clipping"
    elif speech_ratio < MIN_SPEECH_RATIO:
        reason = "insufficient_speech"

    return AudioQuality(
        rms_dbfs=rms_dbfs,
        peak=peak,
        clipping_ratio=clipping_ratio,
        speech_ratio=speech_ratio,
        usable=reason is None,
        reason=reason,
    )


def prepare_model_window(samples: list[int], source_rate: int) -> tuple[np.ndarray, AudioQuality]:
    """Convert browser PCM to normalized 16 kHz mono model input plus quality metadata."""
    wave = pcm16_to_float32(samples)
    quality = assess_audio_quality(wave, source_rate)
    normalized = resample_audio(wave, source_rate, MODEL_SAMPLE_RATE)
    return normalized, quality
