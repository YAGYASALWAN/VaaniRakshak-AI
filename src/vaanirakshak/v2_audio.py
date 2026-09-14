"""Audio normalization, speech gating and quality primitives for VaaniRakshak V2.

Speech detection is deliberately separate from anti-spoof inference. The product can
use WebRTC VAD for frame-level speech gating while retaining the original energy gate
as an explicit fallback/testing implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from math import gcd
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly


MODEL_SAMPLE_RATE = 16_000
FRAME_MS = 30
MIN_WINDOW_RMS_DBFS = -52.0
SPEECH_FRAME_RMS_DBFS = -45.0
MIN_SPEECH_RATIO = 0.30
MAX_CLIPPING_RATIO = 0.10


class SpeechGate(Protocol):
    name: str

    def speech_ratio(self, wave: np.ndarray, sample_rate: int) -> float:
        """Return the fraction of analyzed frames classified as speech in [0, 1]."""


@dataclass(frozen=True)
class AudioQuality:
    rms_dbfs: float
    peak: float
    clipping_ratio: float
    speech_ratio: float
    usable: bool
    reason: str | None
    speech_gate: str = "energy-v1"

    def as_dict(self) -> dict:
        return {
            "rms_dbfs": round(self.rms_dbfs, 3),
            "peak": round(self.peak, 6),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "speech_ratio": round(self.speech_ratio, 6),
            "usable": self.usable,
            "reason": self.reason,
            "speech_gate": self.speech_gate,
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


class EnergySpeechGate:
    """Original explainable amplitude gate retained for fallback and testing."""

    name = "energy-v1"

    def speech_ratio(self, wave: np.ndarray, sample_rate: int) -> float:
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
        return speech_frames / frame_count if frame_count else 0.0


class WebRTCSpeechGate:
    """Frame-level WebRTC VAD gate using 30 ms mono PCM16 frames at 16 kHz."""

    def __init__(self, aggressiveness: int = 2):
        if aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("WebRTC VAD aggressiveness must be 0, 1, 2 or 3")
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "WebRTC VAD is unavailable. Install requirements-v2.txt or select VAANIRAKSHAK_V2_VAD=energy."
            ) from exc
        self._module = webrtcvad
        self.aggressiveness = aggressiveness
        self.name = f"webrtc-vad-m{aggressiveness}"

    def speech_ratio(self, wave: np.ndarray, sample_rate: int) -> float:
        normalized = resample_audio(np.asarray(wave, dtype=np.float32), sample_rate, MODEL_SAMPLE_RATE)
        frame_samples = int(MODEL_SAMPLE_RATE * FRAME_MS / 1000)
        if len(normalized) < frame_samples:
            return 0.0

        pcm = np.clip(np.rint(normalized * 32767.0), -32768, 32767).astype("<i2")
        vad = self._module.Vad(self.aggressiveness)
        speech_frames = 0
        frame_count = 0
        for start in range(0, len(pcm) - frame_samples + 1, frame_samples):
            frame = pcm[start : start + frame_samples]
            speech_frames += int(vad.is_speech(frame.tobytes(), MODEL_SAMPLE_RATE))
            frame_count += 1
        return speech_frames / frame_count if frame_count else 0.0


_ENERGY_GATE = EnergySpeechGate()


def build_speech_gate(mode: str = "webrtc", *, aggressiveness: int = 2) -> SpeechGate:
    normalized = str(mode or "").strip().lower()
    if normalized in {"energy", "energy-v1"}:
        return _ENERGY_GATE
    if normalized in {"webrtc", "webrtc-vad"}:
        return WebRTCSpeechGate(aggressiveness=aggressiveness)
    raise ValueError("speech gate must be 'webrtc' or 'energy'")


def assess_audio_quality(
    wave: np.ndarray,
    sample_rate: int,
    *,
    speech_gate: SpeechGate | None = None,
) -> AudioQuality:
    """Estimate whether a window contains enough usable speech for spoof inference."""
    gate = speech_gate or _ENERGY_GATE
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return AudioQuality(-120.0, 0.0, 0.0, 0.0, False, "empty_audio", gate.name)
    if not np.isfinite(wave).all():
        return AudioQuality(-120.0, 0.0, 0.0, 0.0, False, "invalid_samples", gate.name)

    rms = float(np.sqrt(np.mean(np.square(wave), dtype=np.float64)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
    peak = float(np.max(np.abs(wave)))
    clipping_ratio = float(np.mean(np.abs(wave) >= 0.995))
    speech_ratio = float(gate.speech_ratio(wave, sample_rate))
    if not math.isfinite(speech_ratio) or not 0.0 <= speech_ratio <= 1.0:
        raise ValueError("Speech gate returned an invalid ratio")

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
        speech_gate=gate.name,
    )


def prepare_model_window(
    samples: list[int],
    source_rate: int,
    *,
    speech_gate: SpeechGate | None = None,
) -> tuple[np.ndarray, AudioQuality]:
    """Convert browser PCM to normalized 16 kHz mono model input plus quality metadata."""
    wave = pcm16_to_float32(samples)
    quality = assess_audio_quality(wave, source_rate, speech_gate=speech_gate)
    normalized = resample_audio(wave, source_rate, MODEL_SAMPLE_RATE)
    return normalized, quality
