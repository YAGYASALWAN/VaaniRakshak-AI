"""VaaniRakshak V2 streaming analysis primitives.

The product/session logic remains independent of the anti-spoofing model. Incoming
browser PCM is framed with overlap, speech/quality-gated, normalized to 16 kHz and
only then passed to a detector. The detector and speech gate are separate pluggable
components.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, field
import hashlib
import math
import statistics
import time
from typing import Iterable, Protocol

import numpy as np

from vaanirakshak.v2_audio import AudioQuality, MODEL_SAMPLE_RATE, SpeechGate, prepare_model_window


WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
MIN_FINAL_WINDOW_SECONDS = 2.0
MIN_NEW_TAIL_SECONDS = 0.5
MIN_ANALYZED_WINDOWS = 2
MIN_USABLE_SPEECH_SECONDS = 4.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _threshold_relative(score: float, threshold: float) -> float:
    """Map a detector score to [0, 1] with its operating threshold at 0.5.

    This is NOT probability calibration. It only prevents the product risk engine
    from assuming that identical raw scores from detectors with different decision
    thresholds carry identical meaning.
    """
    if score >= threshold:
        return 0.5 + 0.5 * (score - threshold) / max(1e-9, 1.0 - threshold)
    return 0.5 * score / max(1e-9, threshold)


class Detector(Protocol):
    """Interface every trained or mock V2 detector must implement."""

    name: str
    mode: str
    threshold: float
    calibrated_probability: bool
    notice: str

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        """Return a finite synthetic-speech score in [0, 1].

        `samples` are mono float32 samples normalized to [-1, 1] at 16 kHz.
        The score is compared with the detector's own validated operating
        threshold; the product layer must not invent a model threshold.
        """


class MockDetector:
    """Deterministic placeholder used only to exercise the V2 product pipeline."""

    name = "v2-product-mock"
    mode = "mock"
    threshold = 0.65
    calibrated_probability = False
    notice = (
        "V2 product-skeleton mode uses a deterministic mock detector. Risk and verdict values are "
        "UI/integration test data, not voice-authenticity findings."
    )

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError("Detector expected 16 kHz model input")
        wave = np.asarray(samples, dtype=np.float32)
        if wave.size == 0:
            return 0.5

        rms = float(np.sqrt(np.mean(np.square(wave), dtype=np.float64)))
        peak = float(np.max(np.abs(wave)))
        crossings = int(np.count_nonzero(np.diff(np.signbit(wave))))
        zcr = crossings / max(1, len(wave) - 1)
        crest = peak / max(rms, 1e-6)

        quantized = np.clip(np.rint(wave[:4096] * 32767.0), -32768, 32767).astype("<i2")
        digest = hashlib.blake2s(quantized.tobytes(), digest_size=2).digest()
        jitter = (int.from_bytes(digest, "little") / 65535.0 - 0.5) * 0.18

        score = 0.48
        score += _clamp((zcr - 0.03) / 0.18, -0.5, 0.5) * 0.10
        score += _clamp((2.5 - crest) / 4.0, -0.5, 0.5) * 0.08
        score += _clamp((0.05 - rms) / 0.12, -0.5, 0.5) * 0.05
        score += jitter
        return _clamp(score, 0.05, 0.95)


@dataclass(frozen=True)
class WindowResult:
    index: int
    start_seconds: float
    end_seconds: float
    synthetic_score: float | None
    quality: AudioQuality
    threshold: float
    inference_ms: float | None = None

    @property
    def analyzed(self) -> bool:
        return self.synthetic_score is not None

    @property
    def suspicious(self) -> bool:
        return bool(self.analyzed and float(self.synthetic_score) >= self.threshold)

    def as_dict(self) -> dict:
        evidence_signal = None
        if self.synthetic_score is not None:
            evidence_signal = _threshold_relative(float(self.synthetic_score), self.threshold)
        return {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "synthetic_score": None if self.synthetic_score is None else round(self.synthetic_score, 6),
            "evidence_signal": None if evidence_signal is None else round(evidence_signal, 6),
            "threshold": round(self.threshold, 6),
            "analyzed": self.analyzed,
            "suspicious": self.suspicious,
            "quality": self.quality.as_dict(),
            "inference_ms": None if self.inference_ms is None else round(self.inference_ms, 3),
        }


@dataclass
class StreamingSession:
    detector: Detector
    sample_rate: int
    speech_gate: SpeechGate | None = None
    created_at: float = field(default_factory=time.monotonic)
    total_samples: int = 0
    buffer_start_sample: int = 0
    buffer: array = field(default_factory=lambda: array("h"))
    windows: list[WindowResult] = field(default_factory=list)
    finalized: bool = False

    def __post_init__(self) -> None:
        if not 8_000 <= self.sample_rate <= 96_000:
            raise ValueError("sample_rate must be between 8000 and 96000 Hz")
        threshold = float(getattr(self.detector, "threshold", float("nan")))
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("Detector must expose a finite threshold between 0 and 1")

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * WINDOW_SECONDS)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * HOP_SECONDS)

    def ingest_pcm16le(self, payload: bytes) -> list[WindowResult]:
        if self.finalized:
            raise RuntimeError("Session is already finalized")
        if not payload:
            return []
        if len(payload) % 2:
            raise ValueError("PCM payload must contain complete 16-bit samples")

        chunk = array("h")
        chunk.frombytes(payload)
        if chunk.itemsize != 2:
            raise RuntimeError("Unsupported Python short size")
        import sys
        if sys.byteorder != "little":
            chunk.byteswap()

        self.buffer.extend(chunk)
        self.total_samples += len(chunk)
        emitted: list[WindowResult] = []
        while len(self.buffer) >= self.window_samples:
            window = list(self.buffer[: self.window_samples])
            emitted.append(self._analyze_window(window, self.buffer_start_sample))
            del self.buffer[: self.hop_samples]
            self.buffer_start_sample += self.hop_samples
        return emitted

    def _analyze_window(self, samples: list[int], start_sample: int) -> WindowResult:
        model_wave, quality = prepare_model_window(
            samples,
            self.sample_rate,
            speech_gate=self.speech_gate,
        )
        score: float | None = None
        inference_ms: float | None = None

        if quality.usable:
            started = time.perf_counter()
            score = float(self.detector.score(model_wave, MODEL_SAMPLE_RATE))
            inference_ms = (time.perf_counter() - started) * 1000.0
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Detector returned an invalid score")

        result = WindowResult(
            index=len(self.windows),
            start_seconds=start_sample / self.sample_rate,
            end_seconds=(start_sample + len(samples)) / self.sample_rate,
            synthetic_score=score,
            quality=quality,
            threshold=float(self.detector.threshold),
            inference_ms=inference_ms,
        )
        self.windows.append(result)
        return result

    def finalize(self) -> dict:
        if self.finalized:
            return aggregate_call(self.windows, self.total_samples / self.sample_rate, self.detector)
        self.finalized = True

        min_tail = int(self.sample_rate * MIN_FINAL_WINDOW_SECONDS)
        min_new = int(self.sample_rate * MIN_NEW_TAIL_SECONDS)
        tail_end_sample = self.buffer_start_sample + len(self.buffer)
        last_end_sample = int(round(self.windows[-1].end_seconds * self.sample_rate)) if self.windows else 0
        new_tail_samples = tail_end_sample - last_end_sample if self.windows else len(self.buffer)
        if len(self.buffer) >= min_tail and new_tail_samples >= min_new:
            self._analyze_window(list(self.buffer), self.buffer_start_sample)
        self.buffer = array("h")
        return aggregate_call(self.windows, self.total_samples / self.sample_rate, self.detector)

    def live_summary(self) -> dict:
        return aggregate_call(self.windows, self.total_samples / self.sample_rate, self.detector)


def _estimate_unique_speech_seconds(items: list[WindowResult]) -> float:
    if not items:
        return 0.0
    ordered = sorted(items, key=lambda item: item.start_seconds)
    covered_until = 0.0
    speech = 0.0
    for item in ordered:
        unique_start = max(item.start_seconds, covered_until)
        unique_duration = max(0.0, item.end_seconds - unique_start)
        speech += unique_duration * item.quality.speech_ratio
        covered_until = max(covered_until, item.end_seconds)
    return speech


def aggregate_call(windows: Iterable[WindowResult], duration_seconds: float, detector: Detector) -> dict:
    items = list(windows)
    analyzed = [item for item in items if item.synthetic_score is not None]
    skipped = [item for item in items if item.synthetic_score is None]
    scores = [float(item.synthetic_score) for item in analyzed]
    evidence_scores = [_threshold_relative(float(item.synthetic_score), item.threshold) for item in analyzed]
    suspicious = [item for item in analyzed if item.suspicious]
    usable_speech_seconds = _estimate_unique_speech_seconds(items)

    if scores:
        median_score = statistics.median(scores)
        mean_score = statistics.fmean(scores)
        median_evidence = statistics.median(evidence_scores)
        mean_evidence = statistics.fmean(evidence_scores)
        suspicious_ratio = len(suspicious) / len(scores)
        dispersion = statistics.pstdev(evidence_scores) if len(evidence_scores) > 1 else 0.0
        consistency = _clamp(1.0 - dispersion / 0.30)
        risk_fraction = _clamp(
            0.50 * median_evidence
            + 0.25 * mean_evidence
            + 0.20 * suspicious_ratio
            + 0.05 * consistency
        )
        risk_score = round(risk_fraction * 100)
    else:
        median_score = mean_score = 0.0
        median_evidence = mean_evidence = suspicious_ratio = 0.0
        consistency = 0.0
        risk_score = 0

    enough_evidence = len(analyzed) >= MIN_ANALYZED_WINDOWS and usable_speech_seconds >= MIN_USABLE_SPEECH_SECONDS
    if not enough_evidence:
        risk_label = "Collecting evidence" if duration_seconds < MIN_USABLE_SPEECH_SECONDS else "Insufficient evidence"
        verdict = "Insufficient evidence"
    elif risk_score >= 81:
        risk_label = "High risk"
        verdict = "Likely synthetic"
    elif risk_score >= 61:
        risk_label = "Suspicious"
        verdict = "Suspicious"
    elif risk_score >= 31:
        risk_label = "Uncertain"
        verdict = "Inconclusive"
    else:
        risk_label = "Low risk"
        verdict = "Likely genuine"

    top_regions = sorted(analyzed, key=lambda item: float(item.synthetic_score), reverse=True)[:5]
    top_regions.sort(key=lambda item: item.start_seconds)
    inference_values = [item.inference_ms for item in analyzed if item.inference_ms is not None]
    gate_names = sorted({item.quality.speech_gate for item in items})

    detector_notice = str(getattr(detector, "notice", "Risk is an aggregated security signal and is not proof of authenticity."))
    return {
        "analysis_mode": detector.mode,
        "model": detector.name,
        "calibrated_probability": bool(getattr(detector, "calibrated_probability", False)),
        "speech_gate": gate_names[0] if len(gate_names) == 1 else gate_names,
        "duration_seconds": round(duration_seconds, 3),
        "usable_speech_seconds": round(usable_speech_seconds, 3),
        "windows_seen": len(items),
        "segments_analyzed": len(analyzed),
        "segments_skipped": len(skipped),
        "suspicious_segments": len(suspicious),
        "suspicious_ratio": round(suspicious_ratio, 6),
        "median_synthetic_score": round(median_score, 6),
        "mean_synthetic_score": round(mean_score, 6),
        "median_evidence_signal": round(median_evidence, 6),
        "mean_evidence_signal": round(mean_evidence, 6),
        "consistency": round(consistency, 6),
        "risk_score": risk_score,
        "risk_label": risk_label,
        "verdict": verdict,
        "enough_evidence": enough_evidence,
        "threshold": round(float(detector.threshold), 6),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "mean_inference_ms": round(statistics.fmean(inference_values), 3) if inference_values else None,
        "regions": [region.as_dict() for region in top_regions],
        "notice": detector_notice + " Audio quality gating and 16 kHz normalization are active.",
    }
