"""VaaniRakshak V2 streaming analysis primitives.

This module deliberately keeps the product/session logic separate from the ML model.
The initial detector is a deterministic mock so the complete product can be tested
before a V2 anti-spoofing checkpoint exists. Mock scores MUST NOT be presented as
real authenticity evidence.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, field
import hashlib
import math
import statistics
import time
from typing import Iterable, Protocol


WINDOW_SECONDS = 4.0
MIN_FINAL_WINDOW_SECONDS = 2.0
SUSPICIOUS_THRESHOLD = 0.65
HIGH_RISK_THRESHOLD = 0.80


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class Detector(Protocol):
    """Interface a future trained detector must implement."""

    name: str
    mode: str

    def score(self, samples: list[int], sample_rate: int) -> float:
        """Return a finite synthetic-speech score in [0, 1]."""


class MockDetector:
    """Deterministic placeholder used only to exercise the V2 product pipeline.

    The score is intentionally NOT an anti-spoofing prediction. It combines basic
    signal statistics with a small deterministic hash jitter so different windows
    produce different UI states without introducing non-reproducible randomness.
    """

    name = "v2-product-mock"
    mode = "mock"

    def score(self, samples: list[int], sample_rate: int) -> float:
        if not samples:
            return 0.5
        normalized = [s / 32768.0 for s in samples]
        rms = math.sqrt(sum(x * x for x in normalized) / len(normalized))
        peak = max(abs(x) for x in normalized)
        crossings = sum(
            1 for left, right in zip(normalized, normalized[1:])
            if (left < 0 <= right) or (left >= 0 > right)
        )
        zcr = crossings / max(1, len(normalized) - 1)
        crest = peak / max(rms, 1e-6)

        # Stable per-window jitter: useful for exercising aggregation and UI only.
        digest = hashlib.blake2s(array("h", samples[:4096]).tobytes(), digest_size=2).digest()
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
    synthetic_score: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "synthetic_score": round(self.synthetic_score, 6),
            "suspicious": self.synthetic_score >= SUSPICIOUS_THRESHOLD,
        }


@dataclass
class StreamingSession:
    detector: Detector
    sample_rate: int
    created_at: float = field(default_factory=time.monotonic)
    total_samples: int = 0
    processed_samples: int = 0
    buffer: array = field(default_factory=lambda: array("h"))
    windows: list[WindowResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 8_000 <= self.sample_rate <= 96_000:
            raise ValueError("sample_rate must be between 8000 and 96000 Hz")

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * WINDOW_SECONDS)

    def ingest_pcm16le(self, payload: bytes) -> list[WindowResult]:
        """Ingest signed little-endian 16-bit mono PCM and return new windows."""
        if not payload:
            return []
        if len(payload) % 2:
            raise ValueError("PCM payload must contain complete 16-bit samples")

        chunk = array("h")
        chunk.frombytes(payload)
        if chunk.itemsize != 2:
            raise RuntimeError("Unsupported Python short size")
        # GitHub/Windows/Linux targets are little-endian in practice; preserve the
        # explicit wire format for correctness on an unusual big-endian host.
        import sys
        if sys.byteorder != "little":
            chunk.byteswap()

        self.buffer.extend(chunk)
        self.total_samples += len(chunk)
        emitted: list[WindowResult] = []
        while len(self.buffer) >= self.window_samples:
            window = self.buffer[: self.window_samples]
            del self.buffer[: self.window_samples]
            emitted.append(self._analyze_window(list(window)))
        return emitted

    def _analyze_window(self, samples: list[int]) -> WindowResult:
        score = float(self.detector.score(samples, self.sample_rate))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detector returned an invalid score")
        start = self.processed_samples / self.sample_rate
        self.processed_samples += len(samples)
        result = WindowResult(
            index=len(self.windows),
            start_seconds=start,
            end_seconds=self.processed_samples / self.sample_rate,
            synthetic_score=score,
        )
        self.windows.append(result)
        return result

    def finalize(self) -> dict:
        """Analyze a useful partial tail and return a call-level product summary."""
        min_tail = int(self.sample_rate * MIN_FINAL_WINDOW_SECONDS)
        if len(self.buffer) >= min_tail:
            tail = list(self.buffer)
            self.buffer = array("h")
            self._analyze_window(tail)
        return aggregate_call(self.windows, self.total_samples / self.sample_rate, self.detector)

    def live_summary(self) -> dict:
        return aggregate_call(self.windows, self.total_samples / self.sample_rate, self.detector)


def aggregate_call(windows: Iterable[WindowResult], duration_seconds: float, detector: Detector) -> dict:
    items = list(windows)
    scores = [item.synthetic_score for item in items]
    suspicious = [item for item in items if item.synthetic_score >= SUSPICIOUS_THRESHOLD]

    if scores:
        median_score = statistics.median(scores)
        mean_score = statistics.fmean(scores)
        suspicious_ratio = len(suspicious) / len(scores)
        dispersion = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        consistency = _clamp(1.0 - dispersion / 0.30)
        # Keep risk distinct from a model probability. Multiple-window consistency
        # and suspicious-window prevalence influence the product-level signal.
        risk_fraction = _clamp(
            0.50 * median_score
            + 0.25 * mean_score
            + 0.20 * suspicious_ratio
            + 0.05 * consistency
        )
        risk_score = round(risk_fraction * 100)
    else:
        median_score = mean_score = suspicious_ratio = 0.0
        consistency = 0.0
        risk_score = 0

    if not items:
        risk_label = "Collecting evidence"
        verdict = "Insufficient audio"
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

    top_regions = sorted(items, key=lambda item: item.synthetic_score, reverse=True)[:5]
    top_regions.sort(key=lambda item: item.start_seconds)

    return {
        "analysis_mode": detector.mode,
        "model": detector.name,
        "duration_seconds": round(duration_seconds, 3),
        "segments_analyzed": len(items),
        "suspicious_segments": len(suspicious),
        "suspicious_ratio": round(suspicious_ratio, 6),
        "median_synthetic_score": round(median_score, 6),
        "mean_synthetic_score": round(mean_score, 6),
        "consistency": round(consistency, 6),
        "risk_score": risk_score,
        "risk_label": risk_label,
        "verdict": verdict,
        "threshold": SUSPICIOUS_THRESHOLD,
        "regions": [region.as_dict() for region in top_regions],
        "notice": (
            "V2 product-skeleton mode uses a deterministic mock detector. "
            "Risk and verdict values are UI/integration test data, not voice-authenticity findings."
            if detector.mode == "mock"
            else "Risk is an aggregated security signal and is not proof of authenticity."
        ),
    }
