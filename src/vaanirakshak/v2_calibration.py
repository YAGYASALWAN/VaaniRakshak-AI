"""Probability calibration helpers for VaaniRakshak V2.

Temperature scaling is fitted on development data only. It is a monotonic
transformation of the detector logit, so the detector's existing development-
selected operating point can be transformed to preserve exactly the same binary
decisions while making score interpretation more probabilistic.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar


EPSILON = 1e-6
DEFAULT_ECE_BINS = 15


def _arrays(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must be equally sized nonempty 1-D sequences")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("labels must contain only 0/1")
    if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both bonafide and spoof examples")
    return y, p


def _sample_weights(sample_weights: Iterable[float] | None, count: int) -> np.ndarray:
    if sample_weights is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray(list(sample_weights), dtype=np.float64)
    if weights.ndim != 1 or len(weights) != count:
        raise ValueError("sample_weights must match the number of calibration examples")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("sample_weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sample_weights must contain positive total weight")
    return weights / total


def probability_to_logit(probability):
    p = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def logit_to_probability(logit):
    z = np.asarray(logit, dtype=np.float64)
    result = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def apply_temperature(probabilities: Iterable[float], temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    p = np.asarray(list(probabilities), dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a nonempty 1-D sequence")
    if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return logit_to_probability(probability_to_logit(p) / float(temperature))


def transform_threshold(threshold: float, temperature: float) -> float:
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    calibrated = float(apply_temperature([threshold], temperature)[0])
    if not 0.0 < calibrated < 1.0:
        raise ValueError("calibrated threshold fell outside (0, 1)")
    return calibrated


def binary_nll(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    sample_weights: Iterable[float] | None = None,
) -> float:
    y, p = _arrays(labels, probabilities)
    weights = _sample_weights(sample_weights, len(y))
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.sum(weights * losses))


def brier_score(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y, p = _arrays(labels, probabilities)
    return float(np.mean(np.square(p - y)))


def expected_calibration_error(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    bins: int = DEFAULT_ECE_BINS,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y, p = _arrays(labels, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(np.mean(p[mask]))
        empirical = float(np.mean(y[mask]))
        ece += (count / total) * abs(confidence - empirical)
    return float(ece)


def fit_temperature(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    sample_weights: Iterable[float] | None = None,
) -> float:
    y, p = _arrays(labels, probabilities)
    weights = _sample_weights(sample_weights, len(y))
    logits = probability_to_logit(p)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        calibrated = logit_to_probability(logits / temperature)
        clipped = np.clip(calibrated, EPSILON, 1.0 - EPSILON)
        losses = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        return float(np.sum(weights * losses))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-6})
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Temperature calibration optimization failed")
    temperature = math.exp(float(result.x))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("Temperature calibration produced an invalid value")
    return float(temperature)


def calibration_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict:
    y, p = _arrays(labels, probabilities)
    return {
        "n": int(len(y)),
        "nll": binary_nll(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
    }
