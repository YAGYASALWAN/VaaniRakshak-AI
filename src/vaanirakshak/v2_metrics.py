"""Binary anti-spoof metrics used by VaaniRakshak V2.

Positive label is spoof (1); negative label is bonafide (0). Threshold selection is
performed on development data only. These helpers deliberately avoid an accuracy-
only view and expose FPR/FNR/EER alongside ranking metrics.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _arrays(labels: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be equally sized nonempty 1-D sequences")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary values 0/1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if len(np.unique(y)) != 2:
        raise ValueError("metrics require both bonafide and spoof examples")
    return y, s


def operating_points(labels: Iterable[int], scores: Iterable[float]) -> list[dict]:
    y, s = _arrays(labels, scores)
    order = np.argsort(-s, kind="stable")
    y = y[order]
    s = s[order]
    positives = int(y.sum())
    negatives = int(len(y) - positives)

    # A finite threshold just above the largest score represents the legitimate
    # "reject everything as spoof" operating point with FPR=TPR=0. Keeping it
    # finite lets max-FPR threshold selection remain well-defined on small dev sets.
    reject_all_threshold = float(np.nextafter(s[0], np.inf))
    points = [
        {
            "threshold": reject_all_threshold,
            "tp": 0,
            "fp": 0,
            "tn": negatives,
            "fn": positives,
            "tpr": 0.0,
            "fpr": 0.0,
            "fnr": 1.0,
            "precision": 1.0,
            "recall": 0.0,
        }
    ]
    tp = fp = 0
    i = 0
    while i < len(y):
        threshold = float(s[i])
        j = i
        while j < len(y) and s[j] == s[i]:
            if y[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        fn = positives - tp
        tn = negatives - fp
        recall = tp / positives
        fpr = fp / negatives
        precision = tp / (tp + fp) if tp + fp else 1.0
        points.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": recall,
                "fpr": fpr,
                "fnr": fn / positives,
                "precision": precision,
                "recall": recall,
            }
        )
        i = j
    return points


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    points = operating_points(labels, scores)
    fpr = np.asarray([p["fpr"] for p in points], dtype=np.float64)
    tpr = np.asarray([p["tpr"] for p in points], dtype=np.float64)
    widths = np.diff(fpr)
    heights = (tpr[:-1] + tpr[1:]) * 0.5
    return float(np.sum(widths * heights))


def average_precision(labels: Iterable[int], scores: Iterable[float]) -> float:
    points = operating_points(labels, scores)
    ap = 0.0
    previous_recall = 0.0
    for point in points[1:]:
        recall = float(point["recall"])
        ap += (recall - previous_recall) * float(point["precision"])
        previous_recall = recall
    return float(ap)


def eer(labels: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    points = operating_points(labels, scores)
    best = min(points, key=lambda p: abs(float(p["fpr"]) - float(p["fnr"])))
    value = (float(best["fpr"]) + float(best["fnr"])) / 2.0
    return value, float(best["threshold"])


def threshold_for_max_fpr(labels: Iterable[int], scores: Iterable[float], max_fpr: float = 0.05) -> float:
    if not 0.0 <= max_fpr < 1.0:
        raise ValueError("max_fpr must be in [0, 1)")
    points = [p for p in operating_points(labels, scores) if p["fpr"] <= max_fpr]
    if not points:
        raise ValueError("No threshold satisfies the requested FPR")
    # Security-oriented operating point: within the allowed bonafide false-positive
    # budget, maximize spoof recall; on a tie choose the higher threshold.
    best = max(points, key=lambda p: (float(p["tpr"]), float(p["threshold"])))
    return float(best["threshold"])


def metrics_at_threshold(labels: Iterable[int], scores: Iterable[float], threshold: float) -> dict:
    y, s = _arrays(labels, scores)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / len(y)
    return {
        "n": len(y),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc_auc(y, s),
        "pr_auc": average_precision(y, s),
        "eer": eer(y, s)[0],
    }
