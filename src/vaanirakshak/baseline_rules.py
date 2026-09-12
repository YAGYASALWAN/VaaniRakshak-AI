"""Offline rules for the experimental baseline; distinct from the strict M2 manifest."""
from collections import Counter, defaultdict
import hashlib
import math
import random

LANGUAGES = ("Hindi", "Punjabi")
SPLITS = ("train", "dev", "test")
WARNING = ("Experimental cross-corpus baseline: source corpus and class are confounded. "
           "Scores do not establish real-world deepfake detection or unseen-generator performance. "
           "Speaker separation uses available metadata; cross-corpus identities are unverified.")


def identity(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = str(value).strip()
    return value if value and value.lower() not in {"nan", "none", "null"} else None


def speaker_keys(row, source):
    """Conservative global IDs within each corpus; source/target share a namespace."""
    fields = ("speaker_id",) if source == "indicvoices" else ("Source Speaker_ID", "Target Speaker ID")
    speakers = [identity(row.get(field)) for field in fields]
    if source == "indicvoices" and not speakers[0]:
        raise ValueError("Missing genuine speaker")
    if source == "indicsynth":
        if not speakers[1]:
            raise ValueError("Missing target speaker")
        if str(row.get("Generative Model", "")).lower().startswith("freevc") and not speakers[0]:
            raise ValueError("Missing voice-conversion source speaker")
        if not identity(row.get("Generative Model")):
            raise ValueError("Missing generator")
    return sorted({source + ":speaker:" + s for s in speakers if s})


def assigned_split(keys, seed=42):
    """Discard pairs that cross a preassigned speaker partition, never move speakers."""
    buckets = set()
    for key in keys:
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
        buckets.add("train" if bucket < 60 else "dev" if bucket < 80 else "test")
    return next(iter(buckets)) if len(buckets) == 1 else None


def references(row, source):
    if source != "indicsynth":
        return []
    return sorted({source + ":reference:" + str(value).replace("\\", "/").split("/")[-1]
                   for field in ("Source Reference Audio", "Target Reference Audio")
                   if (value := identity(row.get(field)))})


def remove_reference_conflicts(rows):
    """Drop all recordings whose shared references cross assigned splits."""
    seen = defaultdict(set)
    for row in rows:
        for key in row["references"]:
            seen[key].add(row["split"])
    bad = {key for key, splits in seen.items() if len(splits) > 1}
    return [row for row in rows if not bad.intersection(row["references"])]


def balanced_sample(rows, seed=42):
    """At most 2,000 clips; equal class/language counts within each split."""
    bins = defaultdict(list)
    for row in rows:
        bins[row["split"], row["language"], row["label"]].append(row)
    result = []
    rng = random.Random(seed)
    for split, maximum, minimum in (("train", 300, 30), ("dev", 100, 10), ("test", 100, 10)):
        groups = [bins[split, language, label] for language in LANGUAGES for label in (0, 1)]
        count = min(maximum, *(len(group) for group in groups))
        if count < minimum:
            raise ValueError(f"Insufficient independent {split} data: minimum cell count {count}; "
                             f"need {minimum} per language/class. See preparation_report.json.")
        for group in groups:
            rng.shuffle(group)
            result.extend(group[:count])
    validate_rows(result)
    return result


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty baseline manifest")
    owners = {}
    ids = set()
    for row in rows:
        if row["split"] not in SPLITS or row["language"] not in LANGUAGES or row["label"] not in (0, 1):
            raise ValueError("Invalid split, language or label")
        if row["id"] in ids or not row["speakers"]:
            raise ValueError("Duplicate recording ID or missing speaker")
        ids.add(row["id"])
        for key in row["speakers"] + row["references"] + ["audio:" + row["audio_sha256"],
                                                          "window:" + row["window_sha256"]]:
            if key in owners and owners[key] != row["split"]:
                raise ValueError("Speaker, reference or duplicate audio crosses splits")
            owners[key] = row["split"]
    for split in SPLITS:
        for lang in LANGUAGES:
            for label in (0, 1):
                group = [r for r in rows if (r["split"], r["language"], r["label"]) == (split, lang, label)]
                if len(group) < (30 if split == "train" else 10):
                    raise ValueError(f"Too few recordings in {split}/{lang}/{label}")
                if len({s for r in group for s in r["speakers"]}) < 2:
                    raise ValueError(f"Too few known speakers in {split}/{lang}/{label}")
    return dict(Counter(f'{r["split"]}/{r["language"]}/{r["label"]}' for r in rows))


#: Detection-cost parameters. ``p_spoof`` is the assumed prior probability that a
#: call is spoofed; ``c_miss`` weights a clone scored as genuine (a completed fraud)
#: and ``c_false_alarm`` weights real speech scored as synthetic (a wasted warning).
DCF_PARAMETERS = {"p_spoof": 0.05, "c_miss": 1.0, "c_false_alarm": 1.0}


def detection_curve(labels, scores):
    """Operating points as ``(threshold, false_alarm_rate, miss_rate)``, ties grouped.

    Uses the repository-wide convention: label 1 is synthetic, and a recording is
    called synthetic when ``score >= threshold``. A *miss* is therefore a spoof that
    scored below the threshold, and a *false alarm* is bona fide speech that scored
    above it. Thresholds increase, so the false-alarm rate falls monotonically while
    the miss rate rises. The first point accepts everything as synthetic; the last
    rejects everything and carries a threshold of ``None``, because no finite score
    threshold rejects a recording that scored the maximum.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("A detection curve requires both classes")
    ordered = sorted(zip(scores, labels))
    false_alarms, misses = negatives, 0
    points = [(ordered[0][0], 1.0, 0.0)]
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        misses += sum(y for _, y in group)
        false_alarms -= sum(1 for _, y in group if y == 0)
        # The next distinct score is the smallest threshold that rejects this group.
        boundary = ordered[end][0] if end < len(ordered) else None
        points.append((boundary, false_alarms / negatives, misses / positives))
        index = end
    return points


def equal_error_rate(labels, scores):
    """EER and the threshold at which the miss and false-alarm rates cross.

    Linearly interpolated between the two operating points that bracket the
    crossing, which is the standard treatment when no threshold makes the two rates
    exactly equal. The value does not depend on which class is called positive.
    """
    points = detection_curve(labels, scores)
    for low, high in zip(points, points[1:]):
        first, second = low[1] - low[2], high[1] - high[2]
        if first > 0 >= second:
            share = first / (first - second) if first != second else 0.0
            rate = low[1] + share * (high[1] - low[1])
            # Report the bracketing threshold that sits closest to the crossing.
            closest = min((low, high), key=lambda point: abs(point[1] - point[2]))
            return rate, closest[0]
        if first == 0:
            return low[1], low[0]
    best = min(points, key=lambda point: abs(point[1] - point[2]))
    return (best[1] + best[2]) / 2, best[0]


def min_detection_cost(labels, scores, parameters=None):
    """Minimum normalized detection cost over every threshold, and where it occurs.

    Normalized by the cost of the better trivial system (accept everything or reject
    everything), so 1.0 means the detector adds nothing over that trivial choice.
    """
    settings = dict(DCF_PARAMETERS, **(parameters or {}))
    spoof_prior = settings["p_spoof"]
    if not 0 < spoof_prior < 1 or settings["c_miss"] <= 0 or settings["c_false_alarm"] <= 0:
        raise ValueError("Detection-cost parameters must be positive with 0 < p_spoof < 1")
    miss_weight = settings["c_miss"] * spoof_prior
    false_alarm_weight = settings["c_false_alarm"] * (1 - spoof_prior)
    best_cost, best_threshold = math.inf, None
    for threshold, false_alarm_rate, miss_rate in detection_curve(labels, scores):
        cost = miss_weight * miss_rate + false_alarm_weight * false_alarm_rate
        if cost < best_cost:
            best_cost, best_threshold = cost, threshold
    return best_cost / min(miss_weight, false_alarm_weight), best_threshold, settings


def binary_metrics(labels, scores, threshold=0.5):
    """Rank-based ROC-AUC with ties; confusion rows=true, columns=predicted.

    Also reports threshold-independent EER and minimum normalized detection cost, so
    a cross-corpus result can be compared against published anti-spoofing numbers
    without inheriting this checkpoint's in-domain decision threshold.
    """
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        raise ValueError("Metrics require both classes and aligned predictions")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in scores):
        raise ValueError("Invalid scores")
    matrix = [[0, 0], [0, 0]]
    for y, p in zip(labels, scores):
        matrix[y][int(p >= threshold)] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in ordered[i:j])
        i = j
    eer, eer_threshold = equal_error_rate(labels, scores)
    min_dcf, dcf_threshold, dcf_settings = min_detection_cost(labels, scores)
    return {"n": len(labels), "threshold": threshold, "confusion_matrix": matrix,
            "accuracy": (tp + tn) / len(labels), "precision": tp / max(tp + fp, 1),
            "recall": tp / positives, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "roc_auc": (rank_sum - positives * (positives + 1) / 2) / (positives * negatives),
            "eer": eer, "eer_threshold": eer_threshold,
            "min_dcf": min_dcf, "min_dcf_threshold": dcf_threshold, "dcf_parameters": dcf_settings}
