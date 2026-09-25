"""Image-level, localisation and latency metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from vlm_inspect.types import Box

LOCALIZATION_IOU = 0.1  # loose on purpose: defect extents are fuzzy (see docs/EVAL_PROTOCOL.md)
# A "pointing" box may cover at most this fraction of the image, so a whole-image box cannot
# trivially contain every defect.
POINTING_MAX_AREA_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    n_defective: int
    n_normal: int
    auroc: float
    average_precision: float
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    best_f1: float
    best_f1_threshold: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1_at(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float, float]:
    """Precision, recall, F1 and accuracy when predicting defective for score >= threshold."""
    predicted = scores >= threshold
    tp = float(np.sum(predicted & (labels == 1)))
    fp = float(np.sum(predicted & (labels == 0)))
    fn = float(np.sum(~predicted & (labels == 1)))
    tn = float(np.sum(~predicted & (labels == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, _safe_div(tp + tn, len(labels))


def image_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> ImageMetrics:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    both_classes = len(set(y.tolist())) == 2
    auroc = float(roc_auc_score(y, s)) if both_classes else math.nan
    ap = float(average_precision_score(y, s)) if both_classes else math.nan
    precision, recall, f1, accuracy = f1_at(y, s, threshold)

    # Best achievable F1 over all thresholds: what the method could do if the operator tuned it.
    best_f1, best_threshold = 0.0, threshold
    for t in np.unique(s):
        _, _, f, _ = f1_at(y, s, float(t))
        if f > best_f1:
            best_f1, best_threshold = f, float(t)
    return ImageMetrics(
        n_defective=int(y.sum()),
        n_normal=int(len(y) - y.sum()),
        auroc=auroc,
        average_precision=ap,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        best_f1=best_f1,
        best_f1_threshold=best_threshold,
    )


def localization_hit(
    predicted: Sequence[Box], truth: Sequence[Box], iou_threshold: float = LOCALIZATION_IOU
) -> bool:
    """True if any predicted box overlaps a ground-truth defect box with IoU >= threshold."""
    return any(p.iou(t) >= iou_threshold for p in predicted for t in truth)


def pointing_hit(
    predicted: Sequence[Box],
    truth: Sequence[Box],
    image_area: float,
    max_area_fraction: float = POINTING_MAX_AREA_FRACTION,
) -> bool:
    """True if a ground-truth defect's centre lies inside a predicted box of bounded size.

    Measures "points at the right place" for models that box the affected object rather than
    the defect itself (e.g. the whole candle around a chipped edge), which strict IoU misses.
    """
    for p in predicted:
        if p.area > max_area_fraction * image_area:
            continue
        for t in truth:
            cx, cy = (t.x1 + t.x2) / 2, (t.y1 + t.y2) / 2
            if p.x1 <= cx <= p.x2 and p.y1 <= cy <= p.y2:
                return True
    return False


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": math.nan,
            "p50_ms": math.nan,
            "p95_ms": math.nan,
            "max_ms": math.nan,
        }
    a = np.asarray(latencies_ms, dtype=float)
    return {
        "count": len(a),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
    }
