"""Evaluation metrics, shared by every model so that the comparison stays fair."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Same 8-connectivity as the reference MVTec evaluation code.
_CONNECTIVITY = np.ones((3, 3), dtype=int)


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """AUROC over all pixels of all images. Both arrays have shape (N, H, W)."""
    return float(roc_auc_score(masks.ravel(), maps.ravel()))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over all thresholds, and that threshold.

    The threshold is picked with test labels, so this is an upper bound that a production line
    could not reach. See `calibrate_threshold` for the realistic alternative.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    # The final precision/recall point has no threshold.
    best = int(np.argmax(f1[:-1]))
    return float(f1[best]), float(thresholds[best])


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """F1 of the decision `score >= threshold`."""
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def calibrate_threshold(good_scores: np.ndarray, sigmas: float = 3.0) -> float:
    """Threshold from defect-free scores only: mean + k standard deviations."""
    return float(np.mean(good_scores) + sigmas * np.std(good_scores))


def pro_curve(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-region overlap against false positive rate, one point per distinct score.

    Every connected defect region counts equally however large it is, which is why this metric
    is more informative than pixel AUROC for small defects.
    """
    if masks.shape != maps.shape or masks.ndim != 3:
        raise ValueError(f"expected matching (N, H, W), got {masks.shape} and {maps.shape}")

    pro_steps = np.zeros(masks.size, dtype=np.float64)
    good_pixels = masks.ravel() == 0
    n_good = int(good_pixels.sum())
    if n_good == 0:
        raise ValueError("need at least one defect-free pixel")

    regions: list[tuple[int, np.ndarray]] = []
    height_width = masks.shape[1] * masks.shape[2]
    for index, mask in enumerate(masks):
        labelled, n_regions = ndimage.label(mask, structure=_CONNECTIVITY)
        for region in range(1, n_regions + 1):
            regions.append((index * height_width, np.flatnonzero(labelled == region)))
    if not regions:
        raise ValueError("need at least one defect region")
    for offset, pixels in regions:
        pro_steps[offset + pixels] = 1.0 / (pixels.size * len(regions))

    fpr_steps = good_pixels.astype(np.float64) / n_good
    flat_scores = maps.ravel()
    order = np.argsort(-flat_scores, kind="stable")
    sorted_scores = flat_scores[order]
    fpr = np.cumsum(fpr_steps[order])
    pro = np.cumsum(pro_steps[order])

    # Pixels with equal scores cannot be separated, so keep only the last of each run.
    keep = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    return np.concatenate(([0.0], fpr[keep])), np.concatenate(([0.0], pro[keep]))


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve up to `fpr_limit`, normalised to [0, 1]."""
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
    fpr, pro = pro_curve(masks, maps)
    end = np.searchsorted(fpr, fpr_limit, side="left")
    x = np.append(fpr[:end], fpr_limit)
    y = np.append(pro[:end], np.interp(fpr_limit, fpr, pro))
    return float(np.trapezoid(y, x) / fpr_limit)
