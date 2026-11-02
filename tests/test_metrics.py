from __future__ import annotations

import numpy as np
import pytest

from anomaly_inspect.metrics import (
    aupro,
    calibrate_threshold,
    f1_at_threshold,
    f1_max,
    image_auroc,
    pixel_auroc,
    pro_curve,
)


def test_image_auroc_known_values() -> None:
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert image_auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # Of the four (good, defect) pairs, three are ordered correctly.
    assert image_auroc(labels, np.array([0.1, 0.4, 0.3, 0.9])) == 0.75


def test_pixel_auroc_perfect_separation() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1
    maps = masks.astype(float)
    assert pixel_auroc(masks, maps) == 1.0


def test_f1_max_finds_separating_threshold() -> None:
    f1, threshold = f1_max(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0
    assert 0.2 < threshold <= 0.8


def test_f1_at_threshold_hand_computed() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.7, 0.8, 0.2])
    # At 0.5: TP=1, FP=1, FN=1, so F1 = 2 / (2 + 1 + 1).
    assert f1_at_threshold(labels, scores, 0.5) == pytest.approx(0.5)
    assert f1_at_threshold(labels, scores, 2.0) == 0.0


def test_calibrate_threshold_uses_mean_and_std() -> None:
    assert calibrate_threshold(np.array([1.0, 3.0]), sigmas=2.0) == pytest.approx(4.0)


def _perfect_case() -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[1, 5:7, 5:7] = 1
    return masks, masks.astype(float)


def test_aupro_perfect_is_one() -> None:
    assert aupro(*_perfect_case()) == pytest.approx(1.0)


def test_aupro_inverted_is_zero() -> None:
    masks, maps = _perfect_case()
    assert aupro(masks, -maps) == pytest.approx(0.0)


def test_aupro_constant_map_is_chance() -> None:
    # All pixels tie, so the curve is the straight line from (0, 0) to (1, 1): PRO = FPR.
    masks, _ = _perfect_case()
    maps = np.zeros_like(masks, dtype=float)
    assert aupro(masks, maps, fpr_limit=0.3) == pytest.approx(0.15)
    assert aupro(masks, maps, fpr_limit=1.0) == pytest.approx(0.5)


def test_pro_weights_regions_equally_regardless_of_size() -> None:
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 0:5, 0:5] = 1  # large region: 25 pixels
    masks[0, 9, 9] = 1  # small region: 1 pixel
    maps = np.zeros_like(masks, dtype=float)
    maps[0, 9, 9] = 1.0  # only the small region is found
    fpr, pro = pro_curve(masks, maps)
    assert pro[1] == pytest.approx(0.5)  # half of the regions, although only 1/26 of the pixels
    assert fpr[1] == 0.0


def test_diagonally_touching_pixels_form_one_region() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 0, 0] = masks[0, 1, 1] = 1
    maps = np.zeros_like(masks, dtype=float)
    maps[0, 0, 0] = 1.0
    _, pro = pro_curve(masks, maps)
    assert pro[1] == pytest.approx(0.5)


def test_pro_curve_rejects_input_without_defects() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="defect region"):
        pro_curve(masks, np.zeros_like(masks, dtype=float))
