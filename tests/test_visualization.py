from __future__ import annotations

from pathlib import Path

import numpy as np

from anomaly_inspect.data import Preprocess, list_samples
from anomaly_inspect.visualization import HeatScale, overlay, save_heatmap_figure, select_examples


def test_heat_scale_clips_to_unit_interval() -> None:
    scale = HeatScale(low=1.0, high=3.0)
    assert scale.normalise(np.array([0.0, 2.0, 9.0])).tolist() == [0.0, 0.5, 1.0]


def test_overlay_with_no_heat_keeps_the_image() -> None:
    gray = np.full((2, 2), 0.3)
    assert np.allclose(overlay(gray, np.zeros((2, 2))), 0.3)


def test_examples_are_chosen_without_scores(mvtec_root: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "test")
    assert select_examples(samples) == [0, 1, 2]


def test_figure_is_written(mvtec_root: Path, tmp_path: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "test")
    n = len(samples)
    maps = np.random.default_rng(0).random((n, 224, 224))
    masks = np.zeros((n, 224, 224), dtype=np.uint8)
    masks[2, 10:40, 10:40] = 1
    path = tmp_path / "figure.png"
    save_heatmap_figure(
        samples,
        select_examples(samples),
        maps,
        masks,
        np.linspace(0, 1, n),
        0.5,
        HeatScale(0.0, 1.0),
        Preprocess(),
        "test",
        path,
    )
    assert path.stat().st_size > 1000
