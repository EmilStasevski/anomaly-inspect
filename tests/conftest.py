"""A tiny synthetic dataset in MVTec layout, so that the tests do not need the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, rng: np.random.Generator, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    """Category `widget`: 4 train good, 2 test good, 2 test `scratch` with masks."""
    rng = np.random.default_rng(0)
    root = tmp_path / "mvtec_ad"
    category = root / "widget"
    for index in range(4):
        _write_rgb(category / "train" / "good" / f"{index:03d}.png", rng)
    for index in range(2):
        _write_rgb(category / "test" / "good" / f"{index:03d}.png", rng)
        _write_rgb(category / "test" / "scratch" / f"{index:03d}.png", rng)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        mask_path = category / "ground_truth" / "scratch" / f"{index:03d}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
    return root
