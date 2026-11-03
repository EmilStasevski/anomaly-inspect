"""Anomaly heatmap figures for reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from .data import Preprocess, Sample

# One hue whose opacity follows the score, on a desaturated image, so it stays readable on
# coloured parts. The contour colour is reserved for the ground truth.
HEAT_COLOR = "#E5340B"
TRUTH_COLOR = "#16D5E8"


@dataclass(frozen=True)
class HeatScale:
    """Map scores to [0, 1] with limits from good calibration images only."""

    low: float
    high: float

    @classmethod
    def from_good_maps(cls, maps: np.ndarray) -> HeatScale:
        """Start at the median good pixel. Saturate at 1.5x the highest good pixel."""
        low = float(np.median(maps))
        high = max(float(maps.max()) * 1.5, low + 1e-6)
        return cls(low, high)

    def normalise(self, anomaly_map: np.ndarray) -> np.ndarray:
        return np.clip((anomaly_map - self.low) / (self.high - self.low), 0.0, 1.0)


def overlay(gray: np.ndarray, heat: np.ndarray, max_opacity: float = 0.85) -> np.ndarray:
    """Blend the heat colour over a grayscale image (H, W) in [0, 1]. Returns RGB in [0, 1]."""
    alpha = (heat * max_opacity)[..., None]
    return (1.0 - alpha) * gray[..., None] + alpha * np.asarray(to_rgb(HEAT_COLOR))


def select_examples(samples: list[Sample], n_good: int = 2, max_defect_types: int = 4) -> list[int]:
    """Indices of the first good images and the first image of each defect type.

    The rule does not look at scores, so the figures cannot be cherry-picked.
    """
    good = [i for i, sample in enumerate(samples) if not sample.label][:n_good]
    first_of_type: dict[str, int] = {}
    for index, sample in enumerate(samples):
        if sample.label:
            first_of_type.setdefault(sample.defect_type, index)
    return good + list(first_of_type.values())[:max_defect_types]


def save_heatmap_figure(
    samples: list[Sample],
    indices: list[int],
    maps: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    scale: HeatScale,
    preprocess: Preprocess,
    title: str,
    path: Path,
) -> None:
    """One column per image: the input on top, score overlay and true defect outline below."""
    columns = len(indices)
    figure, axes = plt.subplots(2, columns, figsize=(2.1 * columns, 4.9), squeeze=False)
    box = preprocess.crop_box()
    for column, index in enumerate(indices):
        sample = samples[index]
        with Image.open(sample.image_path) as image:
            resized = image.convert("RGB").resize(
                (preprocess.resize, preprocess.resize), Image.Resampling.BILINEAR
            )
        rgb = resized.crop(box)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

        axes[0, column].imshow(rgb)
        axes[0, column].set_title(sample.defect_type.replace("_", " "), fontsize=10)
        axes[1, column].imshow(overlay(gray, scale.normalise(maps[index])))
        if masks[index].any():
            axes[1, column].contour(masks[index], levels=[0.5], colors=TRUTH_COLOR, linewidths=1.2)
        verdict = "defect" if scores[index] >= threshold else "pass"
        axes[1, column].set_xlabel(f"{scores[index]:.2f}: {verdict}", fontsize=9)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)

    figure.suptitle(
        f"{title}\nRed: anomaly score. Cyan outline: true defect. "
        f"Below: image score, decision at threshold {threshold:.2f}.",
        fontsize=10,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
