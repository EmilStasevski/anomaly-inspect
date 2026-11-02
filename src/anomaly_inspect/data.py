"""MVTec AD discovery, validation and loading.

The preprocessing here is the reference that the Rust service has to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# (train good images, test images) as published with MVTec AD. 3629 train + 1725 test.
EXPECTED_COUNTS: dict[str, tuple[int, int]] = {
    "bottle": (209, 83),
    "cable": (224, 150),
    "capsule": (219, 132),
    "carpet": (280, 117),
    "grid": (264, 78),
    "hazelnut": (391, 110),
    "leather": (245, 124),
    "metal_nut": (220, 115),
    "pill": (267, 167),
    "screw": (320, 160),
    "tile": (230, 117),
    "toothbrush": (60, 42),
    "transistor": (213, 100),
    "wood": (247, 79),
    "zipper": (240, 151),
}
CATEGORIES: tuple[str, ...] = tuple(EXPECTED_COUNTS)
FOCUS_CATEGORIES: tuple[str, ...] = (
    "metal_nut",
    "screw",
    "pill",
    "transistor",
    "bottle",
    "carpet",
)
GOOD = "good"
IMAGE_SUFFIX = ".png"


@dataclass(frozen=True)
class Sample:
    """One image, with its defect mask if it has one."""

    category: str
    split: str
    defect_type: str
    image_path: Path
    mask_path: Path | None

    @property
    def label(self) -> int:
        return int(self.defect_type != GOOD)


@dataclass(frozen=True)
class Preprocess:
    """Resize, centre crop, ImageNet normalisation."""

    resize: int = 256
    crop: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def _crop_box(self) -> tuple[int, int, int, int]:
        start = (self.resize - self.crop) // 2
        return (start, start, start + self.crop, start + self.crop)

    def image(self, image: Image.Image) -> np.ndarray:
        """Return a float32 CHW array. Grayscale categories are expanded to three channels."""
        rgb = image.convert("RGB").resize((self.resize, self.resize), Image.Resampling.BILINEAR)
        array = np.asarray(rgb.crop(self._crop_box()), dtype=np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((array - mean) / std).transpose(2, 0, 1)

    def mask(self, mask: Image.Image) -> np.ndarray:
        """Return a uint8 HW array of 0/1, using nearest neighbour so that labels stay binary."""
        resized = mask.convert("L").resize((self.resize, self.resize), Image.Resampling.NEAREST)
        return (np.asarray(resized.crop(self._crop_box())) > 0).astype(np.uint8)


def list_samples(root: Path, category: str, split: str) -> list[Sample]:
    """List a split in sorted order. Raises FileNotFoundError if the split directory is missing."""
    split_dir = root / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    samples: list[Sample] = []
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        for image_path in sorted(defect_dir.glob(f"*{IMAGE_SUFFIX}")):
            mask_path = None
            if defect_type != GOOD:
                mask_path = (
                    root / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                )
            samples.append(Sample(category, split, defect_type, image_path, mask_path))
    return samples


@dataclass
class CategoryReport:
    """Result of validating one category directory."""

    category: str
    train_good: int = 0
    test_good: int = 0
    test_defect: int = 0
    defect_types: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_category(
    root: Path, category: str, expected: tuple[int, int] | None = None
) -> CategoryReport:
    """Check layout, per-image masks and, if given, the published (train, test) image counts."""
    report = CategoryReport(category)
    for name in ("train", "test", "ground_truth"):
        if not (root / category / name).is_dir():
            report.errors.append(f"missing directory {category}/{name}")
    if report.errors:
        return report

    train = list_samples(root, category, "train")
    report.train_good = len(train)
    non_good = sorted({s.defect_type for s in train if s.label})
    if non_good:
        report.errors.append(f"train split must only contain good images, found {non_good}")

    test = list_samples(root, category, "test")
    report.test_good = sum(1 for s in test if not s.label)
    report.test_defect = sum(1 for s in test if s.label)
    report.defect_types = sorted({s.defect_type for s in test if s.label})
    missing = [s.image_path.name for s in test if s.mask_path and not s.mask_path.is_file()]
    if missing:
        report.errors.append(f"{len(missing)} defect images have no mask, e.g. {missing[0]}")

    if expected is not None:
        found = (report.train_good, report.test_good + report.test_defect)
        if found != expected:
            report.errors.append(f"(train, test) counts {found} differ from published {expected}")
    return report


class Item(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    label: int
    path: str


class MVTecDataset(Dataset[Item]):
    """Preprocessed images and masks. Good images get an all-zero mask."""

    def __init__(self, samples: list[Sample], preprocess: Preprocess | None = None) -> None:
        self.samples = samples
        self.preprocess = preprocess or Preprocess()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Item:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            array = self.preprocess.image(image)
        if sample.mask_path is None:
            mask = np.zeros((self.preprocess.crop, self.preprocess.crop), dtype=np.uint8)
        else:
            with Image.open(sample.mask_path) as mask_image:
                mask = self.preprocess.mask(mask_image)
        return Item(
            image=torch.from_numpy(array),
            mask=torch.from_numpy(mask),
            label=sample.label,
            path=str(sample.image_path),
        )


def split_calibration(
    samples: list[Sample], fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split train images into (fit, calibration). Thresholds come from calibration only."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    order = np.random.default_rng(seed).permutation(len(samples))
    n_calibration = max(1, round(len(samples) * fraction))
    calibration = set(order[:n_calibration].tolist())
    fit = [s for i, s in enumerate(samples) if i not in calibration]
    held_out = [s for i, s in enumerate(samples) if i in calibration]
    return fit, held_out
