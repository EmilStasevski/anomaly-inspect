from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomaly_inspect.data import (
    CATEGORIES,
    EXPECTED_COUNTS,
    MVTecDataset,
    Preprocess,
    list_samples,
    split_calibration,
    validate_category,
)


def test_expected_counts_match_published_totals() -> None:
    assert len(CATEGORIES) == 15
    assert sum(train for train, _ in EXPECTED_COUNTS.values()) == 3629
    assert sum(test for _, test in EXPECTED_COUNTS.values()) == 1725


def test_list_samples_labels_and_mask_paths(mvtec_root: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "test")
    assert [s.label for s in samples] == [0, 0, 1, 1]
    assert samples[0].mask_path is None
    assert samples[2].mask_path is not None
    assert samples[2].mask_path.name == "000_mask.png"


def test_valid_category_has_no_errors(mvtec_root: Path) -> None:
    report = validate_category(mvtec_root, "widget", expected=(4, 4))
    assert report.ok, report.errors
    assert (report.train_good, report.test_good, report.test_defect) == (4, 2, 2)
    assert report.defect_types == ["scratch"]


def test_validation_detects_wrong_counts(mvtec_root: Path) -> None:
    report = validate_category(mvtec_root, "widget", expected=(5, 4))
    assert not report.ok
    assert "differ from published" in report.errors[0]


def test_validation_detects_missing_mask(mvtec_root: Path) -> None:
    (mvtec_root / "widget/ground_truth/scratch/001_mask.png").unlink()
    report = validate_category(mvtec_root, "widget")
    assert any("no mask" in error for error in report.errors)


def test_validation_detects_defects_in_train(mvtec_root: Path) -> None:
    bad = mvtec_root / "widget/train/scratch"
    bad.mkdir()
    Image.new("RGB", (8, 8)).save(bad / "000.png")
    report = validate_category(mvtec_root, "widget")
    assert any("only contain good" in error for error in report.errors)


def test_validation_detects_missing_directory(tmp_path: Path) -> None:
    assert not validate_category(tmp_path, "nothing").ok


def test_dataset_shapes_and_mask_values(mvtec_root: Path) -> None:
    dataset = MVTecDataset(list_samples(mvtec_root, "widget", "test"))
    good, defect = dataset[0], dataset[2]
    assert good["image"].shape == (3, 224, 224)
    assert good["image"].dtype == torch.float32
    assert good["label"] == 0
    assert good["mask"].sum() == 0
    assert defect["label"] == 1
    assert set(defect["mask"].unique().tolist()) == {0, 1}


def test_grayscale_images_become_three_channels() -> None:
    array = Preprocess().image(Image.new("L", (300, 300), color=128))
    assert array.shape == (3, 224, 224)
    assert not np.allclose(array[0], array[1])  # per-channel normalisation was applied


def test_normalisation_values() -> None:
    preprocess = Preprocess()
    array = preprocess.image(Image.new("RGB", (300, 300), color=(255, 0, 255)))
    assert array[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-5)
    assert array[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-5)


def test_centre_crop_removes_the_border() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:16] = 255  # exactly the top band that a 256 -> 224 centre crop discards
    array = Preprocess().image(Image.fromarray(image))
    assert np.allclose(array, array[:, 0, 0][:, None, None])


def test_split_calibration_is_disjoint_and_seeded(mvtec_root: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "train")
    fit, calibration = split_calibration(samples, 0.25, seed=7)
    assert len(fit) == 3
    assert len(calibration) == 1
    assert not {s.image_path for s in fit} & {s.image_path for s in calibration}
    assert split_calibration(samples, 0.25, seed=7) == (fit, calibration)
