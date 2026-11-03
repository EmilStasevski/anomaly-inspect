"""Detector behaviour on CPU with untrained small backbones, so no weights are downloaded."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scipy.spatial.distance import mahalanobis

from anomaly_inspect.models import Detector, PadimBaseline, PatchcoreDetector, build_detector
from anomaly_inspect.tracking import seed_everything


def _padim() -> PadimBaseline:
    return PadimBaseline(feature_dims=8, pre_trained=False)


def _patchcore() -> PatchcoreDetector:
    return PatchcoreDetector(
        backbone="resnet18", coreset_ratio=0.5, num_neighbors=3, pre_trained=False
    )


def _flat_images(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.5 + 0.02 * torch.randn(n, 3, 64, 64, generator=generator)


def _with_defect(images: torch.Tensor) -> torch.Tensor:
    defective = images.clone()
    defective[:, :, 16:32, 16:32] = 3.0
    return defective


@pytest.mark.parametrize("factory", [_padim, _patchcore])
def test_shapes_and_planted_defect(factory) -> None:
    torch.manual_seed(0)
    detector: Detector = factory().eval()
    detector.fit([_flat_images(8)])

    good = _flat_images(2, seed=1)
    good_scores, good_maps = detector.predict(good)
    defect_scores, defect_maps = detector.predict(_with_defect(good))

    assert good_scores.shape == (2,)
    assert good_maps.shape == (2, 64, 64)
    assert defect_scores.min() > good_scores.max()
    # The defect region must be hotter than a far away corner.
    assert defect_maps[0, 16:32, 16:32].mean() > defect_maps[0, 52:, 52:].mean()


@pytest.mark.parametrize("factory", [_padim, _patchcore])
def test_fit_does_not_change_the_backbone(factory) -> None:
    torch.manual_seed(0)
    detector: Detector = factory().eval()
    before = {
        name: value.clone()
        for name, value in detector.state_dict().items()
        if "feature_extractor" in name or "extractor" in name
    }
    assert before
    detector.fit([_flat_images(8)])
    after = detector.state_dict()
    for name, value in before.items():
        assert torch.equal(value, after[name]), name


@pytest.mark.parametrize("factory", [_padim, _patchcore])
def test_save_and_load_reproduce_scores(factory, tmp_path: Path) -> None:
    torch.manual_seed(0)
    fitted: Detector = factory().eval()
    fitted.fit([_flat_images(8)])
    path = tmp_path / "model.pt"
    fitted.save(path)

    # The same random backbone, as a pretrained one would be, with an empty fitted state.
    torch.manual_seed(0)
    fresh: Detector = factory().eval()
    fresh.load(path)

    images = _with_defect(_flat_images(2, seed=1))
    assert torch.equal(fitted.predict(images)[0], fresh.predict(images)[0])


def test_patchcore_memory_bank_follows_coreset_ratio() -> None:
    torch.manual_seed(0)
    detector = _patchcore().eval()
    stats = detector.fit([_flat_images(4)])
    assert stats["memory_bank_size"] == int(stats["n_patches"] * 0.5)


def test_patchcore_fit_is_reproducible_with_a_seed() -> None:
    banks = []
    for _ in range(2):
        seed_everything(0)
        detector = _patchcore().eval()
        detector.fit([_flat_images(4)])
        banks.append(detector.fitted_state()["memory_bank"])
    assert torch.equal(banks[0], banks[1])


def test_padim_distance_matches_scipy() -> None:
    """Check the batched matrix expression against a reference implementation."""
    seed_everything(0)
    detector = _padim().eval()
    detector.fit([_flat_images(16)])

    images = _flat_images(1, seed=2)
    actual = detector.patch_scores(images).flatten()
    embedding = detector._embed(images)[:, detector.dim_index].flatten(2)[0].T.numpy()

    for location in (0, 5, 40):
        expected = mahalanobis(
            embedding[location],
            detector.mean[location].numpy(),
            detector.inverse_covariance[location].numpy(),
        )
        assert actual[location].item() == pytest.approx(expected, rel=1e-3)


def test_padim_rejects_too_many_dims() -> None:
    detector = PadimBaseline(feature_dims=10_000, pre_trained=False).eval()
    with pytest.raises(ValueError, match="feature_dims"):
        detector.fit([_flat_images(2)])


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit or load"):
        _padim().eval().predict(_flat_images(1))


def test_build_detector_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown model family"):
        build_detector("nope", {"layers": []})
