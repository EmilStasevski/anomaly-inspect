"""ONNX parity: the exported graph must give the same result as anomalib's PatchCore."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomaly_inspect.data import Preprocess, list_samples
from anomaly_inspect.export import (
    MIN_OPSET,
    MODEL_FILE,
    PatchcoreOnnxModule,
    create_session,
    euclidean_distance,
    export_patchcore,
    graph_opset,
    run_session,
)
from anomaly_inspect.models import PatchcoreDetector, build_detector
from anomaly_inspect.pipeline import load_config
from anomaly_inspect.tracking import seed_everything

# Scores are in the tens. These bounds are about 1e-5 of that scale.
SCORE_TOLERANCE = 1e-3
MAP_TOLERANCE = 1e-2
SIZE = 64
REAL_CATEGORY = "metal_nut"
REAL_DIR = Path("models/onnx") / REAL_CATEGORY
FITTED_FILE = Path("models/patchcore") / REAL_CATEGORY / "model.pt"
DATA_ROOT = Path("data/mvtec_ad")


@pytest.fixture(scope="module")
def tiny_detector() -> PatchcoreDetector:
    """An untrained small backbone, because parity does not depend on the weights."""
    seed_everything(0)
    detector = PatchcoreDetector(
        backbone="resnet18", coreset_ratio=0.5, num_neighbors=3, pre_trained=False
    ).eval()
    detector.fit([0.5 + 0.1 * torch.randn(4, 3, SIZE, SIZE)])
    return detector


def test_distance_matches_cdist() -> None:
    x, y = torch.randn(5, 16), torch.randn(7, 16)
    assert torch.allclose(euclidean_distance(x, y), torch.cdist(x, y), atol=1e-4)


def test_wrapper_matches_anomalib_without_onnx(tiny_detector: PatchcoreDetector) -> None:
    images = torch.randn(1, 3, SIZE, SIZE)
    expected_score, expected_map = tiny_detector.predict(images)
    with torch.no_grad():
        score, anomaly_map = PatchcoreOnnxModule(tiny_detector).eval()(images)
    assert torch.allclose(score, expected_score, atol=1e-4)
    assert torch.allclose(anomaly_map, expected_map, atol=1e-4)


def test_onnx_matches_pytorch_on_synthetic_model(
    tiny_detector: PatchcoreDetector, tmp_path: Path
) -> None:
    path = tmp_path / MODEL_FILE
    export_patchcore(tiny_detector, path, crop=SIZE)
    assert graph_opset(path) >= MIN_OPSET

    session = create_session(path)
    generator = torch.Generator().manual_seed(1)
    for _ in range(3):
        image = torch.randn(1, 3, SIZE, SIZE, generator=generator)
        expected_score, expected_map = tiny_detector.predict(image)
        score, anomaly_map = run_session(session, image[0].numpy())
        assert float(score) == pytest.approx(float(expected_score[0]), abs=SCORE_TOLERANCE)
        assert np.abs(anomaly_map - expected_map[0].numpy()).max() <= MAP_TOLERANCE


def test_export_rejects_an_old_opset(tiny_detector: PatchcoreDetector, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="opset"):
        export_patchcore(tiny_detector, tmp_path / MODEL_FILE, crop=SIZE, opset=MIN_OPSET - 1)


def test_cannot_export_an_empty_memory_bank() -> None:
    detector = PatchcoreDetector(backbone="resnet18", pre_trained=False).eval()
    with pytest.raises(ValueError, match="empty memory bank"):
        PatchcoreOnnxModule(detector)


@pytest.mark.model
@pytest.mark.data
def test_onnx_matches_pytorch_on_real_model() -> None:
    if not (REAL_DIR / MODEL_FILE).is_file() or not FITTED_FILE.is_file():
        pytest.skip("run `make train` and `make export` first")
    if not DATA_ROOT.is_dir():
        pytest.skip("MVTec AD is not available")

    config = load_config(Path("configs/patchcore.yaml"))
    detector = build_detector("patchcore", config["model"]).eval()
    detector.load(FITTED_FILE)
    preprocess = Preprocess(**config["preprocess"])
    session = create_session(REAL_DIR / MODEL_FILE)

    samples = list_samples(DATA_ROOT, REAL_CATEGORY, "test")
    picked = np.linspace(0, len(samples) - 1, 16).astype(int)
    labels, scores = [], []
    for index in picked:
        with Image.open(samples[index].image_path) as image:
            array = preprocess.image(image)
        expected_score, expected_map = detector.predict(torch.from_numpy(array)[None])
        score, anomaly_map = run_session(session, array)
        assert float(score) == pytest.approx(float(expected_score[0]), abs=SCORE_TOLERANCE)
        assert np.abs(anomaly_map - expected_map[0].numpy()).max() <= MAP_TOLERANCE
        labels.append(samples[index].label)
        scores.append(float(score))
    # The subset has both classes, otherwise this test would not exercise real decisions.
    assert set(labels) == {0, 1}

    metadata = json.loads((REAL_DIR / "model.json").read_text())
    assert metadata["category"] == REAL_CATEGORY
    assert metadata["preprocess"]["crop"] == 224
