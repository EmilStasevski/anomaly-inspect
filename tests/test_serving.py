"""Cross-language checks of the Rust service against the Python reference implementation."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from anomaly_inspect.data import CATEGORIES, Preprocess, list_samples
from anomaly_inspect.export import create_session, run_session

SERVING = Path("serving")
BINARIES = SERVING / "target" / "release"
DATA_ROOT = Path("data/mvtec_ad")
MODEL_ROOT = Path("models/onnx")
CATEGORY = "metal_nut"
SCORE_TOLERANCE = 1e-3


@pytest.fixture(scope="session")
def binaries() -> Path:
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=SERVING, check=True)
    return BINARIES


def rust_tensor(binaries: Path, image_path: Path, tmp_path: Path) -> np.ndarray:
    out = tmp_path / "tensor.bin"
    subprocess.run([binaries / "preprocess_dump", image_path, out], check=True)
    return np.fromfile(out, dtype="<f4").reshape(3, 224, 224)


def assert_bit_identical(rust: np.ndarray, python: np.ndarray) -> None:
    assert rust.dtype == python.dtype
    assert rust.tobytes() == python.tobytes(), f"max difference {np.abs(rust - python).max():.3e}"


@pytest.mark.parametrize(
    ("mode", "size"),
    [
        ("RGB", (1024, 1024)),  # downscale, as for most categories
        ("RGB", (700, 700)),  # the metal_nut size
        ("RGB", (100, 100)),  # upscale
        ("RGB", (256, 256)),  # no resize at all
        ("RGB", (640, 333)),  # different factors per axis
        ("L", (1024, 1024)),  # grayscale, as for screw, grid and zipper
        ("RGBA", (500, 500)),  # the alpha channel is dropped
        ("P", (300, 300)),  # palette
    ],
)
def test_preprocessing_is_bit_identical_on_synthetic_images(
    binaries: Path, tmp_path: Path, mode: str, size: tuple[int, int]
) -> None:
    noise = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 4), dtype=np.uint8)
    image = Image.fromarray(noise, "RGBA").convert(mode)
    path = tmp_path / "input.png"
    image.save(path)
    with Image.open(path) as reopened:
        python = Preprocess().image(reopened)
    assert_bit_identical(rust_tensor(binaries, path, tmp_path), python)


@pytest.mark.data
@pytest.mark.parametrize("category", CATEGORIES)
def test_preprocessing_is_bit_identical_on_real_images(
    binaries: Path, tmp_path: Path, category: str
) -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip("MVTec AD is not available")
    samples = list_samples(DATA_ROOT, category, "test")
    for index in (0, len(samples) // 2, len(samples) - 1):
        path = samples[index].image_path
        with Image.open(path) as image:
            python = Preprocess().image(image)
        assert_bit_identical(rust_tensor(binaries, path, tmp_path), python)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def service(binaries: Path) -> Iterator[str]:
    if not (MODEL_ROOT / CATEGORY / "model.onnx").is_file() or not DATA_ROOT.is_dir():
        pytest.skip("run `make export` first and provide MVTec AD")
    port = _free_port()
    process = subprocess.Popen(
        [binaries / "anomaly-serving"],
        env={
            **os.environ,
            "MODEL_ROOT": str(MODEL_ROOT),
            "MODEL_CATEGORIES": CATEGORY,
            "INTRA_THREADS": "4",
            "BIND_ADDRESS": f"127.0.0.1:{port}",
        },
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            pytest.fail("the service did not become healthy")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def _post(url: str, path: Path, **params: object) -> requests.Response:
    with path.open("rb") as handle:
        return requests.post(f"{url}/predict", files={"image": handle}, params=params, timeout=60)


@pytest.mark.model
@pytest.mark.data
def test_health_lists_the_model(service: str) -> None:
    response = requests.get(f"{service}/health", timeout=5)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "categories": [CATEGORY]}


@pytest.mark.model
@pytest.mark.data
def test_predict_matches_python_onnx_runtime_on_real_images(service: str) -> None:
    session = create_session(MODEL_ROOT / CATEGORY / "model.onnx")
    metadata = json.loads((MODEL_ROOT / CATEGORY / "model.json").read_text())
    samples = list_samples(DATA_ROOT, CATEGORY, "test")
    decisions = set()
    for index in np.linspace(0, len(samples) - 1, 8).astype(int):
        path = samples[index].image_path
        with Image.open(path) as image:
            expected, _ = run_session(session, Preprocess().image(image))
        body = _post(service, path).json()
        assert body["category"] == CATEGORY
        assert body["score"] == pytest.approx(float(expected), abs=SCORE_TOLERANCE)
        assert body["threshold"] == metadata["threshold"]
        assert body["is_anomalous"] == (body["score"] >= metadata["threshold"])
        assert "heatmap_png_base64" not in body
        decisions.add(body["is_anomalous"])
    assert decisions == {True, False}


@pytest.mark.model
@pytest.mark.data
def test_predict_returns_a_heatmap_on_request(service: str) -> None:
    path = list_samples(DATA_ROOT, CATEGORY, "test")[-1].image_path
    body = _post(service, path, heatmap="true").json()
    heatmap = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"])))
    assert heatmap.size == (224, 224)
    assert heatmap.mode == "RGB"


@pytest.mark.model
@pytest.mark.data
def test_unknown_category_is_404(service: str) -> None:
    path = list_samples(DATA_ROOT, CATEGORY, "test")[0].image_path
    response = _post(service, path, category="nope")
    assert response.status_code == 404
    assert "nope" in response.json()["error"]


@pytest.mark.model
@pytest.mark.data
def test_invalid_uploads_are_400(service: str) -> None:
    garbage = requests.post(f"{service}/predict", files={"image": b"not an image"}, timeout=10)
    assert garbage.status_code == 400
    assert "decode" in garbage.json()["error"]
    wrong_field = requests.post(f"{service}/predict", files={"file": b"x"}, timeout=10)
    assert wrong_field.status_code == 400
    assert "image" in wrong_field.json()["error"]


@pytest.mark.model
@pytest.mark.data
def test_metrics_are_in_prometheus_format_and_count_requests(service: str) -> None:
    path = list_samples(DATA_ROOT, CATEGORY, "test")[0].image_path
    assert _post(service, path).ok
    response = requests.get(f"{service}/metrics", timeout=5)
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "# TYPE predictions_total counter" in text
    assert f'predictions_total{{category="{CATEGORY}"' in text
    assert 'prediction_stage_seconds_count{stage="inference"}' in text
