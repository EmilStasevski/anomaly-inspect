"""Export a fitted PatchCore model, including its nearest-neighbour search, to ONNX."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import Preprocess
from .models import PatchcoreDetector

OPSET = 18
MIN_OPSET = 17
MODEL_FILE = "model.onnx"
METADATA_FILE = "model.json"
INPUT_NAME = "image"
OUTPUT_NAMES = ("score", "anomaly_map")


def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise distances (N, M), written as matmul so that the graph stays a plain tensor op."""
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    squared = x_norm + y_norm.transpose(-2, -1) - 2.0 * torch.matmul(x, y.transpose(-2, -1))
    return squared.clamp_min(0.0).sqrt()


class PatchcoreOnnxModule(nn.Module):
    """Single-image PatchCore inference as one traceable computation.

    Mirrors `PatchcoreModel.forward` for batch size 1, without the Python chunking loop. The test
    suite compares it against anomalib's implementation.
    """

    def __init__(self, detector: PatchcoreDetector) -> None:
        super().__init__()
        model = detector.model
        if model.memory_bank.numel() == 0:
            raise ValueError("cannot export an empty memory bank")
        self.backbone = model.feature_extractor
        self.pooler = model.feature_pooler
        self.blur = model.anomaly_map_generator.blur
        self.layers = list(model.layers)
        self.num_neighbors = min(model.num_neighbors, model.memory_bank.shape[0])
        self.register_buffer("memory_bank", model.memory_bank.clone())
        self.memory_bank: Tensor

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        output_size = image.shape[-2:]
        features = self.backbone(image)
        first = self.pooler(features[self.layers[0]])
        pooled = [first]
        for layer in self.layers[1:]:
            pooled.append(
                F.interpolate(self.pooler(features[layer]), size=first.shape[-2:], mode="bilinear")
            )
        embedding = torch.cat(pooled, dim=1)
        _, channels, height, width = embedding.shape
        patches = embedding.permute(0, 2, 3, 1).reshape(-1, channels)

        distances = euclidean_distance(patches, self.memory_bank)
        patch_scores, locations = distances.min(dim=1)

        # Image score: the worst patch, reweighted by how isolated its neighbour is in the bank.
        # index_select keeps these lookups exportable, unlike indexing with a data-dependent scalar.
        worst = torch.argmax(patch_scores, dim=0, keepdim=True)
        worst_feature = patches.index_select(0, worst)
        neighbour = self.memory_bank.index_select(0, locations.index_select(0, worst))
        _, support = euclidean_distance(neighbour, self.memory_bank).topk(
            k=self.num_neighbors, largest=False, dim=1
        )
        support_features = self.memory_bank.index_select(0, support.reshape(-1))
        support_distances = euclidean_distance(worst_feature, support_features)
        weight = 1.0 - F.softmax(support_distances, dim=1)[:, 0]
        score = weight * patch_scores.index_select(0, worst)

        coarse = patch_scores.reshape(1, 1, height, width)
        upsampled = F.interpolate(coarse, size=(output_size[0], output_size[1]))
        return score, self.blur(upsampled).squeeze(1)


def export_patchcore(
    detector: PatchcoreDetector,
    path: Path,
    crop: int = 224,
    opset: int = OPSET,
) -> None:
    """Write a self-contained ONNX file, with the memory bank stored inside the graph."""
    if opset < MIN_OPSET:
        raise ValueError(f"opset must be at least {MIN_OPSET}")
    module = PatchcoreOnnxModule(detector).cpu().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (torch.zeros(1, 3, crop, crop),),
        str(path),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(path))


def create_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    """A CPU session. `threads=0` lets ONNX Runtime decide."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one preprocessed CHW image. Returns the scalar score and the (H, W) map."""
    score, anomaly_map = session.run(list(OUTPUT_NAMES), {INPUT_NAME: image[None]})
    return score[0], anomaly_map[0]


def graph_opset(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    return next(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx"))


def write_metadata(
    path: Path, category: str, preprocess: Preprocess, meta: dict[str, Any]
) -> dict[str, Any]:
    """Everything the Rust service needs apart from the graph."""
    metadata = {
        "category": category,
        "threshold": meta["threshold"],
        "heat_low": meta["heat_low"],
        "heat_high": meta["heat_high"],
        "preprocess": asdict(preprocess),
        "input": INPUT_NAME,
        "outputs": list(OUTPUT_NAMES),
        "mlflow_run_id": meta["mlflow_run_id"],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
