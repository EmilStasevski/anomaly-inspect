"""The two detectors behind one interface: fit on good images, then score and localise."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from anomalib.models.components import TimmFeatureExtractor
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torch import Tensor, nn
from torch.nn import functional as F


class Detector(nn.Module, ABC):
    """An anomaly detector. Backbones are pretrained and never updated."""

    @abstractmethod
    def fit(self, batches: Iterable[Tensor]) -> dict[str, float]:
        """Fit on batches of good images already on the device. Returns statistics to log."""

    @abstractmethod
    def predict(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return image scores (B,) and anomaly maps (B, H, W), higher meaning more anomalous."""

    @abstractmethod
    def fitted_state(self) -> dict[str, Tensor]:
        """Tensors learned by `fit`. The pretrained backbone is deliberately excluded."""

    @abstractmethod
    def load_fitted_state(self, state: Mapping[str, Tensor]) -> None:
        """Restore what `fitted_state` returned."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() for key, value in self.fitted_state().items()}, path)

    def load(self, path: Path) -> None:
        device = next(self.parameters()).device
        self.load_fitted_state(torch.load(path, map_location=device, weights_only=True))


class PadimBaseline(Detector):
    """PaDiM-style baseline: one Gaussian per patch location, scored by Mahalanobis distance.

    Follows Defard et al. (2021): multi-scale ResNet18 features reduced to a random subset of
    dimensions. It keeps no images, only one mean and one inverse covariance per location.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        feature_dims: int = 100,
        covariance_epsilon: float = 0.01,
        sigma: int = 4,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.feature_dims = feature_dims
        self.covariance_epsilon = covariance_epsilon
        self.extractor = TimmFeatureExtractor(
            backbone=backbone, layers=self.layers, pre_trained=pre_trained
        ).eval()
        self.map_generator = AnomalyMapGenerator(sigma=sigma)
        self.dim_index: Tensor
        self.mean: Tensor
        self.inverse_covariance: Tensor
        self.register_buffer("dim_index", torch.empty(0, dtype=torch.long))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inverse_covariance", torch.empty(0))

    def _embed(self, images: Tensor) -> Tensor:
        """Return patch embeddings (B, C, H, W) at the resolution of the first layer."""
        features = self.extractor(images)
        first = features[self.layers[0]]
        scaled = [
            F.interpolate(features[layer], size=first.shape[-2:], mode="nearest")
            for layer in self.layers[1:]
        ]
        return torch.cat([first, *scaled], dim=1)

    @torch.no_grad()
    def fit(self, batches: Iterable[Tensor]) -> dict[str, float]:
        embeddings = torch.cat([self._embed(images) for images in batches])
        n_images, channels, height, width = embeddings.shape
        if self.feature_dims > channels:
            raise ValueError(f"feature_dims {self.feature_dims} exceeds {channels} channels")

        # Uses the global torch RNG, seeded by the caller.
        self.dim_index = torch.randperm(channels, device=embeddings.device)[: self.feature_dims]
        # (locations, images, dims). float64: small categories have nearly singular covariances.
        samples = embeddings[:, self.dim_index].flatten(2).permute(2, 0, 1).double()
        mean = samples.mean(dim=1)
        centred = samples - mean[:, None]
        covariance = centred.transpose(1, 2) @ centred / (n_images - 1)
        covariance += self.covariance_epsilon * torch.eye(self.feature_dims).to(covariance)
        self.mean = mean.float()
        self.inverse_covariance = torch.linalg.inv(covariance).float()
        return {"n_fit_images": float(n_images), "n_locations": float(height * width)}

    @torch.no_grad()
    def patch_scores(self, images: Tensor) -> Tensor:
        """Mahalanobis distance of every patch, with shape (B, 1, H, W)."""
        if self.mean.numel() == 0:
            raise RuntimeError("fit or load the model first")
        embedding = self._embed(images)[:, self.dim_index]
        batch, _, height, width = embedding.shape
        delta = embedding.flatten(2).transpose(1, 2) - self.mean
        squared = (delta[:, :, None] @ self.inverse_covariance @ delta[:, :, :, None]).reshape(
            batch, 1, height, width
        )
        return squared.clamp_min(0).sqrt()

    def predict(self, images: Tensor) -> tuple[Tensor, Tensor]:
        maps = self.map_generator(self.patch_scores(images), tuple(images.shape[-2:])).squeeze(1)
        return maps.amax(dim=(1, 2)), maps

    def fitted_state(self) -> dict[str, Tensor]:
        return {
            "dim_index": self.dim_index,
            "mean": self.mean,
            "inverse_covariance": self.inverse_covariance,
        }

    def load_fitted_state(self, state: Mapping[str, Tensor]) -> None:
        self.dim_index = state["dim_index"]
        self.mean = state["mean"]
        self.inverse_covariance = state["inverse_covariance"]


class PatchcoreDetector(Detector):
    """PatchCore (Roth et al., 2022), using anomalib's model and coreset sampler."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.coreset_ratio = coreset_ratio
        self.model = PatchcoreModel(
            layers=list(layers),
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )

    def fit(self, batches: Iterable[Tensor]) -> dict[str, float]:
        self.model.train()  # In training mode the model collects embeddings instead of scoring.
        n_images = 0
        for images in batches:
            self.model(images)
            n_images += images.shape[0]
        n_patches = sum(len(chunk) for chunk in self.model.embedding_store)
        self.model.subsample_embedding(self.coreset_ratio)
        self.model.eval()
        return {
            "n_fit_images": float(n_images),
            "n_patches": float(n_patches),
            "memory_bank_size": float(self.model.memory_bank.shape[0]),
        }

    @torch.no_grad()
    def predict(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.model.eval()
        output = self.model(images)
        return output.pred_score, output.anomaly_map.squeeze(1)

    def fitted_state(self) -> dict[str, Tensor]:
        return {"memory_bank": self.model.memory_bank}

    def load_fitted_state(self, state: Mapping[str, Tensor]) -> None:
        self.model.memory_bank = state["memory_bank"]
        self.model.eval()


def build_detector(family: str, model_config: Mapping[str, Any]) -> Detector:
    options = dict(model_config)
    options["layers"] = tuple(options["layers"])
    if family == "baseline":
        return PadimBaseline(**options)
    if family == "patchcore":
        return PatchcoreDetector(**options)
    raise ValueError(f"unknown model family {family!r}")
