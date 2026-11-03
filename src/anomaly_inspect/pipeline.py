"""Train and evaluate per category, with everything tracked in MLflow."""

from __future__ import annotations

import csv
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import (
    FOCUS_CATEGORIES,
    MVTecDataset,
    Preprocess,
    Sample,
    list_samples,
    split_calibration,
)
from .metrics import (
    aupro,
    calibrate_threshold,
    decision_rates,
    f1_at_threshold,
    f1_max,
    image_auroc,
    pixel_auroc,
)
from .models import Detector, build_detector
from .tracking import configure_tracking, flatten_params, seed_everything, tracked_run
from .visualization import HeatScale, save_heatmap_figure, select_examples

MODELS_DIR = Path("models")
MODEL_FILE = "model.pt"
META_FILE = "meta.json"
METRIC_NAMES = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "f1_max",
    "f1_calibrated",
    "recall_calibrated",
    "false_reject_rate_calibrated",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocess(config: dict[str, Any]) -> Preprocess:
    return Preprocess(**config["preprocess"])


def _loader(config: dict[str, Any], samples: list[Sample]) -> DataLoader:
    data = config["data"]
    return DataLoader(
        MVTecDataset(samples, _preprocess(config)),
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class Predictions:
    scores: np.ndarray
    maps: np.ndarray
    masks: np.ndarray
    labels: np.ndarray


def predict_all(detector: Detector, loader: DataLoader, target: torch.device) -> Predictions:
    scores, maps, masks, labels = [], [], [], []
    for batch in loader:
        batch_scores, batch_maps = detector.predict(batch["image"].to(target))
        scores.append(batch_scores.cpu().numpy())
        maps.append(batch_maps.cpu().numpy())
        masks.append(batch["mask"].numpy())
        labels.append(batch["label"].numpy())
    return Predictions(
        np.concatenate(scores), np.concatenate(maps), np.concatenate(masks), np.concatenate(labels)
    )


def compute_metrics(predictions: Predictions, threshold: float) -> dict[str, float]:
    labels, scores = predictions.labels, predictions.scores
    best_f1, best_threshold = f1_max(labels, scores)
    recall, false_reject = decision_rates(labels, scores, threshold)
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(predictions.masks, predictions.maps),
        "aupro": aupro(predictions.masks, predictions.maps),
        "f1_max": best_f1,
        "f1_max_threshold": best_threshold,
        "f1_calibrated": f1_at_threshold(labels, scores, threshold),
        "recall_calibrated": recall,
        "false_reject_rate_calibrated": false_reject,
    }


def train_category(config: dict[str, Any], category: str) -> dict[str, float]:
    """Fit on good images, calibrate on held-out good images, save and log the model."""
    family, seed, data = config["family"], config["seed"], config["data"]
    root = Path(data["root"])
    target = device()
    seed_everything(seed)

    fit_samples, calibration_samples = split_calibration(
        list_samples(root, category, "train"), data["calibration_fraction"], seed
    )
    params = {**flatten_params(config), "category": category}

    with tracked_run(family, f"{family}-{category}", params) as run:
        detector = build_detector(family, config["model"]).to(target).eval()

        start = time.perf_counter()
        batches = (batch["image"].to(target) for batch in _loader(config, fit_samples))
        fit_stats = detector.fit(batches)
        fit_seconds = time.perf_counter() - start

        calibration = predict_all(detector, _loader(config, calibration_samples), target)
        sigmas = config["threshold"]["sigmas"]
        threshold = calibrate_threshold(calibration.scores, sigmas)
        scale = HeatScale.from_good_maps(calibration.maps)

        out_dir = MODELS_DIR / family / category
        detector.save(out_dir / MODEL_FILE)
        meta = {
            "family": family,
            "category": category,
            "seed": seed,
            "threshold": threshold,
            "heat_low": scale.low,
            "heat_high": scale.high,
            "n_calibration": len(calibration_samples),
            "mlflow_run_id": run.info.run_id,
            "config": config,
        }
        (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")

        stats = {
            **fit_stats,
            "fit_seconds": fit_seconds,
            "threshold": threshold,
            "n_calibration": float(len(calibration_samples)),
        }
        mlflow.log_metrics(stats)
        mlflow.log_artifact(str(out_dir / MODEL_FILE), artifact_path="model")
        mlflow.log_artifact(str(out_dir / META_FILE), artifact_path="model")

    del detector
    torch.cuda.empty_cache()
    return stats


def evaluate_category(config: dict[str, Any], category: str) -> tuple[str, dict[str, float]]:
    """Evaluate a saved model on the test split and log into the run that trained it."""
    family = config["family"]
    root = Path(config["data"]["root"])
    target = device()
    model_dir = MODELS_DIR / family / category
    meta = json.loads((model_dir / META_FILE).read_text())

    detector = build_detector(family, config["model"]).to(target).eval()
    detector.load(model_dir / MODEL_FILE)
    samples = list_samples(root, category, "test")

    start = time.perf_counter()
    predictions = predict_all(detector, _loader(config, samples), target)
    eval_seconds = time.perf_counter() - start

    metrics = compute_metrics(predictions, meta["threshold"])
    metrics["eval_seconds"] = eval_seconds

    with tempfile.TemporaryDirectory() as tmp:
        figure = Path(tmp) / f"{family}_{category}.png"
        save_heatmap_figure(
            samples,
            select_examples(samples),
            predictions.maps,
            predictions.masks,
            predictions.scores,
            meta["threshold"],
            HeatScale(meta["heat_low"], meta["heat_high"]),
            _preprocess(config),
            f"{family}: {category}",
            figure,
        )
        scores_csv = Path(tmp) / "scores.csv"
        with scores_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "defect_type", "label", "score"])
            for sample, score in zip(samples, predictions.scores, strict=True):
                relative = sample.image_path.relative_to(root)
                writer.writerow([relative, sample.defect_type, sample.label, float(score)])

        configure_tracking()
        with mlflow.start_run(run_id=meta["mlflow_run_id"]):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(figure), artifact_path="heatmaps")
            mlflow.log_artifact(str(scores_csv), artifact_path="predictions")

    del detector
    torch.cuda.empty_cache()
    return meta["mlflow_run_id"], metrics


def log_summary(
    config: dict[str, Any],
    results: dict[str, dict[str, float]],
    run_ids: dict[str, str],
) -> dict[str, float]:
    """Log mean metrics over all categories and over the focus categories."""
    family = config["family"]
    summary: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [results[category][name] for category in results]
        summary[f"mean_{name}"] = float(np.mean(values))
        focus = [results[c][name] for c in FOCUS_CATEGORIES if c in results]
        if focus:
            summary[f"focus_mean_{name}"] = float(np.mean(focus))

    params = {**flatten_params(config), "n_categories": len(results)}
    with tracked_run(family, f"{family}-summary", params):
        mlflow.set_tag("run_type", "summary")
        mlflow.log_metrics(summary)
        mlflow.log_dict({"run_ids": run_ids, "per_category": results}, "summary/per_category.json")
    return summary
