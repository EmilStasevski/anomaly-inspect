#!/usr/bin/env python
"""Refit a category without logging and compare each tensor with the saved model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from anomaly_inspect.data import list_samples, split_calibration
from anomaly_inspect.models import build_detector
from anomaly_inspect.pipeline import MODEL_FILE, MODELS_DIR, _loader, device, load_config
from anomaly_inspect.tracking import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--category", default="metal_nut")
    args = parser.parse_args()

    config = load_config(args.config)
    target = device()
    data = config["data"]
    seed_everything(config["seed"])
    fit_samples, _ = split_calibration(
        list_samples(Path(data["root"]), args.category, "train"),
        data["calibration_fraction"],
        config["seed"],
    )
    # Same order as training: the backbone is built first, then the data is read and fitted.
    detector = build_detector(config["family"], config["model"]).to(target).eval()
    detector.fit(batch["image"].to(target) for batch in _loader(config, fit_samples))

    saved_path = MODELS_DIR / config["family"] / args.category / MODEL_FILE
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    identical = True
    for name, value in detector.fitted_state().items():
        same = torch.equal(value.cpu(), saved[name])
        identical &= same
        print(
            f"{config['family']} {args.category} {name} {tuple(value.shape)}: "
            f"{'identical' if same else 'DIFFERENT'}"
        )
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
