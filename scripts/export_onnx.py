#!/usr/bin/env python
"""Export fitted PatchCore models to ONNX and compare them with PyTorch on real test images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anomaly_inspect.data import FOCUS_CATEGORIES, Preprocess, list_samples
from anomaly_inspect.export import (
    METADATA_FILE,
    MODEL_FILE,
    create_session,
    export_patchcore,
    graph_opset,
    run_session,
    write_metadata,
)
from anomaly_inspect.models import PatchcoreDetector, build_detector
from anomaly_inspect.pipeline import META_FILE, MODELS_DIR, load_config
from anomaly_inspect.pipeline import MODEL_FILE as FITTED_FILE
from anomaly_inspect.tracking import git_commit

SCORE_TOLERANCE = 1e-3
MAP_TOLERANCE = 1e-2


def export_category(
    config: dict, category: str, output_dir: Path, n_images: int
) -> dict[str, float]:
    fitted_dir = MODELS_DIR / "patchcore" / category
    detector = build_detector("patchcore", config["model"]).eval()
    assert isinstance(detector, PatchcoreDetector)
    detector.load(fitted_dir / FITTED_FILE)

    out_dir = output_dir / category
    export_patchcore(detector, out_dir / MODEL_FILE, config["preprocess"]["crop"])
    preprocess = Preprocess(**config["preprocess"])
    meta = json.loads((fitted_dir / META_FILE).read_text())
    write_metadata(out_dir / METADATA_FILE, category, preprocess, meta)

    session = create_session(out_dir / MODEL_FILE)
    samples = list_samples(Path(config["data"]["root"]), category, "test")
    # Evenly spaced, so that good images and every defect type are represented.
    picked = np.linspace(0, len(samples) - 1, min(n_images, len(samples))).astype(int)

    score_errors, map_errors = [], []
    for index in picked:
        with Image.open(samples[index].image_path) as image:
            array = preprocess.image(image)
        torch_score, torch_map = detector.predict(torch.from_numpy(array)[None])
        onnx_score, onnx_map = run_session(session, array)
        score_errors.append(abs(float(torch_score[0]) - float(onnx_score)))
        map_errors.append(float(np.abs(torch_map[0].numpy() - onnx_map).max()))

    return {
        "opset": graph_opset(out_dir / MODEL_FILE),
        "size_mb": (out_dir / MODEL_FILE).stat().st_size / 1e6,
        "memory_bank": detector.model.memory_bank.shape[0],
        "n_images": len(picked),
        "max_score_error": max(score_errors),
        "max_map_error": max(map_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/patchcore.yaml"))
    parser.add_argument("--categories", nargs="+", default=list(FOCUS_CATEGORIES))
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "onnx")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--results", type=Path, default=Path("results/export.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    failed = False
    reports = {}
    for category in args.categories:
        report = export_category(config, category, args.output_dir, args.n_images)
        ok = (
            report["max_score_error"] <= SCORE_TOLERANCE
            and report["max_map_error"] <= MAP_TOLERANCE
        )
        failed |= not ok
        reports[category] = report
        print(
            f"{category:<12}opset {report['opset']}  {report['size_mb']:7.1f} MB  "
            f"bank {report['memory_bank']:>6}  n={report['n_images']}  "
            f"max|score err| {report['max_score_error']:.2e}  "
            f"max|map err| {report['max_map_error']:.2e}  {'ok' if ok else 'FAIL'}",
            flush=True,
        )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "score_tolerance": SCORE_TOLERANCE,
                "map_tolerance": MAP_TOLERANCE,
                "categories": reports,
            },
            indent=2,
        )
        + "\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
