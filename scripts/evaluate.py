#!/usr/bin/env python
"""Evaluate saved detectors on the test split and log per-category and mean metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_inspect.data import CATEGORIES
from anomaly_inspect.pipeline import evaluate_category, load_config, log_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = parser.parse_args()

    config = load_config(args.config)
    results, run_ids = {}, {}
    print(
        f"{'category':<12}{'img_auroc':>10}{'px_auroc':>10}{'aupro':>8}{'f1_max':>8}{'f1_cal':>8}"
    )
    for category in args.categories:
        run_id, metrics = evaluate_category(config, category)
        results[category], run_ids[category] = metrics, run_id
        print(
            f"{category:<12}{metrics['image_auroc']:>10.4f}{metrics['pixel_auroc']:>10.4f}"
            f"{metrics['aupro']:>8.4f}{metrics['f1_max']:>8.4f}{metrics['f1_calibrated']:>8.4f}",
            flush=True,
        )

    summary = log_summary(config, results, run_ids)
    print(
        f"{'mean':<12}{summary['mean_image_auroc']:>10.4f}{summary['mean_pixel_auroc']:>10.4f}"
        f"{summary['mean_aupro']:>8.4f}{summary['mean_f1_max']:>8.4f}"
        f"{summary['mean_f1_calibrated']:>8.4f}"
    )


if __name__ == "__main__":
    main()
