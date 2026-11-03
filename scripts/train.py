#!/usr/bin/env python
"""Fit one detector per category, calibrate its threshold and log it to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_inspect.data import CATEGORIES
from anomaly_inspect.pipeline import load_config, train_category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = parser.parse_args()

    config = load_config(args.config)
    for category in args.categories:
        stats = train_category(config, category)
        print(
            f"{config['family']:<10}{category:<12}fit {stats['fit_seconds']:7.1f}s  "
            f"threshold {stats['threshold']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
