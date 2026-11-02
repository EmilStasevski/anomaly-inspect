#!/usr/bin/env python
"""Validate the MVTec AD directory layout and count the images per category."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anomaly_inspect.data import CATEGORIES, EXPECTED_COUNTS, FOCUS_CATEGORIES, validate_category

DOWNLOAD_HELP = """\
MVTec AD was not found at {root}.

It has to be downloaded by hand because the license (CC BY-NC-SA 4.0) must be accepted:
  1. Open https://www.mvtec.com/company/research/datasets/mvtec-ad and accept the license.
  2. Download the complete dataset archive (about 5 GB).
  3. Extract it so that {root}/bottle/train/good/000.png exists.
Then run `make data-check` again.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        help="subset of categories to check (default: all 15)",
    )
    args = parser.parse_args()

    if not args.root.is_dir() or not any(args.root.iterdir()):
        print(DOWNLOAD_HELP.format(root=args.root), file=sys.stderr)
        return 1

    unknown = sorted(set(args.categories) - set(CATEGORIES))
    if unknown:
        print(f"Unknown categories: {unknown}", file=sys.stderr)
        return 2

    print(f"{'category':<12}{'train':>7}{'test_good':>11}{'test_defect':>13}{'types':>7}  status")
    failures = 0
    totals = [0, 0, 0]
    for category in args.categories:
        report = validate_category(args.root, category, EXPECTED_COUNTS[category])
        marker = "*" if category in FOCUS_CATEGORIES else " "
        status = "ok" if report.ok else "FAIL"
        print(
            f"{category + marker:<12}{report.train_good:>7}{report.test_good:>11}"
            f"{report.test_defect:>13}{len(report.defect_types):>7}  {status}"
        )
        for error in report.errors:
            print(f"    - {error}")
        failures += not report.ok
        totals[0] += report.train_good
        totals[1] += report.test_good
        totals[2] += report.test_defect

    print(f"{'total':<12}{totals[0]:>7}{totals[1]:>11}{totals[2]:>13}")
    print("* focus category")
    if failures:
        print(f"{failures} categories failed validation", file=sys.stderr)
        return 1
    print(f"All {len(args.categories)} categories valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
