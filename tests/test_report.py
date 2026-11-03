from __future__ import annotations

import pytest

from anomaly_inspect.data import CATEGORIES
from anomaly_inspect.report import latency_table, replace_block, summary_table


def test_replace_block_keeps_surrounding_text_and_is_repeatable() -> None:
    text = "a\n<!-- generated:x:start -->\nold\n<!-- generated:x:end -->\nb"
    once = replace_block(text, "x", r"new \1 value")
    assert once == "a\n<!-- generated:x:start -->\nnew \\1 value\n<!-- generated:x:end -->\nb"
    assert replace_block(once, "x", r"new \1 value") == once


def test_replace_block_fails_if_markers_are_missing() -> None:
    with pytest.raises(ValueError, match="not found"):
        replace_block("nothing here", "x", "new")


def test_summary_table_averages_over_all_categories() -> None:
    keys = (
        "image_auroc",
        "pixel_auroc",
        "aupro",
        "f1_max",
        "f1_calibrated",
        "recall_calibrated",
        "false_reject_rate_calibrated",
    )
    baseline = {c: dict.fromkeys(keys, 0.5) for c in CATEGORIES}
    patchcore = {c: dict.fromkeys(keys, 0.0) for c in CATEGORIES}
    patchcore["bottle"]["image_auroc"] = 1.5  # outside the focus mean of 0.25 if it were all 6
    row = summary_table(baseline, patchcore).splitlines()[2]
    assert row == "| Image AUROC | 0.500 | 0.250 | 0.500 | 0.100 |"


def test_latency_table_contains_every_run() -> None:
    run = {"category": "a", "threads": 4, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0}
    bench = {
        "in_process": [{**run, "throughput_per_s": 9.0}],
        "http": [{**run, "throughput_per_s": 8.0}],
        "gpu_reference": [{**run, "throughput_per_s": 7.0}],
    }
    lines = latency_table(bench).splitlines()
    assert len(lines) == 5
    assert "| a | HTTP | CPU, 4 threads | 1.0 | 2.0 | 3.0 | 8.0 |" in lines
