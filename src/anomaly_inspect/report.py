"""Render results as Markdown. Every number comes from MLflow or a benchmark output file."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .data import CATEGORIES, FOCUS_CATEGORIES

Metrics = Mapping[str, float]


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(headers))) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def replace_block(text: str, name: str, content: str) -> str:
    """Replace what is between the markers of a generated block. Missing markers are an error."""
    start, end = f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"block {name!r} not found")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text)


def _mean(per_category: Mapping[str, Metrics], metric: str, categories: Sequence[str]) -> float:
    return sum(per_category[c][metric] for c in categories) / len(categories)


def summary_table(baseline: Mapping[str, Metrics], patchcore: Mapping[str, Metrics]) -> str:
    """Mean metrics of both families, over the focus categories and over all 15."""
    metrics = [
        ("Image AUROC", "image_auroc"),
        ("Pixel AUROC", "pixel_auroc"),
        ("AUPRO (FPR ≤ 0.3)", "aupro"),
        ("F1, best threshold (uses test labels)", "f1_max"),
        ("F1, calibrated threshold", "f1_calibrated"),
        ("Defect recall, calibrated", "recall_calibrated"),
        ("False reject rate, calibrated", "false_reject_rate_calibrated"),
    ]
    rows = []
    for label, key in metrics:
        rows.append(
            [
                label,
                f"{_mean(baseline, key, FOCUS_CATEGORIES):.3f}",
                f"{_mean(patchcore, key, FOCUS_CATEGORIES):.3f}",
                f"{_mean(baseline, key, CATEGORIES):.3f}",
                f"{_mean(patchcore, key, CATEGORIES):.3f}",
            ]
        )
    return table(
        [
            "Mean",
            "Baseline (focus 6)",
            "PatchCore (focus 6)",
            "Baseline (all 15)",
            "PatchCore (all 15)",
        ],
        rows,
    )


def category_table(
    baseline: Mapping[str, Metrics],
    patchcore: Mapping[str, Metrics],
    categories: Sequence[str],
    metrics: Sequence[tuple[str, str]],
) -> str:
    """One row per category, with a baseline and a PatchCore column per metric."""
    headers = ["Category"]
    for label, _ in metrics:
        headers += [f"{label} baseline", f"{label} PatchCore"]
    rows = []
    for category in categories:
        row = [category]
        for _, key in metrics:
            row += [f"{baseline[category][key]:.3f}", f"{patchcore[category][key]:.3f}"]
        rows.append(row)
    return table(headers, rows)


def latency_table(bench: Mapping[str, Any]) -> str:
    """CPU rows per category, thread count and level, then the GPU reference rows."""
    rows = []
    for level, key in (("In process", "in_process"), ("HTTP", "http")):
        for run in bench[key]:
            rows.append(
                [
                    run["category"],
                    level,
                    f"CPU, {run['threads']} thread{'s' if run['threads'] > 1 else ''}",
                    f"{run['p50_ms']:.1f}",
                    f"{run['p95_ms']:.1f}",
                    f"{run['p99_ms']:.1f}",
                    f"{run['throughput_per_s']:.1f}",
                ]
            )
    rows.sort(key=lambda row: row[0])
    for run in bench["gpu_reference"]:
        rows.append(
            [
                run["category"],
                "In process (PyTorch)",
                "GPU",
                f"{run['p50_ms']:.1f}",
                f"{run['p95_ms']:.1f}",
                f"{run['p99_ms']:.1f}",
                f"{run['throughput_per_s']:.1f}",
            ]
        )
    return table(["Category", "Level", "Device", "p50 ms", "p95 ms", "p99 ms", "Images/s"], rows)


def export_table(export: Mapping[str, Any]) -> str:
    rows = [
        [
            category,
            str(report["opset"]),
            f"{report['size_mb']:.0f}",
            str(report["memory_bank"]),
            str(report["n_images"]),
            f"{report['max_score_error']:.1e}",
            f"{report['max_map_error']:.1e}",
        ]
        for category, report in export["categories"].items()
    ]
    return table(
        ["Category", "Opset", "MB", "Memory bank", "Images", "Max score error", "Max map error"],
        rows,
    )


def environment_line(bench: Mapping[str, Any]) -> str:
    environment, method = bench["environment"], bench["method"]
    runtime = re.search(r"git-branch=rel-([0-9.]+)", environment["onnx_runtime"])
    version = runtime.group(1) if runtime else environment["onnx_runtime"]
    memory = [run["peak_rss_mb"] for run in bench["http"]]
    return (
        f"{environment['cpu']}, {environment['logical_cpus']} logical CPUs, "
        f"governor `{environment['cpu_governor']}`, ONNX Runtime {version}, "
        f"batch size {method['batch_size']}, {method['warmup']} warm-up runs, "
        f"{method['iterations']} measured runs, {method['http_tool']} at concurrency "
        f"{method['http_concurrency']}. Peak server memory with one model loaded: "
        f"{min(memory):.0f} to {max(memory):.0f} MB. Commit `{bench['git_commit'][:12]}`."
    )
