"""README charts and the concept diagram, drawn from logged results, in English and German."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Categorical slots 1 and 2 of the validated palette.
PATCHCORE_COLOR = "#2a78d6"
BASELINE_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
PANEL = "#f1f0ec"

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "auroc_title": "Image AUROC per category",
        "auroc_axis": "Image AUROC (detection ranking, 1.0 is perfect)",
        "f1_title": "F1 per category, threshold from good images only",
        "f1_axis": "F1 at the calibrated threshold",
        "latency_title": "Inference latency per image, without HTTP",
        "latency_axis": "Median latency in ms, lower is better",
        "thread": "CPU, {} thread",
        "threads": "CPU, {} threads",
        "gpu": "GPU reference",
        "baseline": "Baseline",
        "fit": "FIT: good parts only",
        "good": "Images of\ngood parts",
        "features": "Pretrained\nfeatures",
        "coreset": "Coreset\nsubsampling",
        "bank": "Memory bank\nof normal patches",
        "inspect": "INSPECT: every new part",
        "new": "New part",
        "same": "Same\nfeatures",
        "nearest": "Nearest-neighbour\nsearch",
        "result": "Score + heatmap\nPass or defect",
        "threshold": "Threshold from\ngood images only",
    },
    "de": {
        "auroc_title": "Image-AUROC je Kategorie",
        "auroc_axis": "Image-AUROC (Erkennungsgüte, 1,0 ist perfekt)",
        "f1_title": "F1 je Kategorie, Schwellwert nur aus Gutteilen",
        "f1_axis": "F1 beim kalibrierten Schwellwert",
        "latency_title": "Inferenzlatenz pro Bild, ohne HTTP",
        "latency_axis": "Median-Latenz in ms, niedriger ist besser",
        "thread": "CPU, {} Thread",
        "threads": "CPU, {} Threads",
        "gpu": "GPU-Referenz",
        "baseline": "Baseline",
        "fit": "FIT: nur Gutteile",
        "good": "Bilder von\nGutteilen",
        "features": "Vortrainierte\nMerkmale",
        "coreset": "Coreset-\nReduktion",
        "bank": "Memory Bank\nnormaler Patches",
        "inspect": "PRÜFEN: jedes neue Teil",
        "new": "Neues Teil",
        "same": "Gleiche\nMerkmale",
        "nearest": "Nächster-Nachbar-\nSuche",
        "result": "Score + Heatmap\nGut oder Defekt",
        "threshold": "Schwellwert nur\naus Gutteilen",
    },
}


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, length=0)


def _finish(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def _dumbbell(
    baseline: Mapping[str, Mapping[str, float]],
    patchcore: Mapping[str, Mapping[str, float]],
    metric: str,
    title: str,
    axis_label: str,
    baseline_label: str,
    path: Path,
    decimal_comma: bool,
) -> None:
    """One row per category, sorted so that the largest gap comes first."""
    categories = sorted(baseline, key=lambda c: patchcore[c][metric] - baseline[c][metric])
    figure, axis = plt.subplots(figsize=(8.0, 0.36 * len(categories) + 1.5), facecolor=SURFACE)
    _style(axis)
    for row, category in enumerate(categories):
        left, right = baseline[category][metric], patchcore[category][metric]
        axis.plot([left, right], [row, row], color=GRID, linewidth=2, zorder=1)
        axis.scatter(left, row, s=70, color=BASELINE_COLOR, edgecolor=SURFACE, zorder=2)
        axis.scatter(right, row, s=70, color=PATCHCORE_COLOR, edgecolor=SURFACE, zorder=3)
    axis.set_yticks(range(len(categories)), categories, color=INK)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel(axis_label, color=INK_SECONDARY)
    if decimal_comma:
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}".replace(".", ","))
    axis.set_title(title, loc="left", color=INK, fontweight="bold", pad=26)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=PATCHCORE_COLOR, markersize=8),
        Line2D([], [], marker="o", linestyle="", color=BASELINE_COLOR, markersize=8),
    ]
    legend = axis.legend(
        handles,
        ["PatchCore", baseline_label],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncols=2,
        frameon=False,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _finish(figure, path)


def auroc_chart(baseline, patchcore, path: Path, language: str) -> None:
    text = TEXT[language]
    _dumbbell(
        baseline,
        patchcore,
        "image_auroc",
        text["auroc_title"],
        text["auroc_axis"],
        text["baseline"],
        path,
        language == "de",
    )


def calibrated_f1_chart(baseline, patchcore, path: Path, language: str) -> None:
    text = TEXT[language]
    _dumbbell(
        baseline,
        patchcore,
        "f1_calibrated",
        text["f1_title"],
        text["f1_axis"],
        text["baseline"],
        path,
        language == "de",
    )


def latency_chart(bench: Mapping[str, Any], path: Path, language: str) -> None:
    """Median in-process latency per device and category, on one linear axis."""
    text = TEXT[language]
    categories = list(dict.fromkeys(run["category"] for run in bench["in_process"]))
    threads = sorted({run["threads"] for run in bench["in_process"]})
    rows = [text["thread" if n == 1 else "threads"].format(n) for n in threads] + [text["gpu"]]

    def value(category: str, row: int) -> float:
        if row < len(threads):
            return next(
                run["p50_ms"]
                for run in bench["in_process"]
                if run["category"] == category and run["threads"] == threads[row]
            )
        return next(r["p50_ms"] for r in bench["gpu_reference"] if r["category"] == category)

    figure, axis = plt.subplots(figsize=(8.0, 3.8), facecolor=SURFACE)
    _style(axis)
    height = 0.36
    colors = (PATCHCORE_COLOR, BASELINE_COLOR)
    for series, category in enumerate(categories):
        offset = (series - 0.5) * (height + 0.03)
        values = [value(category, row) for row in range(len(rows))]
        positions = [row + offset for row in range(len(rows))]
        axis.barh(positions, values, height=height, color=colors[series], label=category)
        for position, measured in zip(positions, values, strict=True):
            label = f"{measured:.0f} ms"
            axis.text(measured + 6, position, label, va="center", color=INK, fontsize=9)
    axis.set_yticks(range(len(rows)), rows, color=INK)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel(text["latency_axis"], color=INK_SECONDARY)
    axis.set_title(text["latency_title"], loc="left", color=INK, fontweight="bold", pad=26)
    legend = axis.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, borderaxespad=0
    )
    for entry in legend.get_texts():
        entry.set_color(INK_SECONDARY)
    _finish(figure, path)


def _box(axis: plt.Axes, x: float, y: float, label: str, fill: str, ink: str = INK) -> None:
    width, height = 2.05, 1.1
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill,
            edgecolor="none",
        )
    )
    axis.text(
        x + width / 2, y + height / 2, label, ha="center", va="center", color=ink, fontsize=10
    )


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color=INK_SECONDARY
        )
    )


def how_it_works(path: Path, language: str) -> None:
    """A schematic with no data in it: how a memory bank turns good images into decisions."""
    text = TEXT[language]
    figure, axis = plt.subplots(figsize=(10.5, 4.3), facecolor=SURFACE)
    axis.set_xlim(0, 10.5)
    axis.set_ylim(0, 4.6)
    axis.axis("off")
    xs = [0.1, 2.75, 5.4, 8.05]
    top, bottom = 2.95, 0.65
    axis.text(0.1, 4.25, text["fit"], color=INK_SECONDARY, fontweight="bold", fontsize=10)
    axis.text(0.1, 1.95, text["inspect"], color=INK_SECONDARY, fontweight="bold", fontsize=10)

    for x, key in zip(xs[:3], ("good", "features", "coreset"), strict=True):
        _box(axis, x, top, text[key], PANEL)
    _box(axis, xs[3], top, text["bank"], PATCHCORE_COLOR, "#ffffff")
    for x, key in zip(xs[:2], ("new", "same"), strict=True):
        _box(axis, x, bottom, text[key], PANEL)
    _box(axis, xs[2], bottom, text["nearest"], PANEL)
    _box(axis, xs[3], bottom, text["result"], BASELINE_COLOR, "#ffffff")

    for x_from, x_to in pairwise(xs):
        _arrow(axis, (x_from + 2.1, top + 0.55), (x_to - 0.05, top + 0.55))
        _arrow(axis, (x_from + 2.1, bottom + 0.55), (x_to - 0.05, bottom + 0.55))
    # The bank feeds the search, and the threshold decides the outcome.
    _arrow(axis, (xs[3] + 0.3, top - 0.03), (xs[2] + 1.75, bottom + 1.13))
    axis.text(
        xs[3] + 1.03,
        0.3,
        text["threshold"].replace("\n", " "),
        ha="center",
        va="center",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    _finish(figure, path)
