"""Render results as Markdown. Every number comes from MLflow or a benchmark output file."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .data import CATEGORIES, FOCUS_CATEGORIES

Metrics = Mapping[str, float]


LABELS: dict[str, dict[str, str]] = {
    "en": {
        "mean": "Mean",
        "category": "Category",
        "baseline_focus": "Baseline (focus 6)",
        "patchcore_focus": "PatchCore (focus 6)",
        "baseline_all": "Baseline (all 15)",
        "patchcore_all": "PatchCore (all 15)",
        "image_auroc": "Image AUROC",
        "pixel_auroc": "Pixel AUROC",
        "aupro": "AUPRO (FPR ≤ 0.3)",
        "aupro_short": "AUPRO",
        "f1_max": "F1, best threshold (uses test labels)",
        "f1_calibrated": "F1, calibrated threshold",
        "recall_calibrated": "Defect recall, calibrated",
        "false_reject_rate_calibrated": "False reject rate, calibrated",
        "level": "Level",
        "device": "Device",
        "images_per_s": "Images/s",
        "in_process": "In process",
        "gpu_level": "In process (PyTorch)",
        "thread": "thread",
        "threads": "threads",
        "memory_bank": "Memory bank",
        "images": "Images",
        "max_score_error": "Max score error",
        "max_map_error": "Max map error",
    },
    "de": {
        "mean": "Mittelwert",
        "category": "Kategorie",
        "baseline_focus": "Baseline (Fokus 6)",
        "patchcore_focus": "PatchCore (Fokus 6)",
        "baseline_all": "Baseline (alle 15)",
        "patchcore_all": "PatchCore (alle 15)",
        "image_auroc": "Image-AUROC",
        "pixel_auroc": "Pixel-AUROC",
        "aupro": "AUPRO (FPR ≤ 0,3)",
        "aupro_short": "AUPRO",
        "f1_max": "F1, bester Schwellwert (nutzt Testlabels)",
        "f1_calibrated": "F1, kalibrierter Schwellwert",
        "recall_calibrated": "Defekt-Recall, kalibriert",
        "false_reject_rate_calibrated": "Falschausschuss-Rate, kalibriert",
        "level": "Ebene",
        "device": "Gerät",
        "images_per_s": "Bilder/s",
        "in_process": "Im Prozess",
        "gpu_level": "Im Prozess (PyTorch)",
        "thread": "Thread",
        "threads": "Threads",
        "memory_bank": "Memory Bank",
        "images": "Bilder",
        "max_score_error": "Max. Score-Fehler",
        "max_map_error": "Max. Map-Fehler",
    },
}


def _localise(cell: str, language: str) -> str:
    """Use a decimal comma in German. Applied to numeric table cells only."""
    return cell.replace(".", ",") if language == "de" else cell


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


def summary_table(
    baseline: Mapping[str, Metrics], patchcore: Mapping[str, Metrics], language: str = "en"
) -> str:
    """Mean metrics of both families, over the focus categories and over all 15."""
    labels = LABELS[language]
    keys = (
        "image_auroc",
        "pixel_auroc",
        "aupro",
        "f1_max",
        "f1_calibrated",
        "recall_calibrated",
        "false_reject_rate_calibrated",
    )
    rows = []
    for key in keys:
        values = [
            _mean(baseline, key, FOCUS_CATEGORIES),
            _mean(patchcore, key, FOCUS_CATEGORIES),
            _mean(baseline, key, CATEGORIES),
            _mean(patchcore, key, CATEGORIES),
        ]
        rows.append([labels[key], *(_localise(f"{value:.3f}", language) for value in values)])
    return table(
        [
            labels["mean"],
            labels["baseline_focus"],
            labels["patchcore_focus"],
            labels["baseline_all"],
            labels["patchcore_all"],
        ],
        rows,
    )


def category_table(
    baseline: Mapping[str, Metrics],
    patchcore: Mapping[str, Metrics],
    categories: Sequence[str],
    metrics: Sequence[tuple[str, str]],
    language: str = "en",
) -> str:
    """One row per category, with a baseline and a PatchCore column per metric."""
    headers = [LABELS[language]["category"]]
    for label, _ in metrics:
        headers += [f"{label} Baseline", f"{label} PatchCore"]
    rows = []
    for category in categories:
        row = [category]
        for _, key in metrics:
            row += [
                _localise(f"{baseline[category][key]:.3f}", language),
                _localise(f"{patchcore[category][key]:.3f}", language),
            ]
        rows.append(row)
    return table(headers, rows)


def focus_table(
    baseline: Mapping[str, Metrics], patchcore: Mapping[str, Metrics], language: str
) -> str:
    labels = LABELS[language]
    metrics = [
        (labels["image_auroc"], "image_auroc"),
        (labels["pixel_auroc"], "pixel_auroc"),
        (labels["aupro_short"], "aupro"),
    ]
    return category_table(baseline, patchcore, FOCUS_CATEGORIES, metrics, language)


def latency_table(bench: Mapping[str, Any], language: str = "en") -> str:
    """CPU rows per category, thread count and level, then the GPU reference rows."""
    labels = LABELS[language]

    def timings(run: Mapping[str, Any]) -> list[str]:
        keys = ("p50_ms", "p95_ms", "p99_ms", "throughput_per_s")
        return [_localise(f"{run[key]:.1f}", language) for key in keys]

    rows = []
    for level, key in ((labels["in_process"], "in_process"), ("HTTP", "http")):
        for run in bench[key]:
            word = labels["threads"] if run["threads"] > 1 else labels["thread"]
            rows.append([run["category"], level, f"CPU, {run['threads']} {word}", *timings(run)])
    rows.sort(key=lambda row: row[0])
    for run in bench["gpu_reference"]:
        rows.append([run["category"], labels["gpu_level"], "GPU", *timings(run)])
    return table(
        [
            labels["category"],
            labels["level"],
            labels["device"],
            "p50 ms",
            "p95 ms",
            "p99 ms",
            labels["images_per_s"],
        ],
        rows,
    )


def export_table(export: Mapping[str, Any], language: str = "en") -> str:
    labels = LABELS[language]
    rows = [
        [
            category,
            str(report["opset"]),
            f"{report['size_mb']:.0f}",
            str(report["memory_bank"]),
            str(report["n_images"]),
            _localise(f"{report['max_score_error']:.1e}", language),
            _localise(f"{report['max_map_error']:.1e}", language),
        ]
        for category, report in export["categories"].items()
    ]
    return table(
        [
            labels["category"],
            "Opset",
            "MB",
            labels["memory_bank"],
            labels["images"],
            labels["max_score_error"],
            labels["max_map_error"],
        ],
        rows,
    )


def _runtime_version(bench: Mapping[str, Any]) -> str:
    runtime = bench["environment"]["onnx_runtime"]
    match = re.search(r"git-branch=rel-([0-9.]+)", runtime)
    return match.group(1) if match else runtime


def environment_line(bench: Mapping[str, Any], language: str = "en") -> str:
    environment, method = bench["environment"], bench["method"]
    memory = [run["peak_rss_mb"] for run in bench["http"]]
    low, high = f"{min(memory):.0f}", f"{max(memory):.0f}"
    commit = bench["git_commit"][:12]
    if language == "de":
        return (
            f"{environment['cpu']}, {environment['logical_cpus']} logische CPUs, "
            f"Governor `{environment['cpu_governor']}`, ONNX Runtime {_runtime_version(bench)}, "
            f"Batchgröße {method['batch_size']}, {method['warmup']} Warm-up-Läufe, "
            f"{method['iterations']} Messläufe, {method['http_tool']} mit Concurrency "
            f"{method['http_concurrency']}. Spitzenspeicher des Servers mit einem geladenen "
            f"Modell: {low} bis {high} MB. Commit `{commit}`."
        )
    return (
        f"{environment['cpu']}, {environment['logical_cpus']} logical CPUs, "
        f"governor `{environment['cpu_governor']}`, ONNX Runtime {_runtime_version(bench)}, "
        f"batch size {method['batch_size']}, {method['warmup']} warm-up runs, "
        f"{method['iterations']} measured runs, {method['http_tool']} at concurrency "
        f"{method['http_concurrency']}. Peak server memory with one model loaded: "
        f"{low} to {high} MB. Commit `{commit}`."
    )


def _run(bench: Mapping[str, Any], section: str, category: str, threads: int) -> Mapping[str, Any]:
    return next(
        run for run in bench[section] if run["category"] == category and run["threads"] == threads
    )


def highlights(
    baseline: Mapping[str, Metrics],
    patchcore: Mapping[str, Metrics],
    bench: Mapping[str, Any],
    export: Mapping[str, Any],
    language: str = "en",
) -> str:
    """The headline numbers. Generated, so that they cannot drift from the logged results."""
    auroc = _mean(patchcore, "image_auroc", CATEGORIES)
    base_auroc = _mean(baseline, "image_auroc", CATEGORIES)
    f1 = _mean(patchcore, "f1_calibrated", CATEGORIES)
    base_f1 = _mean(baseline, "f1_calibrated", CATEGORIES)
    recall = _mean(patchcore, "recall_calibrated", CATEGORIES) * 100
    rejects = _mean(patchcore, "false_reject_rate_calibrated", CATEGORIES) * 100
    http = _run(bench, "http", "metal_nut", 4)["p50_ms"]
    gpu = next(r for r in bench["gpu_reference"] if r["category"] == "metal_nut")["p50_ms"]
    error = max(report["max_score_error"] for report in export["categories"].values())
    if language == "de":
        comma = lambda text: text.replace(".", ",")  # noqa: E731
        return "\n".join(
            [
                f"- **{comma(f'{auroc:.3f}')} mittlerer Image-AUROC** über alle 15 Kategorien "
                f"(Baseline: {comma(f'{base_auroc:.3f}')})",
                f"- **{comma(f'{recall:.1f}')} % der Defekte erkannt** bei "
                f"{comma(f'{rejects:.1f}')} % Falschausschuss, mit einem Schwellwert "
                "ausschließlich aus Gutteilen",
                f"- **{comma(f'{f1:.3f}')} F1 ohne Testlabels** "
                f"(Baseline: {comma(f'{base_f1:.3f}')})",
                f"- **{comma(f'{http:.0f}')} ms Median pro HTTP-Anfrage auf 4 CPU-Threads** "
                f"(metal_nut), {comma(f'{gpu:.0f}')} ms als GPU-Referenz ohne HTTP",
                f"- **ONNX-Parität:** maximale Score-Abweichung {comma(f'{error:.1e}')} gegenüber "
                "PyTorch auf echten Testbildern",
            ]
        )
    return "\n".join(
        [
            f"- **{auroc:.3f} mean image AUROC** across all 15 categories "
            f"(baseline: {base_auroc:.3f})",
            f"- **{recall:.1f}% of defects caught** at a {rejects:.1f}% false reject rate, with a "
            "threshold set from good images alone",
            f"- **{f1:.3f} F1 without touching test labels** (baseline: {base_f1:.3f})",
            f"- **{http:.0f} ms median per HTTP request on 4 CPU threads** (metal_nut), with "
            f"{gpu:.0f} ms as the GPU reference without HTTP",
            f"- **ONNX parity:** maximum score difference of {error:.1e} from PyTorch on real "
            "test images",
        ]
    )
