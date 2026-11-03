"""Seeds, provenance and MLflow run handling."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
# Pipeline outputs. Regenerating them does not mean that the code has changed.
GENERATED_PATHSPECS = (
    ":!results",
    ":!docs/results.md",
    ":!docs/figures",
    ":!README.md",
    ":!README.de.md",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the global RNG used by third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    """HEAD hash with a `-dirty` suffix if the work tree has changes, or `unknown` outside git."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", *GENERATED_PATHSPECS],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status.strip() else commit


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment_info() -> dict[str, str]:
    import anomalib
    import onnxruntime

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "cpu_count": str(os.cpu_count()),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "anomalib": anomalib.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config sections to dotted MLflow parameter names."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def configure_tracking(tracking_uri: str | None = None) -> None:
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )


@contextmanager
def tracked_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    tracking_uri: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start a run in `experiment` with params, git commit and environment already logged."""
    configure_tracking(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.set_tags({"git_commit": git_commit(), **environment_info()})
        yield run
