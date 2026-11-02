from __future__ import annotations

from pathlib import Path

import mlflow
import torch

from anomaly_inspect.tracking import environment_info, git_commit, seed_everything, tracked_run


def test_seed_everything_is_reproducible() -> None:
    seed_everything(3)
    first = torch.rand(4)
    seed_everything(3)
    assert torch.equal(first, torch.rand(4))


def test_environment_info_reports_real_versions() -> None:
    info = environment_info()
    assert info["torch"] == torch.__version__
    assert info["cpu"]


def test_tracked_run_logs_params_and_provenance(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with tracked_run("unit-test", "run", {"seed": 42}, tracking_uri=uri) as run:
        mlflow.log_metric("image_auroc", 0.5)
    stored = mlflow.MlflowClient(uri).get_run(run.info.run_id)
    assert stored.data.params["seed"] == "42"
    assert stored.data.metrics["image_auroc"] == 0.5
    assert stored.data.tags["git_commit"] == git_commit()
    assert stored.data.tags["torch"] == torch.__version__
