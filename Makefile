.DEFAULT_GOAL := help
.PHONY: help setup data-check train eval export bench report serve test lint format mlflow-up mlflow-down

# MLflow writes its files as the calling user, not as root.
export MLFLOW_UID := $(shell id -u)
export MLFLOW_GID := $(shell id -g)
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000

help: ## List the targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the locked environment and verify the GPU
	uv sync --frozen
	uv run pre-commit install
	uv run python scripts/verify_gpu.py
	cargo build --release --manifest-path serving/Cargo.toml
	cargo install --locked oha

data-check: ## Validate data/mvtec_ad
	uv run python scripts/check_data.py

train: mlflow-up ## Fit the baseline and PatchCore on all categories, tracked in MLflow
	uv run python scripts/train.py --config configs/baseline.yaml
	uv run python scripts/train.py --config configs/patchcore.yaml

eval: mlflow-up ## Evaluate both families on the test split, tracked in MLflow
	uv run python scripts/evaluate.py --config configs/baseline.yaml
	uv run python scripts/evaluate.py --config configs/patchcore.yaml

export: ## Export the focus categories to ONNX and check parity with PyTorch
	uv run python scripts/export_onnx.py

bench: ## Measure CPU latency in process and over HTTP, with a GPU reference
	uv run python scripts/bench.py

report: ## Generate docs/results.md and the README tables from MLflow and the benchmark
	uv run python scripts/generate_report.py

serve: ## Start the inference service on 127.0.0.1:8090 (INFERENCE_PORT changes the port)
	docker compose up -d --build --wait inference

test: ## Run the tests
	uv run pytest
	cargo test --release --manifest-path serving/Cargo.toml

lint: ## Lint, check formatting and type-check
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cargo fmt --manifest-path serving/Cargo.toml --check
	cargo clippy --release --manifest-path serving/Cargo.toml --all-targets -- -D warnings

format: ## Apply formatting and autofixes
	uv run ruff check --fix .
	uv run ruff format .

mlflow-up: ## Start the MLflow server on 127.0.0.1:5000
	docker compose up -d --build --wait mlflow

mlflow-down: ## Stop the MLflow server, keeping its data
	docker compose down
