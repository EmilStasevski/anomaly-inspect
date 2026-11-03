#!/usr/bin/env python
"""CPU latency of the Rust service, in process and over HTTP, with GPU PyTorch as a reference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from anomaly_inspect.data import Preprocess
from anomaly_inspect.models import build_detector
from anomaly_inspect.pipeline import MODEL_FILE, MODELS_DIR, load_config
from anomaly_inspect.tracking import cpu_model, git_commit, seed_everything

BINARIES = Path("serving/target/release")
ONNX_ROOT = MODELS_DIR / "onnx"
DATA_ROOT = Path("data/mvtec_ad")
# metal_nut has one of the smallest memory banks of the focus categories, screw the largest.
CATEGORIES = {"metal_nut": "scratch/000.png", "screw": "scratch_head/000.png"}
LOAD_GENERATOR_CPUS = "14,15"


def thread_configurations() -> dict[int, str]:
    """CPU sets for the benchmark process. They are specific to the i5-14400F layout.

    CPUs 0-11 are the 6 performance cores with two hardware threads each, 12-15 are efficiency
    cores. Up to 6 threads get one hardware thread per performance core. 8 threads have to add
    two efficiency cores. CPUs 14 and 15 stay free for the load generator.
    """
    return {1: "0", 4: "0,2,4,6", 8: "0,2,4,6,8,10,12,13"}


def cpu_only_env(**extra: str) -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", **extra}


def bench_in_process(
    category: str, image: Path, threads: int, cpus: str, warmup: int, iterations: int
) -> dict[str, Any]:
    output = subprocess.run(
        [
            "taskset", "-c", cpus,
            BINARIES / "bench_inference",
            ONNX_ROOT / category, image, str(threads), str(warmup), str(iterations),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=cpu_only_env(),
    ).stdout  # fmt: skip
    return json.loads(output)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _peak_rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmHWM not found")


def bench_http(
    category: str, image: Path, threads: int, cpus: str, warmup: int, requests_count: int
) -> dict[str, Any]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/predict"
    server = subprocess.Popen(
        ["taskset", "-c", cpus, BINARIES / "anomaly-serving"],
        env=cpu_only_env(
            MODEL_ROOT=str(ONNX_ROOT),
            MODEL_CATEGORIES=category,
            INTRA_THREADS=str(threads),
            BIND_ADDRESS=f"127.0.0.1:{port}",
            RUST_LOG="warn",
        ),
    )
    try:
        for _ in range(120):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                    break
            except requests.ConnectionError:
                time.sleep(0.25)
        else:
            raise RuntimeError("the server did not become healthy")

        boundary = "benchboundary"
        with tempfile.TemporaryDirectory() as tmp:
            body_path = Path(tmp) / "body.bin"
            body_path.write_bytes(
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                f'filename="i.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                + image.read_bytes()
                + f"\r\n--{boundary}--\r\n".encode()
            )

            def oha(count: int) -> dict[str, Any]:
                output = subprocess.run(
                    [
                        "taskset", "-c", LOAD_GENERATOR_CPUS,
                        "oha", "--no-tui", "--output-format", "json",
                        "-n", str(count), "-c", "1", "-m", "POST",
                        "-T", f"multipart/form-data; boundary={boundary}",
                        "-D", str(body_path), url,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout  # fmt: skip
                return json.loads(output)

            oha(warmup)
            report = oha(requests_count)

        codes = report["statusCodeDistribution"]
        if set(codes) != {"200"} or codes["200"] != requests_count:
            raise RuntimeError(f"unexpected responses: {codes}")
        return {
            "category": category,
            "threads": threads,
            "warmup": warmup,
            "requests": requests_count,
            "concurrency": 1,
            "p50_ms": report["latencyPercentiles"]["p50"] * 1e3,
            "p95_ms": report["latencyPercentiles"]["p95"] * 1e3,
            "p99_ms": report["latencyPercentiles"]["p99"] * 1e3,
            "throughput_per_s": report["summary"]["requestsPerSec"],
            "peak_rss_mb": _peak_rss_mb(server.pid),
        }
    finally:
        server.terminate()
        server.wait(timeout=10)


def bench_gpu(category: str, image: Path, warmup: int, iterations: int) -> dict[str, Any] | None:
    """PyTorch GPU latency of model inference only, as a reference."""
    if not torch.cuda.is_available():
        return None
    config = load_config(Path("configs/patchcore.yaml"))
    seed_everything(config["seed"])
    detector = build_detector("patchcore", config["model"]).to("cuda").eval()
    detector.load(MODELS_DIR / "patchcore" / category / MODEL_FILE)
    with Image.open(image) as opened:
        array = Preprocess(**config["preprocess"]).image(opened)
    tensor = torch.from_numpy(array)[None]

    def once() -> None:
        detector.predict(tensor.to("cuda"))
        torch.cuda.synchronize()

    for _ in range(warmup):
        once()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        once()
        samples.append((time.perf_counter() - start) * 1e3)
    result = {
        "category": category,
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup,
        "iterations": iterations,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "throughput_per_s": iterations / (sum(samples) / 1e3),
    }
    return result


def _governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text().strip() if path.is_file() else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/bench.json"))
    args = parser.parse_args()

    if shutil.which("oha") is None:
        print("oha is missing. Install it with: cargo install --locked oha", file=sys.stderr)
        return 1
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd="serving", check=True)

    in_process, http, gpu = [], [], []
    for category, relative in CATEGORIES.items():
        image = DATA_ROOT / category / "test" / relative
        for threads, cpus in thread_configurations().items():
            run = bench_in_process(category, image, threads, cpus, args.warmup, args.iterations)
            run["cpus"] = cpus
            in_process.append(run)
            print(f"in-process {category} {threads}t p50={run['p50_ms']:.1f} ms", flush=True)
            served = bench_http(category, image, threads, cpus, args.warmup, args.iterations)
            served["cpus"] = cpus
            http.append(served)
            print(f"http       {category} {threads}t p50={served['p50_ms']:.1f} ms", flush=True)
        reference = bench_gpu(category, image, max(args.warmup, 20), args.iterations)
        if reference:
            gpu.append(reference)
            print(f"gpu        {category} p50={reference['p50_ms']:.1f} ms", flush=True)

    runtime = in_process[0].pop("runtime", "unknown")
    for run in in_process:
        run.pop("runtime", None)
    results = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": {
            "cpu": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "cpu_governor": _governor(),
            "platform": platform.platform(),
            "onnx_runtime": runtime,
            "torch": torch.__version__,
        },
        "method": {
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "http_tool": subprocess.run(
                ["oha", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "http_concurrency": 1,
            "cuda_visible_devices_for_cpu_runs": "",
        },
        "in_process": in_process,
        "http": http,
        "gpu_reference": gpu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
