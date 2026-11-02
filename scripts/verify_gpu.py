#!/usr/bin/env python
"""Fail loudly unless PyTorch can run a real computation on the expected GPU."""

from __future__ import annotations

import sys

import torch

EXPECTED_CAPABILITY = (12, 0)


def main() -> int:
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    capability = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)} | capability: {capability}")
    if capability != EXPECTED_CAPABILITY:
        print(
            f"WARNING: expected capability {EXPECTED_CAPABILITY}, this result is machine-specific"
        )

    # Solve a linear system so that the check exercises real kernels, not only an allocation.
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1, device="cuda")
    x = torch.linalg.solve(a, b)
    residual = torch.linalg.norm(a @ x - b).item()
    print(f"linalg.solve on {x.device}: residual {residual:.2e}")
    if residual > 1e-1:
        print("FAIL: residual too large")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
