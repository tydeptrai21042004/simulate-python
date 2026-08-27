#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_tracking.models import LiteArchitecture, LiteUncertaintyFusionNet, UncertaintyFusionNet


def bench(model: torch.nn.Module, x: torch.Tensor, warmup: int, steps: int) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        started = time.perf_counter()
        for _ in range(steps):
            model(x)
    return 1000.0 * (time.perf_counter() - started) / steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=176)
    p.add_argument("--links", type=int, default=6)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    x = torch.randn(args.links, 6, args.length)
    models = {
        "research_U-Fuse": UncertaintyFusionNet(),
        "dense_lite": LiteUncertaintyFusionNet(arch=LiteArchitecture((8, 12, 16), 12, 24)),
    }
    for name, model in models.items():
        ms = bench(model, x, 50, args.steps)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:18s} params={params:7d} frame_ms={ms:.4f} per_link_ms={ms/args.links:.4f}")


if __name__ == "__main__":
    main()
