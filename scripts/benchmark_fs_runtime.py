#!/usr/bin/env python3
"""Benchmark FoundationStereo runtimes (.pth or TensorRT .engine/.plan)."""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.live_boxer import FoundationStereoRuntime


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="Path to a FoundationStereo .pth checkpoint or TensorRT .engine/.plan. Repeat to compare multiple runtimes.",
    )
    p.add_argument("--fs_hw", type=int, default=256)
    p.add_argument("--valid_iters", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_inputs(fs_hw: int, seed: int):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(fs_hw, fs_hw), dtype=np.uint8)
    right = rng.integers(0, 256, size=(fs_hw, fs_hw), dtype=np.uint8)
    return left, right


def summarize(times_ms):
    return {
        "min": min(times_ms),
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "p90": np.percentile(times_ms, 90),
        "max": max(times_ms),
        "fps": 1000.0 / statistics.fmean(times_ms),
    }


def benchmark_model(model_path: str, left: np.ndarray, right: np.ndarray, valid_iters: int, warmup: int, iters: int):
    runtime = FoundationStereoRuntime(model_path, valid_iters=valid_iters)
    times_ms = []
    output = None

    for _ in range(warmup):
        output = runtime.infer(left, right)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = runtime.infer(left, right)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = summarize(times_ms)
    stats["shape"] = tuple(output.shape)
    stats["disp_min"] = float(np.min(output))
    stats["disp_mean"] = float(np.mean(output))
    stats["disp_max"] = float(np.max(output))
    return stats


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    left, right = make_inputs(args.fs_hw, args.seed)

    results = []
    for model_path in args.model:
        print(f"==> Benchmarking {model_path}", flush=True)
        stats = benchmark_model(
            model_path=model_path,
            left=left,
            right=right,
            valid_iters=args.valid_iters,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((model_path, stats))

    print()
    print(
        "model,mean_ms,median_ms,p90_ms,min_ms,max_ms,fps,disp_shape,disp_min,disp_mean,disp_max"
    )
    for model_path, s in results:
        print(
            f"{model_path},"
            f"{s['mean']:.3f},{s['median']:.3f},{s['p90']:.3f},"
            f"{s['min']:.3f},{s['max']:.3f},{s['fps']:.3f},"
            f"{s['shape']},{s['disp_min']:.6f},{s['disp_mean']:.6f},{s['disp_max']:.6f}"
        )


if __name__ == "__main__":
    main()
