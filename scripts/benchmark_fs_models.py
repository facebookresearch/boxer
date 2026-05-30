#!/usr/bin/env python3
"""Benchmark FoundationStereo model presets with synthetic rectified stereo input."""

import argparse
import os
import sys
import statistics
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.live_boxer import (
    FS_MODEL_PRESETS,
    FoundationStereoRuntime,
    infer_fs_impl_from_model_path,
    resolve_fs_hw,
    resolve_fs_model_preset,
)

FS_PRESET_VALID_ITERS = {
    "f256": 16,
    "f320": 16,
    "f384": 12,
    "f512": 8,
    "fast512": None,
    "fast512ct": None,
    "fast512fp32": None,
}


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def bench_one(
    name: str, warmup: int, iters: int, valid_iters: int, consistency: bool
) -> dict:
    model_path = resolve_fs_model_preset(name)
    fs_impl = infer_fs_impl_from_model_path(model_path)
    hw = resolve_fs_hw(model_path, fs_impl)
    rng = np.random.default_rng(1234)
    base = rng.integers(0, 256, size=(hw, hw), dtype=np.uint8)
    left = np.ascontiguousarray(base)
    right = np.ascontiguousarray(np.roll(base, shift=4, axis=1))

    runtime = FoundationStereoRuntime(
        model_path,
        valid_iters,
        fs_impl=fs_impl,
        consistency=consistency,
    )

    for _ in range(warmup):
        disp = runtime.infer(left, right)
    sync_cuda()
    if not consistency and not np.isfinite(disp).any():
        raise RuntimeError("model returned all-NaN/non-finite disparity during warmup")

    samples = []
    for _ in range(iters):
        sync_cuda()
        t0 = time.perf_counter()
        disp = runtime.infer(left, right)
        sync_cuda()
        if not consistency and not np.isfinite(disp).any():
            raise RuntimeError("model returned all-NaN/non-finite disparity")
        samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "name": name,
        "impl": fs_impl,
        "hw": hw,
        "iters": FS_PRESET_VALID_ITERS.get(name),
        "path": model_path,
        "consistency": consistency,
        "mean": statistics.fmean(samples),
        "p50": statistics.median(samples),
        "p90": percentile(samples, 90),
        "min": min(samples),
        "max": max(samples),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "preset",
        "impl",
        "hw",
        "iters",
        "mean ms",
        "p50 ms",
        "p90 ms",
        "min ms",
        "max ms",
    ]
    data = [
        [
            row["name"],
            row["impl"],
            str(row["hw"]),
            str(row["iters"]) if row["iters"] is not None else "n/a",
            f"{row['mean']:.1f}",
            f"{row['p50']:.1f}",
            f"{row['p90']:.1f}",
            f"{row['min']:.1f}",
            f"{row['max']:.1f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in data))
        for i in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in data:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(FS_MODEL_PRESETS),
        choices=sorted(FS_MODEL_PRESETS),
        help="FS presets to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--fs_valid_iters", type=int, default=16)
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Run left-right/right-left consistency path.",
    )
    args = parser.parse_args()

    rows = []
    for name in args.models:
        print(f"==> benchmarking {name}", flush=True)
        try:
            rows.append(
                bench_one(
                    name,
                    args.warmup,
                    args.iters,
                    args.fs_valid_iters,
                    args.consistency,
                )
            )
        except Exception as exc:
            print(f"!! {name} failed: {exc}", flush=True)

    if rows:
        print()
        print_table(rows)


if __name__ == "__main__":
    main()
