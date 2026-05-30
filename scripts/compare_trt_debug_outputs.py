#!/usr/bin/env python3
"""Compare all outputs from two TensorRT engines on a deterministic stereo pair."""

import argparse
import os
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-engine", required=True, help="Reference TensorRT engine")
    parser.add_argument("--test-engine", required=True, help="Engine being debugged")
    parser.add_argument("--hw", type=int, default=256)
    parser.add_argument("--disparity", type=int, default=8)
    parser.add_argument("--top", type=int, default=0, help="Limit printed tensors")
    return parser.parse_args()


def _make_textured_stereo_pair(hw: int, disparity: int):
    y, x = np.mgrid[0:hw, 0:hw]
    base = (
        92.0
        + 42.0 * np.sin(x / 9.0)
        + 31.0 * np.cos(y / 13.0)
        + 18.0 * np.sin((x + y) / 17.0)
    )
    rng = np.random.default_rng(7)
    base += rng.normal(0.0, 4.0, size=(hw, hw))
    left = np.clip(base, 0, 255).astype(np.uint8)
    right = np.empty_like(left)
    right[:, :-disparity] = left[:, disparity:]
    right[:, -disparity:] = left[:, -1:]
    return left, right


def _to_input_tensor(image: np.ndarray):
    rgb = np.repeat(image[None, ..., None], 3, axis=3)
    return torch.from_numpy(rgb).float().cuda().permute(0, 3, 1, 2).contiguous()


def _trt_dtype_to_torch(dtype):
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.bfloat16: torch.bfloat16,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class EngineRunner:
    def __init__(self, path: str):
        import tensorrt as trt

        self.path = os.path.abspath(path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {path}")

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 2:
            raise RuntimeError(f"Expected 2 inputs, got {self.input_names}")

    def run(self, left: torch.Tensor, right: torch.Tensor):
        for name, tensor in zip(self.input_names, [left, right]):
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, tensor.data_ptr())
            outputs[name] = tensor

        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT inference failed: {self.path}")
        torch.cuda.synchronize()
        return {name: tensor.float().cpu().numpy() for name, tensor in outputs.items()}


def _metrics(ref: np.ndarray, test: np.ndarray):
    ref = np.asarray(ref, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    diff = np.abs(test - ref)
    ref_std = float(ref.std())
    test_std = float(test.std())
    if ref.size > 1 and ref_std > 0 and test_std > 0:
        corr = float(np.corrcoef(ref.ravel(), test.ravel())[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": float(diff.mean()),
        "p95": float(np.percentile(diff, 95)),
        "max": float(diff.max()),
        "ref_mean": float(ref.mean()),
        "test_mean": float(test.mean()),
        "ref_std": ref_std,
        "test_std": test_std,
        "corr": corr,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    left_np, right_np = _make_textured_stereo_pair(args.hw, args.disparity)
    left = _to_input_tensor(left_np)
    right = _to_input_tensor(right_np)

    ref_runner = EngineRunner(args.ref_engine)
    test_runner = EngineRunner(args.test_engine)
    if ref_runner.output_names != test_runner.output_names:
        print("Output names differ between engines.", file=sys.stderr)
        print(f"ref={ref_runner.output_names}", file=sys.stderr)
        print(f"test={test_runner.output_names}", file=sys.stderr)
        raise SystemExit(2)

    ref_outputs = ref_runner.run(left, right)
    test_outputs = test_runner.run(left, right)

    rows = []
    for name in ref_runner.output_names:
        if ref_outputs[name].shape != test_outputs[name].shape:
            rows.append((name, None))
            continue
        rows.append((name, _metrics(ref_outputs[name], test_outputs[name])))

    if args.top > 0:
        rows = rows[: args.top]

    print(
        "name\tshape\tmae\tp95\tmax\tcorr\tref_mean\ttest_mean\tref_std\ttest_std"
    )
    for name, metric in rows:
        shape = tuple(ref_outputs[name].shape)
        if metric is None:
            print(f"{name}\t{shape}\tSHAPE_MISMATCH")
            continue
        print(
            f"{name}\t{shape}\t"
            f"{metric['mae']:.6g}\t{metric['p95']:.6g}\t{metric['max']:.6g}\t"
            f"{metric['corr']:.6g}\t"
            f"{metric['ref_mean']:.6g}\t{metric['test_mean']:.6g}\t"
            f"{metric['ref_std']:.6g}\t{metric['test_std']:.6g}"
        )


if __name__ == "__main__":
    main()
