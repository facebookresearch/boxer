#!/usr/bin/env python3

"""Compare FoundationStereo TensorRT engines against the source checkpoint.

These tests are opt-in because they require CUDA, TensorRT, the external
FoundationStereo checkout, and large local model files.
"""

import os
import time
import unittest

import numpy as np
import torch


DEFAULT_CKPT = "/home/demo/Downloads/model_best_bp2.pth"
DEFAULT_ENGINE = "ckpts/fs_256wh_16it_bf16_all_convtranspose_fp32.engine"


def _enabled() -> bool:
    return os.environ.get("BOXER_RUN_FS_ENGINE_COMPARE") == "1"


def _repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))


def _make_textured_stereo_pair(
    hw: int, disparity: int = 8
) -> tuple[np.ndarray, np.ndarray]:
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


def _cuda_available_with_retry(seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while True:
        if torch.cuda.is_available():
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.25)


@unittest.skipUnless(_enabled(), "set BOXER_RUN_FS_ENGINE_COMPARE=1 to run")
class TestFoundationStereoEngineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _cuda_available_with_retry():
            raise unittest.SkipTest("CUDA is required")
        try:
            import tensorrt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("TensorRT is required") from exc

        cls.ckpt_path = os.environ.get("BOXER_FS_CKPT", DEFAULT_CKPT)
        cls.engine_path = _repo_path(os.environ.get("BOXER_FS_ENGINE", DEFAULT_ENGINE))
        cls.hw = int(os.environ.get("BOXER_FS_COMPARE_HW", "256"))
        cls.valid_iters = int(os.environ.get("BOXER_FS_VALID_ITERS", "16"))

        if not os.path.isfile(cls.ckpt_path):
            raise unittest.SkipTest(f"missing checkpoint: {cls.ckpt_path}")
        if not os.path.isfile(cls.engine_path):
            raise unittest.SkipTest(f"missing TensorRT engine: {cls.engine_path}")

        cfg_path = os.path.join(os.path.dirname(cls.ckpt_path), "cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise unittest.SkipTest(f"missing checkpoint cfg.yaml: {cfg_path}")

        from scripts.live_boxer import FoundationStereoRuntime

        cls.ckpt_runtime = FoundationStereoRuntime(
            cls.ckpt_path,
            valid_iters=cls.valid_iters,
            fs_impl="foundation",
            consistency=False,
        )
        cls.engine_runtime = FoundationStereoRuntime(
            cls.engine_path,
            valid_iters=cls.valid_iters,
            fs_impl="foundation",
            consistency=False,
        )

    @classmethod
    def tearDownClass(cls):
        for name in ("ckpt_runtime", "engine_runtime"):
            runtime = getattr(cls, name, None)
            if runtime is not None:
                del runtime
        torch.cuda.empty_cache()

    def test_engine_matches_checkpoint_on_textured_pair(self):
        left, right = _make_textured_stereo_pair(self.hw)

        ckpt_disp = np.asarray(self.ckpt_runtime.infer(left, right), dtype=np.float32)
        engine_disp = np.asarray(self.engine_runtime.infer(left, right), dtype=np.float32)
        torch.cuda.synchronize()

        self.assertEqual(ckpt_disp.shape, (self.hw, self.hw))
        self.assertEqual(engine_disp.shape, ckpt_disp.shape)
        self.assertTrue(np.isfinite(ckpt_disp).all(), "checkpoint produced NaN/Inf")
        self.assertTrue(np.isfinite(engine_disp).all(), "engine produced NaN/Inf")

        diff = np.abs(engine_disp - ckpt_disp)
        mae = float(diff.mean())
        p95 = float(np.percentile(diff, 95))
        max_abs = float(diff.max())
        ckpt_mean = float(ckpt_disp.mean())
        engine_mean = float(engine_disp.mean())
        ckpt_std = float(ckpt_disp.std())
        engine_std = float(engine_disp.std())
        corr = float(np.corrcoef(ckpt_disp.ravel(), engine_disp.ravel())[0, 1])

        print(
            "FoundationStereo parity: "
            f"mae={mae:.4f} p95={p95:.4f} max={max_abs:.4f} "
            f"corr={corr:.5f} "
            f"ckpt_mean={ckpt_mean:.4f} engine_mean={engine_mean:.4f} "
            f"ckpt_std={ckpt_std:.4f} engine_std={engine_std:.4f} "
            f"ckpt_range=({float(ckpt_disp.min()):.4f},"
            f"{float(ckpt_disp.max()):.4f}) "
            f"engine_range=({float(engine_disp.min()):.4f},"
            f"{float(engine_disp.max()):.4f})",
            flush=True,
        )

        self.assertGreater(ckpt_std, 1e-3)
        self.assertGreater(engine_std, 1e-3)
        self.assertLess(mae, float(os.environ.get("BOXER_FS_MAX_MAE", "0.1")))
        self.assertLess(p95, float(os.environ.get("BOXER_FS_MAX_P95", "0.25")))
        if ckpt_std > 0.1 and engine_std > 0.1:
            self.assertGreater(
                corr, float(os.environ.get("BOXER_FS_MIN_CORR", "0.995"))
            )


if __name__ == "__main__":
    unittest.main()
