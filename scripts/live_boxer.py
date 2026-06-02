"""Live Aria Gen2 streaming + BoxerNet demo with interactive 3D viewer.

moderngl-window viewer with three regions:
  * Left:   ImGui control panel (sliders, toggles).
  * Center: Live RGB frame + OWLv2 2D bounding-box overlays.
  * Right:  Interactive 3D scene (orbit camera) with BoxerNet 3D OBBs and
            a camera frustum marker for the current device pose.

Press 'q' or Esc to quit. Right-drag to orbit, left-drag to pan, scroll to zoom.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import colorsys
import glob
import hashlib
import os
import platform
import sys
import tempfile
import time
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import moderngl
import numpy as np
import torch
import torch.nn.functional as F

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

import utils.imgui_compat as imgui
from boxernet.boxernet import BoxerNet
from utils.gravity import gravity_align_T_world_cam
from owl.owl_wrapper import OwlWrapper
from utils.aria_stream import (
    LiveFsState,
    StreamState,
    connect_with_ip_fallback,
    ensure_aria_tools_on_path,
    make_callbacks,
    make_live_fs_callbacks,
    make_slam_probe_callback,
    save_cached_aria_ip,
)
from utils.demo_utils import CKPT_PATH, DEFAULT_BOXERNET_CKPT
from utils.image import draw_bb3s, put_text, render_bb2, render_depth_patches, torch2cv2
from utils.stereo_depth import (
    FS_MODEL_PRESET_HELP,
    FS_MODEL_PRESETS,
    get_autocast_dtype_for_cuda,
    infer_fs_impl_from_model_path,
    is_tensorrt_engine_path,
    resolve_default_foundation_stereo_model,
    resolve_fast_fs_config,
    resolve_fs_hw,
    resolve_fs_model_preset,
    ensure_projectaria_fs_repo_on_path,
    tensorrt_dtype_to_torch,
)
from utils.track_3d_boxes import BoundingBox3DTracker
from utils.taxonomy import load_text_labels
from utils.tw.camera import CameraTW
from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.tw.pose import PoseTW
from utils.video import make_mp4
from utils.viewer_3d import OrbitViewer, launch_viewer


TAB20 = [
    (0.122, 0.467, 0.706),
    (0.682, 0.780, 0.910),
    (1.000, 0.498, 0.055),
    (1.000, 0.733, 0.471),
    (0.173, 0.627, 0.173),
    (0.596, 0.875, 0.541),
    (0.839, 0.153, 0.157),
    (1.000, 0.596, 0.588),
    (0.580, 0.404, 0.741),
    (0.773, 0.690, 0.835),
    (0.549, 0.337, 0.294),
    (0.769, 0.612, 0.580),
    (0.890, 0.467, 0.761),
    (0.969, 0.714, 0.824),
    (0.498, 0.498, 0.498),
    (0.780, 0.780, 0.780),
    (0.737, 0.741, 0.133),
    (0.859, 0.859, 0.553),
    (0.090, 0.745, 0.812),
    (0.620, 0.855, 0.898),
]
def jet_colors_bgr(scores):
    if len(scores) == 0:
        return []
    vals = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
    u8 = (vals * 255).astype(np.uint8).reshape(1, -1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0]
    return [tuple(int(c) for c in row) for row in bgr]


def jet_colors_rgb_float(scores):
    """Return list of (r,g,b) float in [0,1] for each score (jet colormap)."""
    if len(scores) == 0:
        return []
    vals = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
    u8 = (vals * 255).astype(np.uint8).reshape(1, -1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0].astype(np.float32) / 255.0
    rgb = bgr[:, ::-1]
    return [tuple(float(c) for c in row) for row in rgb]


SALIENT_CLASS_RGB = [
    (0.92, 0.12, 0.16),  # red
    (0.10, 0.45, 0.98),  # blue
    (0.05, 0.72, 0.20),  # green
    (0.98, 0.62, 0.05),  # orange
    (0.58, 0.20, 0.92),  # purple
    (0.00, 0.72, 0.78),  # cyan
    (0.96, 0.10, 0.58),  # magenta
    (0.68, 0.78, 0.04),  # lime
    (0.36, 0.22, 0.98),  # indigo
    (0.98, 0.36, 0.05),  # vermilion
]


def _stable_label_color_rgb(label: str) -> tuple[float, float, float]:
    key = (label or "unknown").strip().lower()
    digest = hashlib.md5(key.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    sat = 0.78 + 0.16 * (digest[1] / 255.0)
    val = 0.78 + 0.14 * (digest[2] / 255.0)
    r, g, b = colorsys.hsv_to_rgb(float(hue), float(sat), float(val))
    return float(r), float(g), float(b)


def obb_class_color_rgb(label: str, sem_id: int) -> tuple[float, float, float]:
    if 0 <= int(sem_id) < len(SALIENT_CLASS_RGB):
        return SALIENT_CLASS_RGB[int(sem_id)]
    return _stable_label_color_rgb(label)


def get_obb_color_arrays(
    labels: list[str],
    sem_ids,
    scores,
    use_class_colors: bool,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    if len(labels) == 0:
        return [], np.zeros((0, 3), dtype=np.float32)

    if use_class_colors:
        rgb = np.asarray(
            [
                obb_class_color_rgb(label, int(sem_id))
                for label, sem_id in zip(labels, sem_ids)
            ],
            dtype=np.float32,
        )
    else:
        rgb = np.asarray(jet_colors_rgb_float(scores.tolist()), dtype=np.float32)

    bgr = [
        tuple(int(np.clip(round(ch * 255.0), 0, 255)) for ch in row[::-1])
        for row in rgb
    ]
    return bgr, rgb.astype(np.float32)


def discover_boxernet_checkpoints(current_path: str = "") -> list[str]:
    pattern = os.path.join(CKPT_PATH, "boxernet_*")
    paths = sorted(
        p
        for p in glob.glob(pattern)
        if os.path.isfile(p) and os.path.basename(p).startswith("boxernet_")
    )
    if current_path:
        current_abs = os.path.abspath(os.path.expanduser(current_path))
        if os.path.exists(current_abs) and current_abs not in paths:
            paths.insert(0, current_abs)
    return paths


def _short_ckpt_name(path: str) -> str:
    return os.path.basename(path.rstrip(os.sep)) or path


class FoundationStereoRuntime:
    def __init__(
        self,
        model_path: str,
        valid_iters: int,
        fs_impl: str = "foundation",
        consistency: bool = False,
        consistency_threshold: float = 1.0,
    ):
        self.model_path = model_path
        self.valid_iters = int(valid_iters)
        self.fs_impl = fs_impl
        self.consistency = bool(consistency)
        self.consistency_threshold = float(consistency_threshold)
        self.kind = "torch"
        self.cfg = None
        self.model = None
        self.supports_consistency_batch2 = True
        self.fast_target_hw = None
        self.fast_input_names = None
        self.fast_output_name = None

        if fs_impl == "fast":
            self._init_fast_foundation_stereo(model_path)
            return

        ensure_projectaria_fs_repo_on_path()

        if is_tensorrt_engine_path(model_path):
            import tensorrt as trt

            with open(model_path, "rb") as f:
                engine_data = f.read()
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.trt_runtime = trt.Runtime(self.trt_logger)
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
            if self.trt_engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
            self.trt_context = self.trt_engine.create_execution_context()
            if self.trt_context is None:
                raise RuntimeError(
                    f"Failed to create TensorRT execution context: {model_path}"
                )
            self.trt_context_aux = self.trt_engine.create_execution_context()
            if self.trt_context_aux is None:
                raise RuntimeError(
                    f"Failed to create auxiliary TensorRT execution context: {model_path}"
                )
            self.trt_stream = torch.cuda.Stream()
            self.trt_overlap_stream0 = torch.cuda.Stream()
            self.trt_overlap_stream1 = torch.cuda.Stream()
            self.trt_input_names = []
            self.trt_output_names = []
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.trt_input_names.append(name)
                else:
                    self.trt_output_names.append(name)
            if len(self.trt_input_names) != 2 or len(self.trt_output_names) != 1:
                raise RuntimeError(
                    "Unexpected TensorRT FoundationStereo IO signature: "
                    f"{self.trt_input_names=} {self.trt_output_names=}"
                )
            self.supports_consistency_batch2 = True
            for name in self.trt_input_names:
                try:
                    _, _, max_shape = self.trt_engine.get_tensor_profile_shape(name, 0)
                    if not max_shape or max_shape[0] < 2:
                        self.supports_consistency_batch2 = False
                except Exception:
                    engine_shape = tuple(self.trt_engine.get_tensor_shape(name))
                    if not engine_shape or engine_shape[0] < 2:
                        self.supports_consistency_batch2 = False
            self.kind = "tensorrt"
            self.model = self.trt_engine
            print(f"==> FoundationStereo TensorRT engine loaded: {model_path}", flush=True)
            return

        from core.foundation_stereo import FoundationStereo
        from omegaconf import OmegaConf

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"FoundationStereo checkpoint not found: {model_path}. "
                "Pass a .pth checkpoint or TensorRT .engine/.plan file."
            )
        cfg_path = os.path.join(os.path.dirname(model_path), "cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"FoundationStereo cfg.yaml not found: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        cfg.valid_iters = self.valid_iters
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "vit_size" not in cfg:
            cls_token = ckpt["model"].get(
                "feature.dino.depth_anything.pretrained.cls_token"
            )
            if cls_token is not None and cls_token.shape[-1] == 1024:
                cfg["vit_size"] = "vitl"
            elif cls_token is not None and cls_token.shape[-1] == 384:
                cfg["vit_size"] = "vits"
            else:
                cfg["vit_size"] = "vitl"

        model = FoundationStereo(cfg)
        model.load_state_dict(ckpt["model"])
        self.cfg = cfg
        self.model = model.cuda().eval()
        print(
            f"==> FoundationStereo loaded: {model_path}, "
            f"epoch={ckpt.get('epoch')}, vit={cfg.vit_size}",
            flush=True,
        )

    def _init_fast_foundation_stereo(self, model_path: str):
        fs_repo = "/home/demo/code/Fast-FoundationStereo"
        if fs_repo not in sys.path:
            sys.path.insert(0, fs_repo)
        self.fast_cfg = resolve_fast_fs_config(model_path)
        image_size = self.fast_cfg["image_size"]
        self.fast_target_hw = int(image_size[0])

        if model_path.endswith(".onnx"):
            import onnxruntime as ort

            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            self.fast_input_names = [
                inp.name for inp in self.ort_session.get_inputs()
            ]
            out_names = [out.name for out in self.ort_session.get_outputs()]
            if len(self.fast_input_names) != 2 or len(out_names) != 1:
                raise RuntimeError(
                    "Unexpected Fast-FoundationStereo ONNX IO signature: "
                    f"{self.fast_input_names=} {out_names=}"
                )
            self.fast_output_name = out_names[0]
            self.kind = "fast_onnx"
            self.supports_consistency_batch2 = False
            print(
                f"==> Fast-FoundationStereo ONNX loaded: {model_path} "
                f"({self.fast_target_hw}x{self.fast_target_hw})",
                flush=True,
            )
            return

        if is_tensorrt_engine_path(model_path):
            import tensorrt as trt

            with open(model_path, "rb") as f:
                engine_data = f.read()
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.trt_runtime = trt.Runtime(self.trt_logger)
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
            if self.trt_engine is None:
                raise RuntimeError(
                    f"Failed to deserialize Fast-FoundationStereo engine: {model_path}"
                )
            self.trt_context = self.trt_engine.create_execution_context()
            if self.trt_context is None:
                raise RuntimeError(
                    f"Failed to create Fast-FoundationStereo TRT context: {model_path}"
                )
            self.trt_stream = torch.cuda.Stream()
            self.fast_input_names = []
            self.trt_output_names = []
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.fast_input_names.append(name)
                else:
                    self.trt_output_names.append(name)
            if len(self.fast_input_names) != 2 or len(self.trt_output_names) != 1:
                raise RuntimeError(
                    "Unexpected Fast-FoundationStereo TRT IO signature: "
                    f"{self.fast_input_names=} {self.trt_output_names=}"
                )
            self.fast_output_name = self.trt_output_names[0]
            self.kind = "fast_tensorrt"
            self.supports_consistency_batch2 = False
            self.model = self.trt_engine
            print(
                f"==> Fast-FoundationStereo TensorRT engine loaded: {model_path} "
                f"({self.fast_target_hw}x{self.fast_target_hw})",
                flush=True,
            )
            return

        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.args.valid_iters = self.valid_iters
        self.model = model.cuda().eval()
        self.kind = "fast_torch"
        self.supports_consistency_batch2 = True
        print(
            f"==> Fast-FoundationStereo loaded: {model_path} "
            f"({self.fast_target_hw}x{self.fast_target_hw})",
            flush=True,
        )

    def _prepare_rectified_inputs(
        self, left_rect_batch: np.ndarray, right_rect_batch: np.ndarray
    ):
        from core.utils.utils import InputPadder

        if left_rect_batch.ndim != 3 or right_rect_batch.ndim != 3:
            raise ValueError(
                "Expected FoundationStereo batch inputs shaped [B, H, W], "
                f"got {left_rect_batch.shape=} {right_rect_batch.shape=}"
            )
        left_rgb = np.repeat(left_rect_batch[..., None], 3, axis=3)
        right_rgb = np.repeat(right_rect_batch[..., None], 3, axis=3)
        left_t = (
            torch.from_numpy(left_rgb).float().cuda().permute(0, 3, 1, 2).contiguous()
        )
        right_t = (
            torch.from_numpy(right_rgb)
            .float()
            .cuda()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_p, right_p = padder.pad(left_t, right_t)
        return left_p.contiguous(), right_p.contiguous(), padder

    def _prepare_fast_inputs(
        self, left_rect_batch: np.ndarray, right_rect_batch: np.ndarray
    ):
        bsz = int(left_rect_batch.shape[0])
        hw = int(self.fast_target_hw)
        left_rgb = np.repeat(left_rect_batch[..., None], 3, axis=3)
        right_rgb = np.repeat(right_rect_batch[..., None], 3, axis=3)
        if left_rgb.shape[1] != hw or left_rgb.shape[2] != hw:
            left_rgb = np.stack(
                [
                    cv2.resize(im, (hw, hw), interpolation=cv2.INTER_LINEAR)
                    for im in left_rgb
                ],
                axis=0,
            )
            right_rgb = np.stack(
                [
                    cv2.resize(im, (hw, hw), interpolation=cv2.INTER_LINEAR)
                    for im in right_rgb
                ],
                axis=0,
            )
        left_t = (
            torch.from_numpy(left_rgb).float().cuda().permute(0, 3, 1, 2).contiguous()
        )
        right_t = (
            torch.from_numpy(right_rgb).float().cuda().permute(0, 3, 1, 2).contiguous()
        )
        if self.kind == "fast_torch":
            return left_t, right_t, bsz

        mean = left_t.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = left_t.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        left_t = (left_t / 255.0 - mean) / std
        right_t = (right_t / 255.0 - mean) / std
        return left_t.contiguous(), right_t.contiguous(), bsz

    def _run_trt_context(self, context, left_p: torch.Tensor, right_p: torch.Tensor, stream):
        inputs = [left_p, right_p]
        for name, tensor in zip(self.trt_input_names, inputs):
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self.trt_output_names:
            shape = tuple(context.get_tensor_shape(name))
            dtype = tensorrt_dtype_to_torch(self.trt_engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())

        if stream is None:
            stream_ptr = torch.cuda.current_stream().cuda_stream
        else:
            stream.wait_stream(torch.cuda.current_stream())
            stream_ptr = stream.cuda_stream
        ok = context.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TensorRT FoundationStereo inference failed")
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return outputs[self.trt_output_names[0]].float()

    def _infer_batch(
        self, left_rect_batch: np.ndarray, right_rect_batch: np.ndarray
    ) -> np.ndarray:
        if self.fs_impl == "fast":
            return self._infer_batch_fast(left_rect_batch, right_rect_batch)

        left_p, right_p, padder = self._prepare_rectified_inputs(
            left_rect_batch, right_rect_batch
        )

        if self.kind == "tensorrt":
            disp_t = self._run_trt_context(
                self.trt_context, left_p, right_p, self.trt_stream
            )
            return padder.unpad(disp_t).cpu().numpy()

        autocast_dtype = get_autocast_dtype_for_cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            disp = self.model.forward(
                left_p, right_p, iters=self.cfg.valid_iters, test_mode=True
            )
        return padder.unpad(disp.float()).cpu().numpy()

    def _infer_batch_fast(
        self, left_rect_batch: np.ndarray, right_rect_batch: np.ndarray
    ) -> np.ndarray:
        left_t, right_t, _ = self._prepare_fast_inputs(left_rect_batch, right_rect_batch)

        if self.kind == "fast_tensorrt":
            for name, tensor in zip(self.fast_input_names, [left_t, right_t]):
                self.trt_context.set_input_shape(name, tuple(tensor.shape))
                self.trt_context.set_tensor_address(name, tensor.data_ptr())
            out_name = self.fast_output_name
            out_shape = tuple(self.trt_context.get_tensor_shape(out_name))
            out_dtype = tensorrt_dtype_to_torch(
                self.trt_engine.get_tensor_dtype(out_name)
            )
            disp_t = torch.empty(out_shape, dtype=out_dtype, device="cuda")
            self.trt_context.set_tensor_address(out_name, disp_t.data_ptr())
            self.trt_stream.wait_stream(torch.cuda.current_stream())
            ok = self.trt_context.execute_async_v3(self.trt_stream.cuda_stream)
            if not ok:
                raise RuntimeError("Fast-FoundationStereo TensorRT inference failed")
            torch.cuda.current_stream().wait_stream(self.trt_stream)
            return disp_t.float().cpu().numpy()

        if self.kind == "fast_onnx":
            inputs = {
                self.fast_input_names[0]: left_t.cpu().numpy().astype(np.float32),
                self.fast_input_names[1]: right_t.cpu().numpy().astype(np.float32),
            }
            disp = self.ort_session.run([self.fast_output_name], inputs)[0]
            return np.asarray(disp, dtype=np.float32)

        autocast_dtype = get_autocast_dtype_for_cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            disp = self.model.forward(
                left_t, right_t, iters=self.valid_iters, test_mode=True
            )
        return np.asarray(disp.float().cpu().numpy(), dtype=np.float32)

    def _infer_one_pass(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        disp = self._infer_batch(
            left_rect[None, ...],
            right_rect[None, ...],
        )
        return np.asarray(disp[0]).squeeze()

    def _infer_two_passes_overlap_trt(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        right_flipped: np.ndarray,
        left_flipped: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left0_p, right0_p, padder0 = self._prepare_rectified_inputs(
            left_rect[None, ...], right_rect[None, ...]
        )
        left1_p, right1_p, padder1 = self._prepare_rectified_inputs(
            right_flipped[None, ...], left_flipped[None, ...]
        )
        disp0_t = self._run_trt_context(
            self.trt_context, left0_p, right0_p, self.trt_overlap_stream0
        )
        disp1_t = self._run_trt_context(
            self.trt_context_aux, left1_p, right1_p, self.trt_overlap_stream1
        )
        self.trt_overlap_stream0.synchronize()
        self.trt_overlap_stream1.synchronize()
        disp_lr = np.asarray(padder0.unpad(disp0_t).cpu().numpy()[0]).squeeze()
        disp_rl = np.asarray(padder1.unpad(disp1_t).cpu().numpy()[0]).squeeze()
        return disp_lr, disp_rl

    def infer(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        if not self.consistency:
            return self._infer_one_pass(left_rect, right_rect)

        right_flipped = np.ascontiguousarray(right_rect[:, ::-1])
        left_flipped = np.ascontiguousarray(left_rect[:, ::-1])
        if self.supports_consistency_batch2:
            disp_pair = self._infer_batch(
                np.stack([left_rect, right_flipped], axis=0),
                np.stack([right_rect, left_flipped], axis=0),
            )
            disp_lr = np.asarray(disp_pair[0]).squeeze()
            disp_rl = np.asarray(disp_pair[1]).squeeze()
        elif self.kind == "tensorrt":
            disp_lr, disp_rl = self._infer_two_passes_overlap_trt(
                left_rect, right_rect, right_flipped, left_flipped
            )
        else:
            disp_lr = self._infer_one_pass(left_rect, right_rect)
            disp_rl = self._infer_one_pass(right_flipped, left_flipped)
        disp_rl = np.ascontiguousarray(disp_rl[:, ::-1])

        h, w = disp_lr.shape
        x_coords = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        x_in_right_f = x_coords - disp_lr
        out_of_bounds = (x_in_right_f < 0) | (x_in_right_f > w - 1)
        rows = np.arange(h)[:, None].repeat(w, axis=1)
        x0 = np.floor(np.clip(x_in_right_f, 0, w - 1)).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        alpha = np.clip(x_in_right_f - x0.astype(np.float32), 0.0, 1.0)
        disp_rl_at_match = (
            (1.0 - alpha) * disp_rl[rows, x0] + alpha * disp_rl[rows, x1]
        )
        consistent = (
            np.abs(disp_lr - disp_rl_at_match) < self.consistency_threshold
        ) & ~out_of_bounds
        valid_ratio = float(np.count_nonzero(consistent)) / float(consistent.size)
        if valid_ratio < 0.001:
            return disp_lr.astype(np.float32, copy=False)

        disp_out = disp_lr.astype(np.float32, copy=True)
        disp_out[~consistent] = np.nan
        return disp_out


def run_live_foundation_fs_loop(args, fs_state: LiveFsState):
    from projectaria_tools.core import calibration
    from projectaria_tools.core.image import InterpolationMethod
    from projectaria_tools.core.sophus import SE3

    fs_repo = "/home/demo/code/projectaria_gen2_depth_from_stereo"
    foundation_path = os.path.join(fs_repo, "FoundationStereo")
    if fs_repo not in sys.path:
        sys.path.insert(0, fs_repo)

    from stereo_utils import (
        create_scanline_rectified_cameras,
        disparity_to_depth,
        rectify_stereo_pair,
    )

    fs_runtime = None
    if not args.fs_dry_run:
        fs_runtime = FoundationStereoRuntime(
            args.fs_ckpt,
            args.fs_valid_iters,
            fs_impl=args.fs_impl,
            consistency=args.consistency,
            consistency_threshold=args.consistency_threshold,
        )

    def make_linear_calib(source_calib):
        params = source_calib.get_projection_params()
        src_w, src_h = source_calib.get_image_size()
        scale = min(args.fs_hw / float(src_w), args.fs_hw / float(src_h))
        focal = float(params[0]) * scale * 1.25
        linear_params = np.array(
            [focal, focal, args.fs_hw / 2.0, args.fs_hw / 2.0]
        )
        return calibration.CameraCalibration(
            source_calib.get_label() + f"-linear-{args.fs_hw}",
            calibration.CameraModelType.LINEAR,
            linear_params,
            SE3(),
            args.fs_hw,
            args.fs_hw,
            None,
            source_calib.get_max_solid_angle(),
            source_calib.get_serial_number(),
        )

    def infer(left_rect, right_rect):
        return fs_runtime.infer(left_rect, right_rect)

    processed = 0
    last_pair_ts = None
    t_start = time.time()
    print("==> Waiting for front SLAM fs pairs ...", flush=True)
    while True:
        (
            left_frame,
            right_frame,
            left_calib,
            right_calib,
            _T_world_device,
        ) = fs_state.snapshot()
        if (
            left_frame is None
            or right_frame is None
            or left_calib is None
            or right_calib is None
        ):
            time.sleep(0.005)
            continue
        left_img, left_ts = left_frame
        right_img, right_ts = right_frame
        pair_ts = max(left_ts, right_ts)
        if pair_ts == last_pair_ts:
            time.sleep(0.002)
            continue
        delta_ms = abs(left_ts - right_ts) / 1e6
        if delta_ms > 2.0:
            time.sleep(0.002)
            continue

        T_left_device = left_calib.get_transform_device_camera().inverse()
        T_right_device = right_calib.get_transform_device_camera().inverse()
        T_left_right = T_left_device @ T_right_device.inverse()
        R_left_rect, R_right_rect = create_scanline_rectified_cameras(
            T_left_device, T_right_device
        )
        linear = make_linear_calib(left_calib)
        left_rect, right_rect = rectify_stereo_pair(
            left_img,
            right_img,
            left_calib,
            right_calib,
            linear,
            linear,
            R_left_rect,
            R_right_rect,
            interpolation=InterpolationMethod.BILINEAR,
        )

        t0 = time.time()
        median_depth = float("nan")
        disparity = None
        if fs_runtime is not None:
            disparity = infer(left_rect, right_rect)
            baseline = float(np.linalg.norm(T_left_right.translation()))
            focal = float(linear.get_projection_params()[0])
            depth = disparity_to_depth(disparity, baseline, focal)
            valid = depth[np.isfinite(depth)]
            if valid.size:
                median_depth = float(np.nanmedian(valid))

        processed += 1
        last_pair_ts = pair_ts
        fps = processed / max(time.time() - t_start, 1e-6)
        print(
            f"==> fs[{processed}] hw={args.fs_hw} "
            f"delta={delta_ms:.3f}ms infer={(time.time() - t0) * 1000:.1f}ms "
            f"fps={fps:.2f} median_depth={median_depth:.3f}m",
            flush=True,
        )

        if args.fs_display:
            left_vis = left_rect
            if left_vis.ndim == 2:
                left_vis = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2BGR)
            else:
                left_vis = cv2.cvtColor(left_vis, cv2.COLOR_RGB2BGR)
            if disparity is None:
                disp_vis = np.zeros_like(left_vis)
            else:
                disp_u8 = cv2.normalize(
                    disparity, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                disp_vis = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
            vis = np.concatenate([left_vis, disp_vis], axis=1)
            cv2.putText(
                vis,
                f"{args.fs_hw}px  {median_depth:.2f}m",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Live FoundationStereo", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return

        if args.fs_max_frames > 0 and processed >= args.fs_max_frames:
            return


def build_cam(intrinsics, T_camera_rig, calib_image_size, image_size, target_hw):
    calib_w, calib_h = calib_image_size
    image_w, image_h = image_size
    valid_radius = float(np.sqrt(calib_w * calib_w + calib_h * calib_h) / 2.0)
    cam = CameraTW.from_surreal(
        width=calib_w,
        height=calib_h,
        type_str="Fisheye624",
        params=torch.tensor(intrinsics, dtype=torch.float32),
        T_camera_rig=T_camera_rig,
        valid_radius=torch.tensor([valid_radius], dtype=torch.float32),
    ).float()
    cam = cam.scale_to_size((image_w, image_h))
    return cam.scale_to_size((target_hw, target_hw))


def build_fisheye_cam_at_image_size(
    intrinsics, T_camera_rig, calib_image_size, image_size
):
    calib_w, calib_h = calib_image_size
    image_w, image_h = image_size
    valid_radius = float(np.sqrt(calib_w * calib_w + calib_h * calib_h) / 2.0)
    cam = CameraTW.from_surreal(
        width=calib_w,
        height=calib_h,
        type_str="Fisheye624",
        params=torch.tensor(intrinsics, dtype=torch.float32),
        T_camera_rig=T_camera_rig,
        valid_radius=torch.tensor([valid_radius], dtype=torch.float32),
    ).float()
    return cam.scale_to_size((image_w, image_h))


def apply_live_rotation(
    img_torch: torch.Tensor, cam: CameraTW, mode: str
) -> tuple[torch.Tensor, CameraTW, torch.Tensor, str]:
    """Return the live Gen2 rotation policy matching AriaLoader(unrotate=True).

    AriaLoader treats Nebula/oatmeal/Aria Gen2 frames as already upright:
      rotated starts False, unrotate=True is a no-op, and Boxer sees rotated0=False.
    The 90-degree CW unrotate path is only for Gen1 VRS frames.
    """
    if mode == "none":
        return (
            img_torch,
            cam,
            torch.tensor([False]),
            "Gen2/oatmeal: no image rotation, no cam.rotate_90_cw",
        )
    if mode == "cw":
        return (
            torch.rot90(img_torch, k=3, dims=(2, 3)),
            cam.rotate_90_cw(),
            torch.tensor([False]),
            "forced 90deg CW image + cam.rotate_90_cw, then rotated0=False",
        )
    if mode == "cam_cw":
        return (
            img_torch,
            cam.rotate_90_cw(),
            torch.tensor([False]),
            "forced cam.rotate_90_cw only; image stays in live display orientation",
        )
    if mode == "ccw":
        return (
            torch.rot90(img_torch, k=1, dims=(2, 3)),
            cam.rotate_90_ccw(),
            torch.tensor([False]),
            "forced 90deg CCW image + cam.rotate_90_ccw, then rotated0=False",
        )
    if mode == "cam_ccw":
        return (
            img_torch,
            cam.rotate_90_ccw(),
            torch.tensor([False]),
            "forced cam.rotate_90_ccw only; image stays in live display orientation",
        )
    raise ValueError(f"Unknown live rotation mode: {mode}")


def live_resized_rgb_to_bgr(arr_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cw":
        arr_rgb = np.rot90(arr_rgb, k=3)
    elif mode == "ccw":
        arr_rgb = np.rot90(arr_rgb, k=1)
    return cv2.cvtColor(np.ascontiguousarray(arr_rgb), cv2.COLOR_RGB2BGR)


def rectify_rgb_for_owl(
    img_torch: torch.Tensor,
    fisheye_cam: CameraTW,
    target_hw: int,
    pinhole_fxy: float | None = None,
) -> tuple[torch.Tensor, CameraTW]:
    W_src = int(round(float(fisheye_cam.size[0].item())))
    H_src = int(round(float(fisheye_cam.size[1].item())))
    W = int(target_hw)
    H = int(target_hw)
    if pinhole_fxy is None:
        pinhole_fxy = float(fisheye_cam.f[0].item()) * 1.2
    w_ratio = float(W) / float(W_src)
    h_ratio = float(H) / float(H_src)
    fx = float(pinhole_fxy) * w_ratio
    fy = float(pinhole_fxy) * h_ratio
    cx = float(fisheye_cam.c[0].item()) * w_ratio
    cy = float(fisheye_cam.c[1].item()) * h_ratio
    pinhole_cam = CameraTW.from_surreal(
        width=W,
        height=H,
        type_str="pinhole",
        params=torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
        T_camera_rig=fisheye_cam.T_camera_rig,
    ).float()
    device = img_torch.device
    xx, yy = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="ij",
    )
    target = torch.stack([xx, yy], dim=-1).view(-1, 2).float()[None]
    rays, _ = pinhole_cam.unproject(target)
    source, _ = fisheye_cam.project(rays)
    source = source[0]
    source[..., 0] = (source[..., 0] / max(W_src - 1, 1)) * 2.0 - 1.0
    source[..., 1] = (source[..., 1] / max(H_src - 1, 1)) * 2.0 - 1.0
    source = source.view(1, W, H, 2).permute(0, 2, 1, 3).float()
    rectified = F.grid_sample(
        img_torch,
        source,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return rectified, pinhole_cam


def map_bb2_xxyy_between_cams(
    bb2_xxyy: torch.Tensor,
    source_cam: CameraTW,
    target_cam: CameraTW,
) -> torch.Tensor:
    if bb2_xxyy.numel() == 0:
        return bb2_xxyy.clone()

    tl = bb2_xxyy[:, [0, 2]]
    bl = bb2_xxyy[:, [0, 3]]
    br = bb2_xxyy[:, [1, 3]]
    tr = bb2_xxyy[:, [1, 2]]
    corners = torch.stack([tl, bl, br, tr], dim=1).float()
    rays, valid_src = source_cam.unproject(corners)
    proj, valid_tgt = target_cam.project(rays)
    valid = valid_src & valid_tgt

    x = proj[..., 0]
    y = proj[..., 1]
    width = float(target_cam.size.reshape(-1, 2)[0, 0].item())
    height = float(target_cam.size.reshape(-1, 2)[0, 1].item())
    x = torch.clamp(x, min=0.0, max=width - 1.0)
    y = torch.clamp(y, min=0.0, max=height - 1.0)

    invalid_fill_xmin = torch.full_like(x, width - 1.0)
    invalid_fill_xmax = torch.zeros_like(x)
    invalid_fill_ymin = torch.full_like(y, height - 1.0)
    invalid_fill_ymax = torch.zeros_like(y)

    xmin = torch.min(torch.where(valid, x, invalid_fill_xmin), dim=1).values
    xmax = torch.max(torch.where(valid, x, invalid_fill_xmax), dim=1).values
    ymin = torch.min(torch.where(valid, y, invalid_fill_ymin), dim=1).values
    ymax = torch.max(torch.where(valid, y, invalid_fill_ymax), dim=1).values

    out = torch.stack([xmin, xmax, ymin, ymax], dim=-1)
    any_valid = valid.any(dim=1)
    non_empty = (xmax > xmin) & (ymax > ymin)
    keep = any_valid & non_empty
    return out[keep]


def _fmt_tensor(x, precision=4):
    return np.array2string(
        x.detach().cpu().float().numpy(), precision=precision, suppress_small=False
    )


def log_geometry_debug(
    arr_rgb,
    intr,
    T_wr,
    T_cr,
    calib_image_size,
    cam,
    target_hw,
    rotated,
    rotation_policy,
) -> None:
    image_h, image_w = arr_rgb.shape[:2]
    calib_w, calib_h = calib_image_size
    print("==> Geometry debug", flush=True)
    print(
        f"    stream frame HxW={image_h}x{image_w}, "
        f"calib WxH={calib_w}x{calib_h}, Boxer HW={target_hw}",
        flush=True,
    )
    print(
        f"    scale calib->frame: sx={image_w / calib_w:.6f}, "
        f"sy={image_h / calib_h:.6f}; frame->Boxer: "
        f"sx={target_hw / image_w:.6f}, sy={target_hw / image_h:.6f}",
        flush=True,
    )
    print(
        "    raw Fisheye624 params "
        f"f={intr[0]:.4f}, cx={intr[1]:.4f}, cy={intr[2]:.4f}, "
        f"dist={np.array2string(np.asarray(intr[3:], dtype=np.float32), precision=4)}",
        flush=True,
    )
    print(
        f"    Boxer cam size={_fmt_tensor(cam.size)}, f={_fmt_tensor(cam.f)}, "
        f"c={_fmt_tensor(cam.c)}, valid_radius={_fmt_tensor(cam.valid_radius)}",
        flush=True,
    )
    print(
        f"    AriaLoader-compatible rotation: {rotation_policy}; "
        f"rotated0={bool(rotated.item())}",
        flush=True,
    )
    print(
        f"    raw T_camera_rig t={_fmt_tensor(T_cr.t)}, rpy_deg={_fmt_tensor(T_cr.to_euler(rad=False, silent=True))}",
        flush=True,
    )
    print(
        f"    Boxer cam.T_camera_rig t={_fmt_tensor(cam.T_camera_rig.t)}, "
        f"rpy_deg={_fmt_tensor(cam.T_camera_rig.to_euler(rad=False, silent=True))}",
        flush=True,
    )
    print(
        f"    T_world_rig t={_fmt_tensor(T_wr.t)}, rpy_deg={_fmt_tensor(T_wr.to_euler(rad=False, silent=True))}",
        flush=True,
    )
    T_wc = T_wr @ cam.T_camera_rig.inverse()
    T_wv = gravity_align_T_world_cam(T_wc.unsqueeze(0), z_grav=True)
    print(
        f"    T_world_cam t={_fmt_tensor(T_wc.t)}, rpy_deg={_fmt_tensor(T_wc.to_euler(rad=False, silent=True))}",
        flush=True,
    )
    print(
        f"    T_world_voxel rpy_deg={_fmt_tensor(T_wv.to_euler(rad=False, silent=True))}",
        flush=True,
    )


def run_inference(
    state: StreamState,
    owl: OwlWrapper,
    boxernet: BoxerNet,
    text_labels: list,
    sem_name_to_id: dict,
    sem_id_to_name: dict,
    HW: int,
    detector_hw: int,
    thresh3d: float,
    bb2_line_width: int,
    bb3_line_width: int,
    dev: str,
    pdtype,
    debug_geometry: bool,
    live_rotation: str,
    enable_owl: bool,
    enable_boxer: bool,
    bb3_use_class_colors: bool,
    boxer_sdp_w: torch.Tensor,
    rectify_rgb_for_owl_boxes: bool,
    bench: bool = False,
    render_cpu_overlays: bool = True,
):
    """Run one OWL+BoxerNet pass on the latest frame.

    Returns dict with keys: viz_2d_bgr, obb_pr_w, T_wr, cam, n_2d, n_3d, ts_ns
    Or None if no frame is available yet.
    """
    bench_total_t0 = time.perf_counter()
    bench_last = bench_total_t0
    bench_times = {}

    def bench_mark(name: str, sync_cuda: bool = False) -> None:
        nonlocal bench_last
        if not bench:
            return
        if sync_cuda and dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        bench_times[name] = (now - bench_last) * 1000.0
        bench_last = now

    frame, T_wr, intr, T_cr, csize = state.snapshot()
    bench_mark("snapshot")
    t_after_snapshot = time.perf_counter()
    if frame is None or T_wr is None or intr is None or T_cr is None:
        return None

    arr_rgb, ts_ns = frame
    image_h, image_w = arr_rgb.shape[:2]
    arr_resized = cv2.resize(arr_rgb, (HW, HW), interpolation=cv2.INTER_LINEAR)
    img_torch = torch.from_numpy(arr_resized).permute(2, 0, 1)[None].float() / 255.0
    cam = build_cam(intr, T_cr, csize, (image_w, image_h), HW)
    img_torch, cam, rotated0, rotation_policy = apply_live_rotation(
        img_torch, cam, live_rotation
    )
    bench_mark("preprocess")
    t_after_preprocess = time.perf_counter()
    owl_img_torch = img_torch
    owl_cam = cam
    owl_rotated0 = rotated0
    if rectify_rgb_for_owl_boxes:
        owl_src_torch = (
            torch.from_numpy(arr_rgb).permute(2, 0, 1)[None].float() / 255.0
        )
        owl_fisheye_cam = build_fisheye_cam_at_image_size(
            intr, T_cr, csize, (image_w, image_h)
        )
        owl_img_torch, owl_cam = rectify_rgb_for_owl(
            owl_src_torch, owl_fisheye_cam, HW
        )
        owl_img_torch, owl_cam, owl_rotated0, _ = apply_live_rotation(
            owl_img_torch, owl_cam, live_rotation
        )
        if not state.rectify_debug_logged:
            diff = float((owl_img_torch - img_torch).abs().mean().item())
            print(
                f"==> rectify_debug live diff_mean={diff:.6f} "
                f"orig={tuple(owl_src_torch.shape)} rect={tuple(owl_img_torch.shape)}",
                flush=True,
            )
            try:
                debug_dir = os.path.join(REPO_ROOT, "tmp", "rectify_debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(debug_dir, "orig_rgb.png"),
                    cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(debug_dir, "owl_rectified.png"),
                    torch2cv2(
                        owl_img_torch,
                        rotate=bool(owl_rotated0.item()),
                        ensure_rgb=True,
                    ),
                )
                print(
                    f"==> rectify_debug wrote {debug_dir}/orig_rgb.png and owl_rectified.png",
                    flush=True,
                )
            except Exception as exc:
                print(f"==> rectify_debug write failed: {exc}", flush=True)
            with state.lock:
                state.rectify_debug_logged = True
    bench_mark("rectify")
    t_after_rectify = time.perf_counter()
    sdp_w = boxer_sdp_w if boxer_sdp_w is not None else torch.zeros(0, 3)
    use_cuda_timing = dev == "cuda" and torch.cuda.is_available()

    debug_this_frame = debug_geometry and not state.debug_geometry_logged
    if debug_this_frame:
        log_geometry_debug(
            arr_rgb,
            intr,
            T_wr,
            T_cr,
            csize,
            cam,
            HW,
            rotated0,
            rotation_policy,
        )

    t_before_owl_sync = time.perf_counter()
    if use_cuda_timing:
        torch.cuda.synchronize()
    t_after_owl_sync = time.perf_counter()
    t_start = time.perf_counter()
    if enable_owl:
        bb2d, scores2d, label_ints, _ = owl.forward(
            owl_img_torch * 255.0,
            rotated=bool(owl_rotated0.item()),
            resize_to_HW=(detector_hw, detector_hw),
        )
        bench_mark("owl_forward", sync_cuda=True)
        bb2d_display = bb2d.clone()
        scores2d_display = scores2d.clone()
        label_ints_display = list(label_ints)
        if rectify_rgb_for_owl_boxes and bb2d.shape[0] > 0:
            mapped_boxes = []
            mapped_scores = []
            mapped_labels = []
            for i in range(bb2d.shape[0]):
                mapped = map_bb2_xxyy_between_cams(bb2d[i : i + 1], owl_cam, cam)
                if mapped.shape[0] == 0:
                    continue
                mapped_boxes.append(mapped[0])
                mapped_scores.append(scores2d[i])
                mapped_labels.append(label_ints[i])
            if mapped_boxes:
                bb2d = torch.stack(mapped_boxes, dim=0)
                scores2d = torch.stack(mapped_scores, dim=0)
                label_ints = mapped_labels
            else:
                bb2d = torch.zeros((0, 4), dtype=torch.float32)
                scores2d = torch.zeros((0,), dtype=torch.float32)
                label_ints = []
    else:
        bb2d = torch.zeros((0, 4), dtype=torch.float32)
        scores2d = torch.zeros((0,), dtype=torch.float32)
        label_ints = []
        bb2d_display = bb2d
        scores2d_display = scores2d
        label_ints_display = []
        bench_mark("owl_forward")
    bench_mark("owl_post")
    if use_cuda_timing:
        torch.cuda.synchronize()
    t_owl_model_done = time.perf_counter()
    if debug_this_frame:
        if bb2d.shape[0] > 0:
            print(
                f"    OWL bb2d handed to Boxer: n={bb2d.shape[0]}, "
                f"min={_fmt_tensor(bb2d.min(dim=0).values)}, "
                f"max={_fmt_tensor(bb2d.max(dim=0).values)}, "
                f"top_score={float(scores2d.max()):.4f}",
                flush=True,
            )
        else:
            print("    OWL bb2d handed to Boxer: n=0", flush=True)
        print(f"    sdp_w handed to Boxer: shape={tuple(sdp_w.shape)}", flush=True)
        with state.lock:
            state.debug_geometry_logged = True
    labels2d = [text_labels[i] for i in label_ints]
    labels2d_display = [text_labels[i] for i in label_ints_display]
    rotated_bool = bool(rotated0.item())

    # Center panel: RGB with 2D and projected 3D overlays.
    viz_rgb = live_resized_rgb_to_bgr(arr_resized, live_rotation)
    if render_cpu_overlays:
        viz_2d = torch2cv2(
            owl_img_torch, rotate=bool(owl_rotated0.item()), ensure_rgb=True
        )
    else:
        viz_2d = viz_rgb
    bb2_texts = [f"{l[:10]} {s:.2f}" for s, l in zip(scores2d, labels2d)]
    bb2_colors = jet_colors_bgr(scores2d)
    bb2_colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in bb2_colors]
    if render_cpu_overlays and bb2d_display.shape[0] > 0:
        bb2_texts_display = [
            f"{l[:10]} {s:.2f}" for s, l in zip(scores2d_display, labels2d_display)
        ]
        bb2_colors_display = jet_colors_bgr(scores2d_display)
        viz_2d = render_bb2(
            viz_2d,
            bb2d_display,
            scale=float(bb2_line_width),
            rotated=bool(owl_rotated0.item()),
            texts=bb2_texts_display,
            clr=bb2_colors_display,
        )
    owl_title = (
        f"OWLv2 rectified pinhole {detector_hw}x{detector_hw}"
        if rectify_rgb_for_owl_boxes
        else f"OWLv2 {detector_hw}x{detector_hw}"
    )
    if render_cpu_overlays:
        put_text(viz_2d, owl_title, scale=0.6, line=0)
        put_text(viz_2d, f"t={ts_ns / 1e9:.3f}s", scale=0.5, line=2)
    viz_3d = viz_rgb.copy() if render_cpu_overlays else viz_rgb
    bench_mark("viz_2d")
    t_viz_2d_done = time.perf_counter()

    obb_pr_w = ObbTW(torch.zeros(0, 165))
    scores3d = torch.zeros(0)
    labels3d: list = []
    bb3_rgb_colors = np.zeros((0, 3), dtype=np.float32)
    bb3_overlay_rgb_colors = np.zeros((0, 3), dtype=np.float32)
    sdp_patch = None
    sdp_patch_valid = 0
    sdp_patch_median = float("nan")
    n_2d = bb2d.shape[0]
    n_3d = 0
    t_boxer_start = time.perf_counter()
    t_boxer_done = t_boxer_start

    if enable_boxer and n_2d > 0:
        if use_cuda_timing:
            torch.cuda.synchronize()
            t_boxer_start = time.perf_counter()
        datum = {
            "img0": img_torch,
            "cam0": cam,
            "T_world_rig0": T_wr,
            "rotated0": rotated0,
            "sdp_w": sdp_w,
            "bb2d": bb2d,
        }
        if dev == "mps":
            out = boxernet.forward(datum)
        else:
            with torch.autocast(device_type=dev, dtype=pdtype):
                out = boxernet.forward(datum)
        bench_mark("boxer_forward", sync_cuda=True)
        obb_pr_w = out["obbs_pr_w"].cpu()[0]
        if "sdp_patch0" in out:
            sdp_patch = out["sdp_patch0"].detach()
            sdp_valid = sdp_patch > 0.0
            sdp_patch_valid = int(sdp_valid.sum().item())
            if sdp_patch_valid > 0:
                sdp_patch_median = float(torch.median(sdp_patch[sdp_valid]).item())
        else:
            sdp_patch = None

        sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
        for i, lab in enumerate(labels2d):
            if lab not in sem_name_to_id:
                nid = len(sem_name_to_id)
                sem_name_to_id[lab] = nid
                sem_id_to_name[nid] = lab
            sem_ids[i] = sem_name_to_id[lab]
        obb_pr_w.set_sem_id(sem_ids)

        all_scores = obb_pr_w.prob.squeeze(-1).clone()
        keepers = all_scores >= thresh3d
        obb_pr_w = obb_pr_w[keepers].clone()
        scores3d = all_scores[keepers]
        labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
        n_3d = len(labels3d)
        bench_mark("boxer_post", sync_cuda=True)
        t_boxer_model_done = time.perf_counter()

        if n_3d > 0:
            sem_ids3d = obb_pr_w.sem_id.squeeze(-1).cpu().numpy().astype(int).tolist()
            bb3_colors, bb3_rgb_colors = get_obb_color_arrays(
                labels3d, sem_ids3d, scores3d, bb3_use_class_colors
            )
            bb3_overlay_colors, bb3_overlay_rgb_colors = get_obb_color_arrays(
                labels3d, sem_ids3d, scores3d, False
            )
            obb_pr_w.set_color(torch.from_numpy(bb3_rgb_colors).float())
            bb3_texts = [
                f"{label[:10]} {float(score):.2f}"
                for label, score in zip(labels3d, scores3d.tolist())
            ]
            if render_cpu_overlays:
                viz_3d = draw_bb3s(
                    viz_3d,
                    T_wr,
                    cam,
                    obb_pr_w,
                    draw_label=False,
                    draw_score=False,
                    render_obb_corner_steps=6,
                    already_rotated=rotated_bool,
                    rotate_label=rotated_bool,
                    colors=bb3_overlay_colors,
                    texts=bb3_texts,
                    text_sz=0.35,
                    thickness=bb3_line_width,
                )
        bench_mark("viz_3d")
        if use_cuda_timing:
            torch.cuda.synchronize()
        t_boxer_done = time.perf_counter()
    else:
        t_boxer_model_done = t_boxer_start
        bench_mark("boxer_skip")
        t_boxer_done = time.perf_counter()

    t_final_overlay_start = time.perf_counter()
    if render_cpu_overlays:
        put_text(viz_3d, "Projected BoxerNet 3DBBs", scale=0.6, line=0)
    t_rgb_total_done = time.perf_counter()
    timings = {
        "snapshot": (t_after_snapshot - bench_total_t0) * 1000.0,
        "preprocess": (t_after_preprocess - t_after_snapshot) * 1000.0,
        "rectify": (t_after_rectify - t_after_preprocess) * 1000.0,
        "setup": (t_before_owl_sync - t_after_rectify) * 1000.0,
        "gpu_wait": (t_after_owl_sync - t_before_owl_sync) * 1000.0,
        "owl": (t_owl_model_done - t_start) * 1000.0,
        "owl_render": (t_viz_2d_done - t_owl_model_done) * 1000.0,
        "boxer_setup": (t_boxer_start - t_viz_2d_done) * 1000.0,
        "boxer": (t_boxer_model_done - t_boxer_start) * 1000.0,
        "boxer_render": (t_boxer_done - t_boxer_model_done) * 1000.0,
        "post_boxer": (t_final_overlay_start - t_boxer_done) * 1000.0,
        "final_overlay": (t_rgb_total_done - t_final_overlay_start) * 1000.0,
    }
    timings["rgb_sum"] = sum(timings.values())
    timings["rgb_actual"] = (t_rgb_total_done - bench_total_t0) * 1000.0
    timings["rgb_gap"] = timings["rgb_actual"] - timings["rgb_sum"]
    bench_times["rgb_total"] = timings["rgb_actual"]

    return {
        "viz_rgb_bgr": viz_rgb,
        "viz_2d_bgr": viz_2d,
        "viz_3d_bgr": viz_3d,
        "obb_pr_w": obb_pr_w,
        "scores3d": scores3d,
        "labels3d": labels3d,
        "bb3_rgb_colors": bb3_rgb_colors,
        "bb3_overlay_rgb_colors": bb3_overlay_rgb_colors,
        "bb2d_overlay": bb2d.detach().cpu().numpy().astype(np.float32),
        "bb2_texts": bb2_texts,
        "bb2_rgb_colors": bb2_colors_rgb,
        "sdp_patch0": sdp_patch.cpu() if sdp_patch is not None else None,
        "sdp_patch_valid": sdp_patch_valid,
        "sdp_patch_median": sdp_patch_median,
        "owl_ms": timings["owl"],
        "boxer_ms": timings["boxer"],
        "rgb_infer_ms": (t_boxer_done - t_start) * 1000.0,
        "T_wr": T_wr,
        "cam": cam,
        "rotated0": rotated0,
        "n_2d": n_2d,
        "n_3d": n_3d,
        "ts_ns": ts_ns,
        "bench": bench_times,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

_LINE_VS = """
#version 330
in vec2 in_quad_pos;
in vec3 start_pos;
in vec3 end_pos;
in vec3 line_color;
in float line_prob;
uniform mat4 mvp;
uniform float line_width;
uniform vec2 viewport_size;
out vec3 v_color;
out float v_prob;
void main() {
    vec4 clip_start = mvp * vec4(start_pos, 1.0);
    vec4 clip_end = mvp * vec4(end_pos, 1.0);
    vec2 ndc_start = clip_start.xy / clip_start.w;
    vec2 ndc_end = clip_end.xy / clip_end.w;
    vec2 screen_start = (ndc_start * 0.5 + 0.5) * viewport_size;
    vec2 screen_end = (ndc_end * 0.5 + 0.5) * viewport_size;
    vec2 line_vec = screen_end - screen_start;
    float line_length = length(line_vec);
    vec2 line_dir = line_length > 0.0 ? line_vec / line_length : vec2(1.0, 0.0);
    vec2 line_perp = vec2(-line_dir.y, line_dir.x);
    float t = in_quad_pos.x * 0.5 + 0.5;
    vec2 center = mix(screen_start, screen_end, t);
    vec2 offset = line_perp * (line_width * 0.5) * in_quad_pos.y;
    vec2 screen_pos = center + offset;
    vec2 ndc_pos = (screen_pos / viewport_size) * 2.0 - 1.0;
    float depth = mix(clip_start.z / clip_start.w, clip_end.z / clip_end.w, t);
    float w = mix(clip_start.w, clip_end.w, t);
    gl_Position = vec4(ndc_pos * w, depth * w, w);
    v_color = line_color;
    v_prob = line_prob;
}
"""

_LINE_FS = """
#version 330
uniform float alpha;
uniform float prob_threshold;
in vec3 v_color;
in float v_prob;
out vec4 f_color;
void main() {
    float final_alpha = v_prob >= prob_threshold ? alpha : 0.0;
    f_color = vec4(v_color, final_alpha);
}
"""


class LiveBoxerViewer(OrbitViewer):
    title = "Live BoxerNet"
    window_size = (3200, 1800)

    # Injected before mglw.run_window_config(LiveBoxerViewer):
    state: StreamState = None
    owl: OwlWrapper = None
    boxernet: BoxerNet = None
    text_labels: list = None
    sem_name_to_id: dict = None
    sem_id_to_name: dict = None
    boxernet_ckpt: str = ""
    HW: int = 960
    detector_hw: int = 960
    init_thresh3d: float = 0.5
    dev: str = "cpu"
    pdtype = torch.float32
    debug_geometry: bool = False
    live_rotation: str = "none"
    fs_state: Optional[LiveFsState] = None
    fs_ckpt: str = ""
    fs_impl: str = "foundation"
    fs_hw: int = 256
    fs_valid_iters: int = 16
    consistency: bool = False
    consistency_threshold: float = 1.0
    fsp_every: int = 1
    fs_disparity_median: int = 0
    fs_point_stride: int = 2
    fs_max_depth: float = 5.0
    vio_world_is_y_up: bool = False
    show_fs_points: bool = True
    show_fs_trajectory: bool = True
    fs_use_depth_colormap: bool = False
    fs_point_size: float = 2.0
    fs_point_alpha: float = 0.85
    fs_line_width: float = 2.0
    fs_frustum_scale: float = 0.12
    enable_owl: bool = True
    enable_boxer: bool = True
    enable_tracker: bool = False
    enable_foundation_stereo: bool = True
    rectify_rgb_for_owl_boxes: bool = False
    max_steps: int = 0
    bench: bool = False
    bench_every: int = 30
    fs_async: bool = True
    rgb_gpu_overlays: bool = True
    record_fps: float = 5.0
    ui_capture_path: str = ""
    ui_capture_frame: int = 3
    fs_debug_stats: bool = False
    follow_mode: bool = False
    follow_back: float = 3.0
    follow_up: float = 3.0
    follow_lookahead: float = 0.20
    follow_smoothing: float = 0.25
    show_obbs_3d: bool = True
    show_frustum: bool = True
    show_world_axes: bool = True
    show_rgb_fs_points: bool = False
    show_rgb_fs: bool = False
    show_rgb_owl: bool = True
    show_rgb_boxer: bool = True
    show_rgb_tracker: bool = False
    split_rgb_overlays: bool = True
    show_raw_by_track_match: bool = True
    show_track_assoc_lines: bool = True
    fs_color_points_by_obb: bool = True
    use_fs_for_boxer_sdp: bool = True
    fs_boxer_max_points: int = 12000
    tracker_line_width: float = 6.0

    # Layout
    ui_panel_width = 520
    rgb_panel_width = 960
    viz_panel_width = 0
    prompt_bar_height = 112.0
    frustum_scale = 0.12
    initial_prompts_csv: str = ""

    def init_scene(self) -> None:
        # Line shader (instanced quads)
        self.line_prog = self.ctx.program(
            vertex_shader=_LINE_VS, fragment_shader=_LINE_FS
        )
        quad_vertices = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                 1.0,  1.0,
                -1.0, -1.0,
                 1.0,  1.0,
                -1.0,  1.0,
            ],
            dtype=np.float32,
        )
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())

        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)

        # Per-frame caches
        self._last_ts = -1
        self._target_inited = False
        self._n_2d = 0
        self._n_3d = 0
        self._frame_count = 0
        self._frame_count_t0 = time.perf_counter()
        self._last_loop_t0 = time.perf_counter()
        self._render_steps = 0
        self._fps = 0.0
        self._prediction_count = 0
        self._prediction_count_t0 = time.perf_counter()
        self._prediction_fps = 0.0
        self._prediction_update_ms = 0.0
        self._prediction_e2e_ms = 0.0
        self._prediction_period_ms = 0.0
        self._last_prediction_done_t: Optional[float] = None
        self._prediction_pending_t0: Optional[float] = None
        self._prediction_fs_age_ms = float("nan")
        self._fs_last_apply_t: Optional[float] = None
        self._fs_last_pipeline_ms = 0.0
        self._owl_ms = 0.0
        self._owl_render_ms = 0.0
        self._boxer_ms = 0.0
        self._boxer_render_ms = 0.0
        self._rgb_timing_sum_ms = 0.0
        self._rgb_timing_actual_ms = 0.0
        self._rgb_timing_gap_ms = 0.0
        self._rgb_timing_overhead_ms = 0.0
        self._loop_timing_sum_ms = 0.0
        self._loop_timing_actual_ms = 0.0
        self._loop_timing_gap_ms = 0.0
        self._loop_timing_overhead_ms = 0.0
        self._fs_update_ms = 0.0
        self._rgb_update_ms = 0.0
        self._timing_histories: dict[str, list[float]] = {}
        self._total_frame_ms = 0.0
        self._render_only_ms = 0.0
        self._bench_enabled = bool(type(self).bench)
        self._bench_every = max(1, int(type(self).bench_every))
        self._bench_loop_idx = 0
        self._bench_infer_idx = 0
        self._bench_last_loop = {}
        self._bench_last_infer = {}
        self._tracker_ms = 0.0
        self._fs_points_in_obbs = 0
        self._boxer_sdp_patch_valid = 0
        self._boxer_sdp_patch_median = float("nan")
        self._n_tracks = 0
        self._tracker_frame_idx = 0
        self._recording = False
        self._record_dir: Optional[str] = None
        self._record_frame_idx = 0
        self._record_fps = max(1.0, float(type(self).record_fps))
        self._last_record_mp4: Optional[str] = None
        self._ui_capture_path = str(type(self).ui_capture_path or "")
        self._ui_capture_frame = max(1, int(type(self).ui_capture_frame))
        self._ui_capture_done = False

        # GL resources
        self._rgb_texture: Optional[moderngl.Texture] = None
        self._rgb_tex_size: Optional[tuple[int, int]] = None
        self._obb_vbo: Optional[moderngl.Buffer] = None
        self._obb_vao: Optional[moderngl.VertexArray] = None
        self._obb_count = 0
        self._tracked_obb_vbo: Optional[moderngl.Buffer] = None
        self._tracked_obb_vao: Optional[moderngl.VertexArray] = None
        self._tracked_obb_count = 0
        self._match_line_vbo: Optional[moderngl.Buffer] = None
        self._match_line_vao: Optional[moderngl.VertexArray] = None
        self._match_line_count = 0
        self._frustum_vbo: Optional[moderngl.Buffer] = None
        self._frustum_vao: Optional[moderngl.VertexArray] = None
        self._frustum_count = 0
        self._axis_vbo: Optional[moderngl.Buffer] = None
        self._axis_vao: Optional[moderngl.VertexArray] = None
        self._axis_count = 0
        self._axis_origin: Optional[np.ndarray] = None
        self.point_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_color;
                uniform mat4 mvp;
                uniform float point_size;
                out vec3 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = point_size;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                uniform float alpha;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, alpha);
                }
            """,
        )
        self.fs_point_vbo: Optional[moderngl.Buffer] = None
        self.fs_point_vao: Optional[moderngl.VertexArray] = None
        self.fs_point_count = 0
        self.fs_trail_vbo: Optional[moderngl.Buffer] = None
        self.fs_trail_vao: Optional[moderngl.VertexArray] = None
        self.fs_trail_count = 0
        self._fs_last_pair_ts = -1
        self._fs_pair_seen = 0
        self._fs_last_seen_pair_ts = -1
        self._fs_processed = 0
        self._fs_infer_ms = 0.0
        self._fs_min_depth = float("nan")
        self._fs_max_depth = float("nan")
        self._fs_mean_depth = float("nan")
        self._fs_median_depth = float("nan")
        self._fs_pair_delta_ms = 0.0
        self._fs_t0 = time.time()
        self._fs_target_inited = False
        self._fs_pose_tail: list[tuple[int, np.ndarray]] = []
        self._last_T_world_rgb_cam: Optional[np.ndarray] = None
        self._fs_last_T_world_device: Optional[np.ndarray] = None
        self._fs_last_T_world_rect: Optional[np.ndarray] = None
        self._fs_debug_last_print = 0
        self._fs_overlay_pts_world: Optional[np.ndarray] = None
        self._fs_overlay_depths: Optional[np.ndarray] = None
        self._fs_boxer_pts_world: Optional[np.ndarray] = None
        self._fs_boxer_pair_ts: int = -1
        self._fs_overlay_debug_last_print = 0
        self.fs_runtime: Optional[FoundationStereoRuntime] = None
        self._fs_async_enabled = bool(type(self).fs_async)
        self._fs_executor: Optional[ThreadPoolExecutor] = None
        self._fs_future: Optional[Future] = None
        self._fs_pending_meta: Optional[dict] = None
        self._fs_pending_pair_ts: int = -1
        if self._fs_async_enabled:
            self._fs_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="foundation_stereo"
            )
        self._latest_obbs_3d = ObbTW(torch.zeros(0, 165))
        self._latest_scores_3d = torch.zeros(0)
        self._latest_tracked_obbs_3d = ObbTW(torch.zeros(0, 165))
        self._latest_tracked_scores_3d = torch.zeros(0)
        self._latest_tracked_texts: list[str] = []
        self._latest_tracked_colors_bgr: list[tuple[int, int, int]] = []
        self._latest_track_ids: list[int] = []
        self._latest_raw_track_matches: dict[int, int] = {}
        self._rgb_overlay_rotated = False
        self._rgb_overlay_img_hw: tuple[int, int] = (0, 0)
        self._rgb_overlay_bb2: list[tuple[float, float, float, float, tuple[int, int, int], str]] = []
        self._rgb_overlay_bb3_lines: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
        self._rgb_overlay_tracked_bb3_lines: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
        self._n_track_matches = 0
        self._ui_interaction_active = False
        self._resize_drag_active = False
        self._resize_ui_panel_width = float(type(self).ui_panel_width)
        self._resize_rgb_panel_width = float(type(self).rgb_panel_width)
        self._resize_viz_panel_width = float(type(self).viz_panel_width)
        self.prompt_editor_text = str(type(self).initial_prompts_csv)
        self.boxernet_ckpts = discover_boxernet_checkpoints(str(type(self).boxernet_ckpt))
        self.current_boxernet_ckpt = str(type(self).boxernet_ckpt or "")
        if not self.current_boxernet_ckpt and self.boxernet_ckpts:
            self.current_boxernet_ckpt = self.boxernet_ckpts[0]
        current_abs = os.path.abspath(os.path.expanduser(self.current_boxernet_ckpt))
        ckpt_abs = [
            os.path.abspath(os.path.expanduser(path)) for path in self.boxernet_ckpts
        ]
        self.boxernet_ckpt_index = ckpt_abs.index(current_abs) if current_abs in ckpt_abs else 0
        self._boxernet_load_status = (
            f"Loaded: {_short_ckpt_name(self.current_boxernet_ckpt)}"
            if self.current_boxernet_ckpt
            else "No BoxerNet checkpoint found"
        )
        self.tracker = BoundingBox3DTracker(
            iou_threshold=0.25,
            min_hits=3,
            conf_threshold=float(self.init_thresh3d),
            samp_per_dim=8,
            max_missed=90,
            force_cpu=(self.dev == "cpu"),
            verbose=False,
        )

        # ImGui-controlled state
        self.thresh2d = float(self.owl.min_confidence)
        self.owl_nms_iou = float(getattr(self.owl, "nms_iou_threshold", 0.5))
        self.thresh3d = float(self.init_thresh3d)
        self.show_obbs_3d = bool(type(self).show_obbs_3d)
        self.show_frustum = bool(type(self).show_frustum)
        self.show_world_axes = bool(type(self).show_world_axes)
        self.enable_owl = bool(type(self).enable_owl)
        self.enable_boxer = bool(type(self).enable_boxer)
        self.enable_tracker = bool(type(self).enable_tracker)
        self.enable_foundation_stereo = bool(type(self).enable_foundation_stereo)
        self.rectify_rgb_for_owl_boxes = bool(type(self).rectify_rgb_for_owl_boxes)
        self.fs_debug_stats = bool(type(self).fs_debug_stats)
        self.fsp_every = max(1, int(type(self).fsp_every))
        self.fs_disparity_median = max(0, int(type(self).fs_disparity_median))
        self.follow_mode = bool(type(self).follow_mode)
        self.follow_back = float(type(self).follow_back)
        self.follow_up = float(type(self).follow_up)
        self.follow_lookahead = float(type(self).follow_lookahead)
        self.follow_smoothing = float(type(self).follow_smoothing)
        self.show_fs_points = bool(type(self).show_fs_points)
        self.show_fs_trajectory = bool(type(self).show_fs_trajectory)
        self.show_rgb_fs_points = bool(type(self).show_rgb_fs_points)
        self.show_rgb_fs = bool(type(self).show_rgb_fs)
        self.show_rgb_owl = bool(type(self).show_rgb_owl)
        self.show_rgb_boxer = bool(type(self).show_rgb_boxer)
        self.show_rgb_tracker = bool(type(self).show_rgb_tracker)
        self.split_rgb_overlays = bool(type(self).split_rgb_overlays)
        self.show_raw_by_track_match = bool(type(self).show_raw_by_track_match)
        self.show_track_assoc_lines = bool(type(self).show_track_assoc_lines)
        self.fs_use_depth_colormap = bool(type(self).fs_use_depth_colormap)
        self.fs_color_points_by_obb = bool(type(self).fs_color_points_by_obb)
        self.use_fs_for_boxer_sdp = bool(type(self).use_fs_for_boxer_sdp)
        self.fs_boxer_max_points = int(type(self).fs_boxer_max_points)
        self.tracker_line_width = float(type(self).tracker_line_width)
        self.bb3_use_class_colors = True
        self.bb2_line_width = 2
        self.bb3_image_line_width = 2
        self.line_width = 3.0
        self.frustum_line_width = 3.0
        self.axis_line_width = 5.0
        self.axis_length = 0.5

        # Better default viewing pose: look down at origin
        self.camera_distance = 4.0
        self.camera_azimuth = -90.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype="f4")
        self._rebuild_world_axes(self.camera_target)

        if self.enable_foundation_stereo and self.fs_state is not None:
            self._load_foundation_stereo()

    def _reset_tracker_state(self) -> None:
        self.tracker.reset()
        self._tracker_frame_idx = 0
        self._tracker_ms = 0.0
        self._n_tracks = 0
        self._latest_tracked_obbs_3d = ObbTW(torch.zeros(0, 165))
        self._latest_tracked_scores_3d = torch.zeros(0)
        self._latest_tracked_texts = []
        self._latest_tracked_colors_bgr = []
        self._latest_track_ids = []
        self._latest_raw_track_matches = {}
        self._n_track_matches = 0
        self._rebuild_tracked_obb_lines(
            self._latest_tracked_obbs_3d, self._latest_tracked_scores_3d
        )

    def _reload_boxernet_checkpoint(self, ckpt_path: str) -> None:
        ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
        if not os.path.exists(ckpt_path):
            self._boxernet_load_status = f"Missing: {_short_ckpt_name(ckpt_path)}"
            print(f"==> BoxerNet checkpoint missing: {ckpt_path}", flush=True)
            return
        print(f"==> Loading BoxerNet checkpoint: {ckpt_path}", flush=True)
        old_boxernet = self.boxernet
        self.boxernet = None
        if self.dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            new_boxernet = BoxerNet.load_from_checkpoint(ckpt_path, device=self.dev)
        except Exception as err:
            self.boxernet = old_boxernet
            self._boxernet_load_status = (
                f"Load failed: {_short_ckpt_name(ckpt_path)} ({err})"
            )
            print(f"==> BoxerNet checkpoint load failed: {err}", flush=True)
            return
        self.boxernet = new_boxernet
        self.current_boxernet_ckpt = ckpt_path
        self.HW = int(new_boxernet.hw)
        self._last_ts = -1
        self._reset_tracker_state()
        self._boxernet_load_status = (
            f"Loaded: {_short_ckpt_name(ckpt_path)}  hw={int(new_boxernet.hw)}"
        )
        print(
            f"==> BoxerNet checkpoint active: {ckpt_path} (hw={int(new_boxernet.hw)})",
            flush=True,
        )

    def _apply_prompt_editor(self) -> None:
        prompts = [s.strip() for s in self.prompt_editor_text.split(",")]
        prompts = [s for s in prompts if s]
        if not prompts:
            return
        self.owl.set_text_prompts(prompts)
        self.text_labels = list(prompts)
        self.sem_name_to_id = {label: i for i, label in enumerate(self.text_labels)}
        self.sem_id_to_name = {i: label for i, label in enumerate(self.text_labels)}
        self._reset_tracker_state()
        print(
            f"==> Updated live text prompts: {len(self.text_labels)} -> {', '.join(self.text_labels)}",
            flush=True,
        )

    # -- viewport / camera --

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        w, h = self.wnd.size
        vw = max(1, int(w - self.ui_panel_width))
        return vw, h

    @staticmethod
    def _world_to_viewer(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)

    def _get_3d_viewport_origin_x(self) -> int:
        return int(self.ui_panel_width)

    def _is_in_prompt_bar(self, x: float, y: float) -> bool:
        win_w, win_h = self.wnd.size
        prompt_h = float(type(self).prompt_bar_height)
        prompt_x = float(self.ui_panel_width)
        return prompt_x <= float(x) < float(win_w) and float(y) >= float(win_h) - prompt_h

    def _is_in_3d_viewport(self, x: float, y: float) -> bool:
        if self._is_in_prompt_bar(x, y):
            return False
        return super()._is_in_3d_viewport(x, y)

    def _get_orbit_pick_points(self) -> Optional[np.ndarray]:
        pts = []
        if self._fs_overlay_pts_world is not None and len(self._fs_overlay_pts_world) > 0:
            fs_pts = np.asarray(self._fs_overlay_pts_world, dtype=np.float32)
            max_fs_pick_pts = 16384
            if len(fs_pts) > max_fs_pick_pts:
                stride = max(1, len(fs_pts) // max_fs_pick_pts)
                fs_pts = fs_pts[::stride]
            pts.append(self._world_to_viewer(fs_pts))

        for obbs in (self._latest_tracked_obbs_3d, self._latest_obbs_3d):
            if obbs is None or len(obbs) == 0:
                continue
            corners = obbs.bb3corners_world.cpu().numpy().reshape(-1, 3).astype(np.float32)
            if len(corners) > 0:
                pts.append(self._world_to_viewer(corners))

        if not pts:
            return None
        return np.concatenate(pts, axis=0)

    def _clamp_panel_width_values(
        self, ui_w: float, rgb_w: float, viz_w: float
    ) -> tuple[float, float, float]:
        win_w, _ = self.wnd.size
        min_3d_width = 260
        ui_w = float(np.clip(ui_w, 260, 700))
        rgb_w = float(np.clip(rgb_w, 320, 1500))
        viz_w = 0.0
        max_total = max(560, win_w - min_3d_width)
        total = ui_w + rgb_w
        if total <= max_total:
            return ui_w, rgb_w, viz_w
        overflow = total - max_total
        shrink_rgb = min(overflow, max(0, rgb_w - 320))
        rgb_w -= shrink_rgb
        overflow -= shrink_rgb
        if overflow > 0:
            ui_w = max(240, ui_w - overflow)
        return ui_w, rgb_w, viz_w

    def _clamp_panel_widths(self) -> None:
        (
            self.ui_panel_width,
            self.rgb_panel_width,
            self.viz_panel_width,
        ) = self._clamp_panel_width_values(
            self.ui_panel_width, self.rgb_panel_width, self.viz_panel_width
        )

    def get_camera_matrices(self):
        from utils.viewer_3d import _look_at, _perspective_projection

        vw, vh = self._get_3d_viewport_size()
        aspect_ratio = vw / max(1, vh)
        projection = _perspective_projection(45.0, aspect_ratio, 0.05, 200.0)
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        cx = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        cy = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        cz = self.camera_distance * np.sin(elevation_rad)
        camera_pos = self.camera_target + np.array([cx, cy, cz])
        view = _look_at(tuple(camera_pos), tuple(self.camera_target), (0.0, 0.0, 1.0))
        mvp = np.eye(4, dtype="f4") @ view @ projection
        return projection, view, mvp

    def _bench_should_print(self, idx: int, elapsed_ms: float) -> bool:
        if not self._bench_enabled:
            return False
        return idx <= 5 or idx % self._bench_every == 0 or elapsed_ms >= 200.0

    @staticmethod
    def _bench_fmt(parts: dict) -> str:
        return " ".join(f"{k}={float(v):.1f}ms" for k, v in parts.items())

    def _push_timing(self, name: str, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        hist = self._timing_histories.setdefault(name, [])
        hist.append(value)
        if len(hist) > 30:
            del hist[: len(hist) - 30]

    def _timing_mean30(self, name: str, fallback: float = 0.0) -> float:
        hist = self._timing_histories.get(name)
        if not hist:
            return float(fallback)
        return float(np.mean(hist))

    def _fmt_ms_mean30(self, name: str, value: float, signed: bool = False) -> str:
        fmt = "+.1f" if signed else ".1f"
        return f"{float(value):{fmt}} ms (m30 {self._timing_mean30(name, value):.1f} ms)"

    def _fmt_value_mean30(self, name: str, value: float, suffix: str = "") -> str:
        suffix_part = f" {suffix}" if suffix else ""
        m30_suffix = suffix_part
        return (
            f"{float(value):.1f}{suffix_part} "
            f"(m30 {self._timing_mean30(name, value):.1f}{m30_suffix})"
        )

    @staticmethod
    def _imgui_ui_busy() -> bool:
        busy = False
        for attr in ("is_any_item_active", "is_any_item_focused"):
            fn = getattr(imgui, attr, None)
            if fn is not None:
                try:
                    busy = busy or bool(fn())
                except Exception:
                    pass
        try:
            io = imgui.get_io()
            mouse_down = getattr(io, "mouse_down", [])
            any_mouse_down = any(bool(v) for v in mouse_down)
            busy = busy or (bool(getattr(io, "want_capture_mouse", False)) and any_mouse_down)
        except Exception:
            pass
        return busy

    @staticmethod
    def _rotate_image_point_cw(x: float, y: float, src_h: int) -> tuple[float, float]:
        return float(src_h - 1 - y), float(x)

    def _project_obbs_to_rgb_overlay_lines(
        self,
        obbs: ObbTW,
        T_world_cam: PoseTW,
        cam: CameraTW,
        rotated: bool,
        src_h: int,
        colors: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
        if obbs is None or len(obbs) == 0:
            return []
        edge_pts_world = obbs.T_world_object * obbs.bb3edge_pts_object(6)
        edge_pts_cam = T_world_cam.inverse() * edge_pts_world
        bsz = edge_pts_cam.shape[0]
        pts_cam = edge_pts_cam.view(bsz, -1, 3)
        pts2, valid = cam.project(pts_cam)
        pts2_np = pts2.detach().cpu().numpy().reshape(bsz, 12, 6, 2).astype(np.float32)
        valid_np = valid.detach().cpu().numpy().reshape(bsz, 12, 6).astype(bool)
        if rotated:
            x = pts2_np[..., 0].copy()
            y = pts2_np[..., 1].copy()
            pts2_np[..., 0] = float(src_h - 1) - y
            pts2_np[..., 1] = x

        lines = []
        for i in range(bsz):
            if i < len(colors):
                color = tuple(float(c) for c in colors[i])
            else:
                color = (1.0, 1.0, 1.0)
            lines.append((pts2_np[i], valid_np[i], color))
        return lines

    def _set_rgb_overlay_data(self, result: dict) -> None:
        rotated = bool(result["rotated0"].item())
        if self._rgb_tex_size is None:
            self._rgb_overlay_img_hw = (0, 0)
        else:
            tex_w, tex_h = self._rgb_tex_size
            self._rgb_overlay_img_hw = (int(tex_h), int(tex_w))
        self._rgb_overlay_rotated = rotated
        self._rgb_overlay_bb2 = []
        self._rgb_overlay_bb3_lines = []
        self._rgb_overlay_tracked_bb3_lines = []

        bb2d = np.asarray(result.get("bb2d_overlay", np.zeros((0, 4))), dtype=np.float32)
        bb2_texts = list(result.get("bb2_texts", []))
        bb2_colors = list(result.get("bb2_rgb_colors", []))
        src_h = int(self.HW)
        for i, bb in enumerate(bb2d):
            x0, x1, y0, y1 = [float(v) for v in bb[:4]]
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if rotated:
                corners = [self._rotate_image_point_cw(x, y, src_h) for x, y in corners]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            color = bb2_colors[i] if i < len(bb2_colors) else (0, 255, 0)
            text = bb2_texts[i] if i < len(bb2_texts) else ""
            self._rgb_overlay_bb2.append((min(xs), min(ys), max(xs), max(ys), color, text))

        T_world_cam = result["T_wr"].float() @ result["cam"].T_camera_rig.inverse()
        obbs = result["obb_pr_w"]
        scores = result["scores3d"]
        colors = np.asarray(
            result.get("bb3_overlay_rgb_colors", np.zeros((0, 3))),
            dtype=np.float32,
        )
        if obbs.shape[0] > 0 and scores.numel() > 0:
            self._rgb_overlay_bb3_lines = self._project_obbs_to_rgb_overlay_lines(
                obbs,
                T_world_cam,
                result["cam"],
                rotated,
                src_h,
                colors,
            )

        tracked_colors = np.zeros((0, 3), dtype=np.float32)
        if self._latest_tracked_colors_bgr:
            tracked_colors = np.asarray(
                [
                    (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)
                    for b, g, r in self._latest_tracked_colors_bgr
                ],
                dtype=np.float32,
            )
        self._rgb_overlay_tracked_bb3_lines = self._project_obbs_to_rgb_overlay_lines(
            self._latest_tracked_obbs_3d,
            T_world_cam,
            result["cam"],
            rotated,
            src_h,
            tracked_colors,
        )

    def _draw_rgb_gpu_overlays(self, draw_list, img_min, draw_w: float, draw_h: float) -> None:
        tex_w, tex_h = self._rgb_tex_size if self._rgb_tex_size is not None else (0, 0)
        if tex_w <= 0 or tex_h <= 0:
            return
        sx = float(draw_w) / float(tex_w)
        sy = float(draw_h) / float(tex_h)

        if self.show_rgb_owl:
            text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
            for x0, y0, x1, y1, color, text in self._rgb_overlay_bb2:
                r, g, b = [float(c) / 255.0 for c in color]
                col = imgui.get_color_u32_rgba(r, g, b, 1.0)
                bg_col = imgui.get_color_u32_rgba(r * 0.25, g * 0.25, b * 0.25, 0.75)
                rx0 = img_min.x + x0 * sx
                ry0 = img_min.y + y0 * sy
                rx1 = img_min.x + x1 * sx
                ry1 = img_min.y + y1 * sy
                draw_list.add_rect(
                    rx0,
                    ry0,
                    rx1,
                    ry1,
                    col,
                    0.0,
                    0,
                    float(self.bb2_line_width),
                )
                if text:
                    tw, th = imgui.calc_text_size(text)
                    draw_list.add_rect_filled(rx0 - 1, ry0 - th - 2, rx0 + tw + 2, ry0, bg_col)
                    draw_list.add_text(rx0, ry0 - th - 1, text_col, text)

        if self.show_rgb_boxer:
            for edge_pts, edge_valid, color in self._rgb_overlay_bb3_lines:
                col = imgui.get_color_u32_rgba(
                    float(color[0]), float(color[1]), float(color[2]), 1.0
                )
                for e in range(edge_pts.shape[0]):
                    for s in range(edge_pts.shape[1] - 1):
                        if edge_valid[e, s] and edge_valid[e, s + 1]:
                            x0 = img_min.x + float(edge_pts[e, s, 0]) * sx
                            y0 = img_min.y + float(edge_pts[e, s, 1]) * sy
                            x1 = img_min.x + float(edge_pts[e, s + 1, 0]) * sx
                            y1 = img_min.y + float(edge_pts[e, s + 1, 1]) * sy
                            draw_list.add_line(
                                x0,
                                y0,
                                x1,
                                y1,
                                col,
                                float(self.bb3_image_line_width),
                            )
        if self.show_rgb_tracker:
            tracked_width = float(max(2, int(round(self.bb3_image_line_width + 1))))
            for edge_pts, edge_valid, color in self._rgb_overlay_tracked_bb3_lines:
                col = imgui.get_color_u32_rgba(
                    float(color[0]), float(color[1]), float(color[2]), 1.0
                )
                for e in range(edge_pts.shape[0]):
                    for s in range(edge_pts.shape[1] - 1):
                        if edge_valid[e, s] and edge_valid[e, s + 1]:
                            x0 = img_min.x + float(edge_pts[e, s, 0]) * sx
                            y0 = img_min.y + float(edge_pts[e, s, 1]) * sy
                            x1 = img_min.x + float(edge_pts[e, s + 1, 0]) * sx
                            y1 = img_min.y + float(edge_pts[e, s + 1, 1]) * sy
                            draw_list.add_line(x0, y0, x1, y1, col, tracked_width)

    def _shutdown_fs_executor(self) -> None:
        if self._fs_executor is not None:
            self._fs_executor.shutdown(wait=False, cancel_futures=True)
            self._fs_executor = None
            self._fs_future = None
            self._fs_pending_meta = None
            self._fs_pending_pair_ts = -1

    # -- inference + GPU upload --

    def _maybe_run_inference(self) -> None:
        infer_t0 = time.perf_counter()
        # Cheap snapshot to skip duplicate frames before doing real work
        with self.state.lock:
            frame = self.state.frame
        if frame is None:
            return
        if frame[1] == self._last_ts:
            return
        snapshot_ms = (time.perf_counter() - infer_t0) * 1000.0

        # Update OWL threshold from the slider before running
        self.owl.min_confidence = float(self.thresh2d)
        self.owl.nms_iou_threshold = float(self.owl_nms_iou)

        t_sdp = time.perf_counter()
        boxer_sdp_w = self._get_boxer_sdp_w()
        sdp_ms = (time.perf_counter() - t_sdp) * 1000.0
        t_models = time.perf_counter()
        result = run_inference(
            self.state,
            self.owl,
            self.boxernet,
            self.text_labels,
            self.sem_name_to_id,
            self.sem_id_to_name,
            self.HW,
            self.detector_hw,
            float(self.thresh3d),
            int(round(self.bb2_line_width)),
            int(round(self.bb3_image_line_width)),
            self.dev,
            self.pdtype,
            self.debug_geometry,
            self.live_rotation,
            self.enable_owl,
            self.enable_boxer and self.enable_owl,
            self.bb3_use_class_colors,
            boxer_sdp_w,
            self.rectify_rgb_for_owl_boxes,
            bench=self._bench_enabled,
            render_cpu_overlays=bool(self.split_rgb_overlays)
            or not bool(type(self).rgb_gpu_overlays),
        )
        models_ms = (time.perf_counter() - t_models) * 1000.0
        if result is None:
            return

        self._last_ts = result["ts_ns"]
        self._n_2d = result["n_2d"]
        self._n_3d = result["n_3d"]
        self._owl_ms = float(result["owl_ms"])
        self._boxer_ms = float(result["boxer_ms"])
        result_timings = dict(result.get("timings", {}))
        self._owl_render_ms = float(result_timings.get("owl_render", 0.0))
        self._boxer_render_ms = float(result_timings.get("boxer_render", 0.0))
        self._push_timing("owl", self._owl_ms)
        self._push_timing("boxer", self._boxer_ms)
        self._push_timing("owl_render", self._owl_render_ms)
        self._push_timing("boxer_render", self._boxer_render_ms)
        self._boxer_sdp_patch_valid = int(result["sdp_patch_valid"])
        self._boxer_sdp_patch_median = float(result["sdp_patch_median"])
        t_convert = time.perf_counter()
        self._latest_obbs_3d = self._maybe_convert_obbs_world(result["obb_pr_w"])
        self._latest_scores_3d = result["scores3d"]
        convert_ms = (time.perf_counter() - t_convert) * 1000.0
        t_tracker = time.perf_counter()
        self._update_tracker(
            result["obb_pr_w"],
            result["scores3d"],
            result["T_wr"],
            result["cam"],
        )
        tracker_total_ms = (time.perf_counter() - t_tracker) * 1000.0
        t_track_viz = time.perf_counter()
        self._apply_track_match_visuals()
        track_viz_ms = (time.perf_counter() - t_track_viz) * 1000.0
        T_world_rgb_cam = result["T_wr"] @ result["cam"].T_camera_rig.inverse()
        self._last_T_world_rgb_cam = (
            T_world_rgb_cam.matrix.detach().cpu().numpy().astype(np.float32)
        )

        t_panels = time.perf_counter()
        panels = []
        if self.show_rgb_fs_points:
            panels.append(
                self._render_fs_points_overlay_image(
                    result["viz_rgb_bgr"],
                    self._fs_overlay_pts_world,
                    result["cam"],
                    result["T_wr"],
                )
            )
        if self.show_rgb_fs:
            panels.append(
                self._render_sdp_patch_overlay_image(
                    result["viz_rgb_bgr"],
                    result["sdp_patch0"],
                    bool(result["rotated0"].item()),
                )
            )
        use_gpu_overlays = bool(type(self).rgb_gpu_overlays)
        split_rgb_overlays = bool(self.split_rgb_overlays)
        if split_rgb_overlays:
            if self.show_rgb_owl:
                panels.append(result["viz_2d_bgr"])
            if self.show_rgb_boxer:
                panels.append(result["viz_3d_bgr"])
            if self.show_rgb_tracker:
                panels.append(
                    self._render_tracked_obb_overlay_image(
                        result["viz_rgb_bgr"],
                        result["T_wr"],
                        result["cam"],
                        bool(result["rotated0"].item()),
                    )
                )
        elif not use_gpu_overlays:
            panel = result["viz_rgb_bgr"].copy()
            if self.show_rgb_owl:
                panel = result["viz_2d_bgr"].copy()
            if self.show_rgb_boxer:
                panel = result["viz_3d_bgr"].copy()
            if self.show_rgb_tracker:
                panel = self._render_tracked_obb_overlay_image(
                    panel,
                    result["T_wr"],
                    result["cam"],
                    bool(result["rotated0"].item()),
                )
            if self.show_rgb_owl or self.show_rgb_boxer or self.show_rgb_tracker:
                panels.append(panel)
        if not panels:
            panels.append(result["viz_rgb_bgr"])
        panel_render_ms = (time.perf_counter() - t_panels) * 1000.0

        t_stack = time.perf_counter()
        if len(panels) == 1:
            viz_bgr = panels[0]
        else:
            separator = np.full((6, panels[0].shape[1], 3), 24, dtype=np.uint8)
            stacked = []
            for idx, panel in enumerate(panels):
                if idx > 0:
                    stacked.append(separator)
                stacked.append(panel)
            viz_bgr = np.vstack(stacked)
        rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
        stack_ms = (time.perf_counter() - t_stack) * 1000.0
        t_upload = time.perf_counter()
        self._upload_rgb_texture(rgb)
        upload_ms = (time.perf_counter() - t_upload) * 1000.0
        if bool(type(self).rgb_gpu_overlays) and not bool(self.split_rgb_overlays):
            self._set_rgb_overlay_data(result)

        # Right panel: rebuild 3D line geometry
        t_geom = time.perf_counter()
        self._rebuild_obb_lines(self._latest_obbs_3d, result["scores3d"])
        self._rebuild_frustum(result["cam"], result["T_wr"])
        geom_ms = (time.perf_counter() - t_geom) * 1000.0
        infer_total_ms = (time.perf_counter() - infer_t0) * 1000.0
        rgb_pipeline_sum_ms = float(result_timings.get("rgb_sum", models_ms))
        rgb_pipeline_actual_ms = float(result_timings.get("rgb_actual", models_ms))
        models_overhead_ms = models_ms - rgb_pipeline_actual_ms
        component_measured_sum_ms = (
            snapshot_ms
            + sdp_ms
            + rgb_pipeline_sum_ms
            + models_overhead_ms
            + convert_ms
            + tracker_total_ms
            + track_viz_ms
            + panel_render_ms
            + stack_ms
            + upload_ms
            + geom_ms
        )
        rgb_update_overhead_ms = infer_total_ms - component_measured_sum_ms
        component_sum_ms = component_measured_sum_ms + rgb_update_overhead_ms
        self._rgb_timing_sum_ms = component_sum_ms
        self._rgb_timing_actual_ms = infer_total_ms
        self._rgb_timing_gap_ms = infer_total_ms - component_sum_ms
        self._rgb_timing_overhead_ms = rgb_update_overhead_ms
        self._push_timing("rgb_sum", self._rgb_timing_sum_ms)
        self._push_timing("rgb_actual", self._rgb_timing_actual_ms)
        self._push_timing("rgb_gap", self._rgb_timing_gap_ms)
        self._push_timing("rgb_overhead", self._rgb_timing_overhead_ms)
        prediction_active = bool(self.enable_owl or (self.enable_boxer and self.enable_owl))
        self._prediction_update_ms = infer_total_ms if prediction_active else 0.0
        self._prediction_pending_t0 = infer_t0 if prediction_active else None
        if self._fs_last_apply_t is None:
            self._prediction_fs_age_ms = float("nan")
        else:
            self._prediction_fs_age_ms = (
                time.perf_counter() - self._fs_last_apply_t
            ) * 1000.0
        if not prediction_active:
            self._prediction_e2e_ms = 0.0
            self._prediction_period_ms = 0.0
            self._last_prediction_done_t = None
            self._prediction_count = 0
            self._prediction_count_t0 = time.perf_counter()

        if self._bench_enabled:
            self._bench_infer_idx += 1
            bench = dict(result.get("bench", {}))
            self._bench_last_infer = {
                "snapshot": snapshot_ms,
                "sdp": sdp_ms,
                "models_call": models_ms,
                "fs_pipeline": self._fs_last_pipeline_ms,
                "fs_age": self._prediction_fs_age_ms,
                "rgb_model_sum": rgb_pipeline_sum_ms,
                "rgb_model_actual": rgb_pipeline_actual_ms,
                "models_overhead": models_overhead_ms,
                "owl": self._owl_ms,
                "owl_render": self._owl_render_ms,
                "boxer": self._boxer_ms,
                "boxer_render": self._boxer_render_ms,
                "convert": convert_ms,
                "tracker": tracker_total_ms,
                "track_viz": track_viz_ms,
                "panels": panel_render_ms,
                "stack": stack_ms,
                "upload": upload_ms,
                "geom": geom_ms,
                "overhead": rgb_update_overhead_ms,
                "sum": component_sum_ms,
                "actual": infer_total_ms,
                "gap": infer_total_ms - component_sum_ms,
            }
            if self._bench_should_print(self._bench_infer_idx, infer_total_ms):
                print(
                    "==> bench_rgb "
                    f"frame={self._bench_infer_idx} ts={self._last_ts} "
                    f"n2d={self._n_2d} n3d={self._n_3d} "
                    f"tracks={self._n_tracks} "
                    + self._bench_fmt(self._bench_last_infer),
                    flush=True,
                )
                if bench:
                    print(
                        "==> bench_rgb_detail "
                        f"frame={self._bench_infer_idx} "
                        + self._bench_fmt(bench),
                        flush=True,
                    )

    def _upload_rgb_texture(self, img_rgb: np.ndarray) -> None:
        h, w = img_rgb.shape[:2]
        if self._rgb_texture is None or self._rgb_tex_size != (w, h):
            if self._rgb_texture is not None:
                self.imgui.remove_texture(self._rgb_texture)
                self._rgb_texture.release()
            self._rgb_texture = self.ctx.texture((w, h), 3, img_rgb.tobytes())
            self._rgb_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.imgui.register_texture(self._rgb_texture)
            self._rgb_tex_size = (w, h)
        else:
            self._rgb_texture.write(img_rgb.tobytes())

    def _upload_line_data(self, attr: str, data: np.ndarray) -> None:
        vbo_name = f"{attr}_vbo"
        vao_name = f"{attr}_vao"
        count_name = f"{attr}_count"

        existing_vbo = getattr(self, vbo_name, None)
        existing_vao = getattr(self, vao_name, None)
        if existing_vbo is not None:
            existing_vbo.release()
        if existing_vao is not None:
            existing_vao.release()

        if data.size == 0:
            setattr(self, vbo_name, None)
            setattr(self, vao_name, None)
            setattr(self, count_name, 0)
            return

        data = np.asarray(data, dtype=np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )
        setattr(self, vbo_name, vbo)
        setattr(self, vao_name, vao)
        setattr(self, count_name, len(data))

    def _render_sdp_patch_overlay_image(
        self,
        base_bgr: np.ndarray,
        sdp_patch0: Optional[torch.Tensor],
        rotated0: bool,
    ) -> np.ndarray:
        overlay = base_bgr.copy()
        if sdp_patch0 is None:
            put_text(overlay, "Boxer SDP patch overlay unavailable", scale=0.6, line=0)
            return overlay

        HH, WW = overlay.shape[:2]
        viz_sdp, sdp_resized = render_depth_patches(
            sdp_patch0[0].cpu(),
            rotated=rotated0,
            HH=HH,
            WW=WW,
        )
        viz_sdp = np.ascontiguousarray(viz_sdp)
        mask = sdp_resized > 0.1
        if np.any(mask):
            mask3 = mask[:, :, None]
            overlay = np.where(
                mask3,
                (
                    (viz_sdp.astype(np.uint16) * 51 + overlay.astype(np.uint16) * 205)
                    >> 8
                ).astype(np.uint8),
                overlay,
            )
        put_text(overlay, "Boxer SDP patches", scale=0.6, line=0)
        return overlay

    def _render_fs_points_overlay_image(
        self,
        base_bgr: np.ndarray,
        pts_world: Optional[np.ndarray],
        cam: CameraTW,
        T_wr: PoseTW,
    ) -> np.ndarray:
        overlay = base_bgr.copy()
        if pts_world is None or len(pts_world) == 0:
            put_text(overlay, "FoundationStereo RGB points unavailable", scale=0.6, line=0)
            return overlay

        pts_world = np.asarray(pts_world, dtype=np.float32)
        max_points = 40000
        if len(pts_world) > max_points:
            step = int(np.ceil(len(pts_world) / max_points))
            pts_world = pts_world[::step]

        pts_world_t = torch.from_numpy(pts_world).to(device=T_wr.device, dtype=torch.float32)
        T_wc = T_wr @ cam.T_camera_rig.inverse()
        pts_cam = T_wc.inverse().transform(pts_world_t)
        pts_2d, valid = cam.project(pts_cam.unsqueeze(0))
        pts_2d = pts_2d.squeeze(0).detach().cpu().numpy()
        valid = valid.squeeze(0).detach().cpu().numpy().astype(bool)
        z = pts_cam[..., 2].detach().cpu().numpy()
        valid &= np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            put_text(overlay, "FoundationStereo RGB points unavailable", scale=0.6, line=0)
            return overlay

        pts_2d = np.round(pts_2d[valid]).astype(np.int32)
        z = z[valid]
        hh, ww = overlay.shape[:2]
        in_bounds = (
            (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < ww)
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < hh)
        )
        if not np.any(in_bounds):
            put_text(overlay, "FoundationStereo RGB points unavailable", scale=0.6, line=0)
            return overlay

        pts_2d = pts_2d[in_bounds]
        z = z[in_bounds]
        z_norm = np.clip((z - 0.1) / (5.0 - 0.1), 0.0, 1.0)
        colors = cv2.applyColorMap(
            np.round(z_norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET
        )[:, 0, :]
        for (x, y), color in zip(pts_2d, colors):
            cv2.circle(overlay, (int(x), int(y)), 1, tuple(int(c) for c in color.tolist()), -1)
        put_text(overlay, "FoundationStereo RGB points", scale=0.6, line=0)
        return overlay

    def _render_tracked_obb_overlay_image(
        self,
        base_bgr: np.ndarray,
        T_wr: PoseTW,
        cam: CameraTW,
        rotated0: bool,
    ) -> np.ndarray:
        overlay = base_bgr.copy()
        if self._latest_tracked_obbs_3d is None or len(self._latest_tracked_obbs_3d) == 0:
            put_text(overlay, "Tracked 3DBBs unavailable", scale=0.6, line=0)
            return overlay
        overlay = draw_bb3s(
            viz=overlay,
            T_world_rig=T_wr,
            cam=cam,
            obbs=self._latest_tracked_obbs_3d,
            already_rotated=rotated0,
            rotate_label=rotated0,
            colors=self._latest_tracked_colors_bgr,
            texts=self._latest_tracked_texts,
            text_sz=0.35,
            thickness=max(2, int(round(self.bb3_image_line_width + 1))),
        )
        put_text(overlay, "Tracked 3DBBs", scale=0.6, line=0)
        return overlay

    def _build_obb_line_instances(
        self, obbs: ObbTW, scores: torch.Tensor
    ) -> np.ndarray:
        N = len(obbs)
        if N == 0:
            return np.zeros((0, 10), dtype=np.float32)

        corners = obbs.bb3corners_world  # (N, 8, 3)
        edge_idx = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long)
        batch_idx = torch.arange(N)[:, None].expand(N, 12)
        s_idx = edge_idx[:, 0][None, :].expand(N, 12)
        e_idx = edge_idx[:, 1][None, :].expand(N, 12)
        s = corners[batch_idx, s_idx]
        e = corners[batch_idx, e_idx]

        obb_colors = obbs.color.float().cpu()
        if obb_colors.ndim == 1:
            obb_colors = obb_colors.unsqueeze(0)
        if obb_colors.shape[0] == N and torch.all(obb_colors >= 0):
            col = obb_colors
        else:
            rgb = jet_colors_rgb_float(scores.tolist())
            col = (
                torch.tensor(rgb, dtype=torch.float32)
                if N > 0
                else torch.zeros(0, 3)
            )
        col = col[:, None, :].expand(N, 12, 3)
        prob = scores.float()[:, None, None].expand(N, 12, 1)

        instance = torch.cat([s, e, col, prob], dim=2).reshape(-1, 10)
        return instance.cpu().numpy().astype("f4")

    def _rebuild_obb_lines(self, obbs: ObbTW, scores: torch.Tensor) -> None:
        instance_np = self._build_obb_line_instances(obbs, scores)
        self._upload_line_data("_obb", instance_np)

    def _rebuild_tracked_obb_lines(self, obbs: ObbTW, scores: torch.Tensor) -> None:
        instance_np = self._build_obb_line_instances(obbs, scores)
        self._upload_line_data("_tracked_obb", instance_np)

    def _update_tracker(
        self,
        obb_pr_w: ObbTW,
        scores3d: torch.Tensor,
        T_wr: PoseTW,
        cam: CameraTW,
    ) -> None:
        self._tracker_ms = 0.0
        self._n_tracks = 0
        self._n_track_matches = 0
        self._latest_tracked_obbs_3d = ObbTW(torch.zeros(0, 165))
        self._latest_tracked_scores_3d = torch.zeros(0)
        self._latest_tracked_texts = []
        self._latest_tracked_colors_bgr = []
        self._latest_track_ids = []
        self._latest_raw_track_matches = {}
        self._rebuild_tracked_obb_lines(
            self._latest_tracked_obbs_3d, self._latest_tracked_scores_3d
        )
        self._upload_line_data("_match_line", np.zeros((0, 10), dtype=np.float32))
        if not self.enable_tracker:
            return
        self.tracker.conf_threshold = float(self.thresh3d)

        observed_points = None
        if self._fs_boxer_pts_world is not None and len(self._fs_boxer_pts_world) > 0:
            boxer_pts = self._fs_boxer_pts_world
            if len(boxer_pts) > int(self.fs_boxer_max_points):
                step = int(np.ceil(len(boxer_pts) / max(int(self.fs_boxer_max_points), 1)))
                boxer_pts = boxer_pts[::step]
            observed_points = torch.from_numpy(
                np.asarray(boxer_pts, dtype=np.float32)
            )

        t0 = time.perf_counter()
        active_tracks = self.tracker.update(
            obb_pr_w,
            self._tracker_frame_idx,
            cam=cam,
            T_world_rig=T_wr,
            observed_points=observed_points,
        )
        self._tracker_ms = (time.perf_counter() - t0) * 1000.0
        self._push_timing("tracker", self._tracker_ms)
        self._tracker_frame_idx += 1
        self._n_tracks = len(active_tracks)
        self._latest_raw_track_matches = dict(self.tracker.last_matches)
        self._n_track_matches = len(self._latest_raw_track_matches)
        if len(active_tracks) == 0:
            return

        tracked_obbs = torch.stack([t.obb for t in active_tracks])
        track_scores = tracked_obbs.prob.squeeze(-1).detach().cpu().float()
        track_colors = torch.tensor(
            [TAB20[t.track_id % len(TAB20)] for t in active_tracks],
            dtype=torch.float32,
        )
        tracked_obbs.set_color(track_colors)
        self._latest_tracked_obbs_3d = self._maybe_convert_obbs_world(tracked_obbs)
        self._latest_tracked_scores_3d = track_scores
        self._latest_track_ids = [t.track_id for t in active_tracks]
        self._latest_tracked_texts = [
            f"T{t.track_id} {t.cached_text[:10]}" for t in active_tracks
        ]
        self._latest_tracked_colors_bgr = [
            tuple(int(round(255.0 * c)) for c in reversed(TAB20[t.track_id % len(TAB20)]))
            for t in active_tracks
        ]
        self._rebuild_tracked_obb_lines(
            self._latest_tracked_obbs_3d, self._latest_tracked_scores_3d
        )

    def _apply_track_match_visuals(self) -> None:
        self._upload_line_data("_match_line", np.zeros((0, 10), dtype=np.float32))
        if (
            self._latest_obbs_3d is None
            or len(self._latest_obbs_3d) == 0
            or not self._latest_raw_track_matches
            or self._latest_tracked_obbs_3d is None
            or len(self._latest_tracked_obbs_3d) == 0
        ):
            return

        track_id_to_idx = {tid: i for i, tid in enumerate(self._latest_track_ids)}
        raw_obbs = self._latest_obbs_3d.clone()
        raw_colors = raw_obbs.color.float().clone()
        matched_lines = []
        raw_centers = raw_obbs.bb3_center_world.detach().cpu().numpy().astype(np.float32)
        tracked_centers = (
            self._latest_tracked_obbs_3d.bb3_center_world.detach().cpu().numpy().astype(np.float32)
        )
        for raw_idx, track_id in self._latest_raw_track_matches.items():
            track_idx = track_id_to_idx.get(track_id)
            if track_idx is None or raw_idx >= len(raw_obbs):
                continue
            rgb = np.array(TAB20[track_id % len(TAB20)], dtype=np.float32)
            if self.show_raw_by_track_match:
                raw_colors[raw_idx] = torch.from_numpy(rgb)
            if self.show_track_assoc_lines:
                start = raw_centers[raw_idx]
                end = tracked_centers[track_idx]
                if np.linalg.norm(end - start) > 1e-4:
                    matched_lines.append(
                        np.concatenate(
                            [
                                start,
                                end,
                                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                np.array([1.0], dtype=np.float32),
                            ],
                            axis=0,
                        )
                    )
        if self.show_raw_by_track_match:
            raw_obbs.set_color(raw_colors)
            self._latest_obbs_3d = raw_obbs
        if matched_lines:
            self._upload_line_data(
                "_match_line", np.asarray(matched_lines, dtype=np.float32)
            )

    def _rebuild_frustum(self, cam: CameraTW, T_wr: PoseTW) -> None:
        if self._frustum_vbo is not None:
            self._frustum_vbo.release()
            self._frustum_vbo = None
        if self._frustum_vao is not None:
            self._frustum_vao.release()
            self._frustum_vao = None
        self._frustum_count = 0

        T_wc = T_wr @ cam.T_camera_rig.inverse()
        origin = T_wc.t.reshape(3).cpu().float()
        fx = cam.f[..., 0].item()
        fy = cam.f[..., 1].item()
        w_img = cam.size[..., 0].item()
        h_img = cam.size[..., 1].item()
        cx = cam.c[..., 0].item()
        cy = cam.c[..., 1].item()
        d = self.frustum_scale
        pts_cam = torch.tensor(
            [
                [(0.0 - cx) / fx * d, (0.0 - cy) / fy * d, d],
                [(w_img - cx) / fx * d, (0.0 - cy) / fy * d, d],
                [(w_img - cx) / fx * d, (h_img - cy) / fy * d, d],
                [(0.0 - cx) / fx * d, (h_img - cy) / fy * d, d],
            ],
            dtype=torch.float32,
        )
        R_wc = T_wc.R.reshape(3, 3).cpu().float()
        pts_world = (R_wc @ pts_cam.T).T + origin
        color = torch.tensor([1.0, 0.85, 0.1], dtype=torch.float32)
        segs = []
        for i in range(4):
            segs.append(torch.cat([origin, pts_world[i], color, torch.ones(1)]))
        for i in range(4):
            j = (i + 1) % 4
            segs.append(torch.cat([pts_world[i], pts_world[j], color, torch.ones(1)]))
        data = torch.stack(segs).numpy().astype("f4")
        self._frustum_count = len(data)
        self._frustum_vbo = self.ctx.buffer(data.tobytes())
        self._frustum_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self._frustum_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )

    def _rebuild_world_axes(self, origin_np: np.ndarray) -> None:
        if self._axis_vbo is not None:
            self._axis_vbo.release()
            self._axis_vbo = None
        if self._axis_vao is not None:
            self._axis_vao.release()
            self._axis_vao = None
        self._axis_count = 0

        origin = np.asarray(origin_np, dtype=np.float32).reshape(3)
        self._axis_origin = origin.copy()
        length = float(self.axis_length)
        axes = np.array(
            [
                [length, 0.0, 0.0],
                [0.0, length, 0.0],
                [0.0, 0.0, length],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.05, 0.05],  # X red
                [0.05, 1.0, 0.05],  # Y green
                [0.1, 0.35, 1.0],  # Z blue, world-up
            ],
            dtype=np.float32,
        )
        starts = np.repeat(origin[None, :], 3, axis=0)
        ends = starts + axes
        probs = np.ones((3, 1), dtype=np.float32)
        data = np.concatenate([starts, ends, colors, probs], axis=1).astype("f4")

        self._axis_count = len(data)
        self._axis_vbo = self.ctx.buffer(data.tobytes())
        self._axis_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self._axis_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )

    @staticmethod
    def _zup_from_yup_matrix() -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _depth_jet_colors(depths: np.ndarray, near: float = 0.1, far: float = 5.0):
        t = np.clip((depths.astype(np.float32) - near) / (far - near), 0.0, 1.0)
        u8 = (t * 255).astype(np.uint8).reshape(1, -1)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0].astype(np.float32) / 255.0
        return bgr[:, ::-1].astype("f4")

    def _color_fs_points_from_obbs(
        self, pts_world: np.ndarray, base_colors: np.ndarray
    ) -> np.ndarray:
        if (
            not self.fs_color_points_by_obb
            or self._latest_obbs_3d is None
            or len(self._latest_obbs_3d) == 0
            or len(pts_world) == 0
        ):
            self._fs_points_in_obbs = 0
            return base_colors

        obbs = self._latest_obbs_3d.clone()
        expand_m = 0.03
        bb3_object = obbs.bb3_object.clone()
        bb3_object[:, 0] -= expand_m
        bb3_object[:, 1] += expand_m
        bb3_object[:, 2] -= expand_m
        bb3_object[:, 3] += expand_m
        bb3_object[:, 4] -= expand_m
        bb3_object[:, 5] += expand_m
        obbs.set_bb3_object(bb3_object)

        obb_colors = obbs.color.cpu().numpy().astype(np.float32)
        if obb_colors.ndim == 1:
            obb_colors = obb_colors.reshape(1, 3)
        if obb_colors.shape[0] != len(obbs) or np.any(obb_colors < 0):
            self._fs_points_in_obbs = 0
            return base_colors

        out = base_colors.copy()
        pts_world_t = torch.from_numpy(pts_world.astype(np.float32))
        scores_np = self._latest_scores_3d.cpu().numpy().astype(np.float32)
        order = np.argsort(scores_np)
        covered = np.zeros((len(pts_world),), dtype=bool)
        for idx in order:
            mask = (
                obbs[idx].points_inside_bb3(pts_world_t).cpu().numpy().astype(bool)
            )
            if np.any(mask):
                out[mask] = obb_colors[idx]
                covered |= mask
        self._fs_points_in_obbs = int(covered.sum())
        return out

    def _get_boxer_sdp_w(self) -> torch.Tensor:
        if (
            not self.use_fs_for_boxer_sdp
            or not self.enable_foundation_stereo
            or self._fs_boxer_pts_world is None
            or len(self._fs_boxer_pts_world) == 0
        ):
            return torch.zeros(0, 3, dtype=torch.float32)

        pts = self._fs_boxer_pts_world
        max_points = max(1, int(self.fs_boxer_max_points))
        if len(pts) > max_points:
            step = int(np.ceil(len(pts) / float(max_points)))
            pts = pts[::step]
        return torch.from_numpy(np.ascontiguousarray(pts.astype(np.float32)))

    def _maybe_print_fs_debug_stats(
        self,
        baseline: float,
        source_focal: float,
        rectified_focal: float,
        disparity: np.ndarray,
        z_valid: np.ndarray,
        pts_rect: np.ndarray,
    ) -> None:
        disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
        if disp_valid.size == 0 or z_valid.size == 0 or pts_rect.size == 0:
            return
        if self._processed != 1 and self._processed % 60 != 0:
            return

        disp_q = np.percentile(disp_valid, [10, 50, 90])
        depth_q = np.percentile(z_valid, [10, 50, 90])
        xyz_min = pts_rect.min(axis=0)
        xyz_max = pts_rect.max(axis=0)
        print(
            "==> fs_debug "
            f"baseline={baseline:.4f}m "
            f"f_src={source_focal:.2f}px "
            f"f_rect={rectified_focal:.2f}px "
            f"disp[p10,p50,p90]=[{disp_q[0]:.2f},{disp_q[1]:.2f},{disp_q[2]:.2f}]px "
            f"depth[p10,p50,p90]=[{depth_q[0]:.3f},{depth_q[1]:.3f},{depth_q[2]:.3f}]m "
            f"rect_xyz_min=[{xyz_min[0]:.3f},{xyz_min[1]:.3f},{xyz_min[2]:.3f}]m "
            f"rect_xyz_max=[{xyz_max[0]:.3f},{xyz_max[1]:.3f},{xyz_max[2]:.3f}]m",
            flush=True,
        )

    def _maybe_print_fs_debug_stats(
        self,
        baseline: float,
        source_focal: float,
        rectified_focal: float,
        disparity: np.ndarray,
        z_valid: np.ndarray,
        pts_rect: np.ndarray,
    ) -> None:
        disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
        if disp_valid.size == 0 or z_valid.size == 0 or pts_rect.size == 0:
            return
        if self._processed != 1 and self._processed % 60 != 0:
            return

        disp_q = np.percentile(disp_valid, [10, 50, 90])
        depth_q = np.percentile(z_valid, [10, 50, 90])
        xyz_min = pts_rect.min(axis=0)
        xyz_max = pts_rect.max(axis=0)
        print(
            "==> fs_debug "
            f"baseline={baseline:.4f}m "
            f"f_src={source_focal:.2f}px "
            f"f_rect={rectified_focal:.2f}px "
            f"disp[p10,p50,p90]=[{disp_q[0]:.2f},{disp_q[1]:.2f},{disp_q[2]:.2f}]px "
            f"depth[p10,p50,p90]=[{depth_q[0]:.3f},{depth_q[1]:.3f},{depth_q[2]:.3f}]m "
            f"rect_xyz_min=[{xyz_min[0]:.3f},{xyz_min[1]:.3f},{xyz_min[2]:.3f}]m "
            f"rect_xyz_max=[{xyz_max[0]:.3f},{xyz_max[1]:.3f},{xyz_max[2]:.3f}]m",
            flush=True,
        )

    def _maybe_convert_vio_world(self, T_world_device: np.ndarray) -> np.ndarray:
        if not self.vio_world_is_y_up:
            return T_world_device
        return self._zup_from_yup_matrix() @ T_world_device

    def _maybe_convert_obbs_world(self, obbs: ObbTW) -> ObbTW:
        if not self.vio_world_is_y_up or obbs is None or len(obbs) == 0:
            return obbs
        T_fix = PoseTW.from_matrix(
            torch.from_numpy(self._zup_from_yup_matrix()).float()
        )
        return obbs.transform(T_fix)

    def _load_foundation_stereo(self) -> None:
        ensure_projectaria_fs_repo_on_path()
        self.fs_runtime = FoundationStereoRuntime(
            self.fs_ckpt,
            self.fs_valid_iters,
            fs_impl=self.fs_impl,
            consistency=self.consistency,
            consistency_threshold=self.consistency_threshold,
        )

    def _make_fs_linear_calib(self, source_calib):
        from projectaria_tools.core import calibration
        from projectaria_tools.core.sophus import SE3

        params = source_calib.get_projection_params()
        src_w, src_h = source_calib.get_image_size()
        scale = min(self.fs_hw / float(src_w), self.fs_hw / float(src_h))
        focal = float(params[0]) * scale * 1.25
        linear_params = np.array(
            [focal, focal, self.fs_hw / 2.0, self.fs_hw / 2.0]
        )
        return calibration.CameraCalibration(
            source_calib.get_label() + f"-linear-{self.fs_hw}",
            calibration.CameraModelType.LINEAR,
            linear_params,
            SE3(),
            self.fs_hw,
            self.fs_hw,
            None,
            source_calib.get_max_solid_angle(),
            source_calib.get_serial_number(),
        )

    def _infer_fs_disparity(self, left_rect, right_rect):
        if self.fs_runtime is None:
            return None
        disparity = self.fs_runtime.infer(left_rect, right_rect)
        return self._filter_fs_disparity(disparity)

    def _filter_fs_disparity(self, disparity: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if disparity is None:
            return None
        k = int(self.fs_disparity_median)
        if k <= 1:
            return disparity
        if k % 2 == 0:
            k += 1
        valid = np.isfinite(disparity)
        if not np.any(valid):
            return disparity
        fill = float(np.nanmedian(disparity[valid]))
        disp = np.where(valid, disparity, fill).astype(np.float32)
        filtered = cv2.medianBlur(disp, k)
        return np.where(valid, filtered, np.nan).astype(np.float32)

    def _update_fs_geometry(
        self,
        pair_ts: int,
        T_world_rect: np.ndarray,
        linear,
        depth: np.ndarray,
        left_rect: np.ndarray,
    ) -> None:
        cam_t_world = T_world_rect[:3, 3].reshape(1, 3).astype(np.float32)
        self._fs_pose_tail.append((pair_ts, cam_t_world[0]))
        min_ts = pair_ts - 2_000_000_000
        self._fs_pose_tail = [
            (ts, p) for ts, p in self._fs_pose_tail if ts >= min_ts
        ]
        if len(self._fs_pose_tail) >= 2:
            segments = []
            for p0, p1 in zip(
                [p for _, p in self._fs_pose_tail[:-1]],
                [p for _, p in self._fs_pose_tail[1:]],
            ):
                color = np.array([0.1, 0.9, 1.0], dtype=np.float32)
                segments.append(
                    np.concatenate(
                        [p0.astype(np.float32), p1.astype(np.float32), color, [1.0]]
                    )
                )
            self._upload_line_data("fs_trail", np.asarray(segments, dtype=np.float32))
        else:
            self.fs_trail_count = 0

        _ = linear

    def _apply_follow_view(
        self,
        T_world_camera: np.ndarray,
        _T_world_device: Optional[np.ndarray] = None,
    ) -> None:
        origin_world = np.asarray(T_world_camera[:3, 3], dtype=np.float32)
        R_world_camera = np.asarray(T_world_camera[:3, :3], dtype=np.float32)
        forward_world = R_world_camera @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward_norm = float(np.linalg.norm(forward_world))
        if forward_norm > 1e-6:
            forward_world = forward_world / forward_norm
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        camera_world = (
            origin_world
            - forward_world * float(self.follow_back)
            + up_world * float(self.follow_up)
        )
        target_world = origin_world + forward_world * float(self.follow_lookahead)
        target_view = target_world.astype("f4")
        camera_view = camera_world.astype("f4")
        delta = camera_view - target_view
        dist = float(np.linalg.norm(delta))
        if dist < 1e-5:
            return

        azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
        elevation = float(np.degrees(np.arcsin(np.clip(delta[2] / dist, -1.0, 1.0))))

        blend = float(np.clip(self.follow_smoothing, 0.0, 1.0))
        if blend <= 0.0:
            self.camera_target = target_view
            self.camera_distance = dist
            self.camera_azimuth = azimuth
            self.camera_elevation = elevation
            return

        self.camera_target = (
            (1.0 - blend) * self.camera_target + blend * target_view
        ).astype("f4")
        self.camera_distance = (1.0 - blend) * float(self.camera_distance) + blend * dist
        self.camera_azimuth = (1.0 - blend) * float(self.camera_azimuth) + blend * azimuth
        self.camera_elevation = (
            (1.0 - blend) * float(self.camera_elevation) + blend * elevation
        )

    def _get_follow_pose(self) -> Optional[np.ndarray]:
        if self._last_T_world_rgb_cam is not None:
            return self._last_T_world_rgb_cam
        return self._fs_last_T_world_rect

    def _seed_free_orbit_from_follow_view(
        self,
        T_world_rect: np.ndarray,
        T_world_device: Optional[np.ndarray] = None,
    ) -> None:
        prev_blend = float(self.follow_smoothing)
        self.follow_smoothing = 0.0
        try:
            self._apply_follow_view(T_world_rect, T_world_device)
        finally:
            self.follow_smoothing = prev_blend

    def _maybe_print_fs_debug_stats(
        self,
        baseline: float,
        source_focal: float,
        rectified_focal: float,
        disparity: np.ndarray,
        depth: np.ndarray,
        z_valid: np.ndarray,
        pts_rect: np.ndarray,
        count_empty: bool = True,
    ) -> None:
        if not self.fs_debug_stats:
            return
        if self._fs_processed != 1 and self._fs_processed - self._fs_debug_last_print < 60:
            return

        disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
        if disp_valid.size == 0 or z_valid.size == 0 or pts_rect.size == 0:
            if count_empty:
                self._fs_debug_last_print = self._fs_processed
            return
        self._fs_debug_last_print = self._fs_processed

        disp_q = np.percentile(disp_valid, [10, 50, 90])
        depth_q = np.percentile(z_valid, [10, 50, 90])
        xyz_min = pts_rect.min(axis=0)
        xyz_max = pts_rect.max(axis=0)
        print(
            "==> fs_debug "
            f"baseline={baseline:.4f}m "
            f"f_src={source_focal:.2f}px "
            f"f_rect={rectified_focal:.2f}px "
            f"disp[p10,p50,p90]=[{disp_q[0]:.2f},{disp_q[1]:.2f},{disp_q[2]:.2f}]px "
            f"depth[p10,p50,p90]=[{depth_q[0]:.3f},{depth_q[1]:.3f},{depth_q[2]:.3f}]m "
            f"rect_xyz_min=[{xyz_min[0]:.3f},{xyz_min[1]:.3f},{xyz_min[2]:.3f}]m "
            f"rect_xyz_max=[{xyz_max[0]:.3f},{xyz_max[1]:.3f},{xyz_max[2]:.3f}]m",
            flush=True,
        )

    def _run_fs_disparity_async_job(self, left_rect: np.ndarray, right_rect: np.ndarray):
        t0 = time.perf_counter()
        disparity = self._infer_fs_disparity(left_rect, right_rect)
        return disparity, (time.perf_counter() - t0) * 1000.0

    def _apply_fs_disparity_result(
        self,
        meta: dict,
        disparity: np.ndarray,
        infer_ms: float,
        fs_bench: Optional[dict] = None,
    ) -> None:
        if disparity is None:
            return
        if fs_bench is None:
            fs_bench = {}
        fs_total_t0 = time.perf_counter()
        fs_last = fs_total_t0

        def fs_mark(name: str) -> None:
            nonlocal fs_last
            if not self._bench_enabled:
                return
            now = time.perf_counter()
            fs_bench[name] = (now - fs_last) * 1000.0
            fs_last = now

        pair_ts = int(meta["pair_ts"])
        pair_delta_ms = float(meta["pair_delta_ms"])
        T_left_right = meta["T_left_right"]
        T_device_left = meta["T_device_left"]
        T_world_device = meta["T_world_device"]
        T_world_device_raw = meta["T_world_device_raw"]
        R_left_rect = meta["R_left_rect"]
        linear = meta["linear"]
        left_rect = meta["left_rect"]
        left_calib = meta["left_calib"]
        source_focal = float(meta["source_focal"])
        baseline = float(np.linalg.norm(T_left_right.translation()))
        focal = float(linear.get_projection_params()[0])
        disparity_to_depth = meta["disparity_to_depth"]

        self._fs_pair_delta_ms = pair_delta_ms
        self._fs_infer_ms = float(infer_ms)
        self._push_timing("fsp", self._fs_infer_ms)
        depth = disparity_to_depth(disparity, baseline, focal)
        fs_mark("depth")
        self._maybe_print_fs_debug_stats(
            baseline=baseline,
            source_focal=source_focal,
            rectified_focal=focal,
            disparity=disparity,
            depth=depth,
            z_valid=depth[np.isfinite(depth) & (depth > 0.0)].astype(np.float32),
            pts_rect=np.zeros((0, 3), dtype=np.float32),
            count_empty=False,
        )
        h, w = depth.shape
        stride = max(1, int(self.fs_point_stride))
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[0:h:stride, 0:w:stride]
        valid = np.isfinite(z) & (z > 0.03) & (z < float(self.fs_max_depth))
        fs_mark("valid_mask")
        if not np.any(valid):
            if self.fs_debug_stats and (
                self._fs_processed == 0
                or self._fs_processed - self._fs_debug_last_print >= 30
            ):
                disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
                disp_finite = disparity[np.isfinite(disparity)]
                depth_valid = depth[np.isfinite(depth) & (depth > 0.0)]
                if disp_finite.size > 0:
                    disp_stats = (
                        f"disp_raw[min,mean,max]="
                        f"[{float(np.nanmin(disp_finite)):.4f},"
                        f"{float(np.nanmean(disp_finite)):.4f},"
                        f"{float(np.nanmax(disp_finite)):.4f}]"
                    )
                else:
                    disp_stats = "disp_raw=none"
                print(
                    f"==> fs_points_debug impl={self.fs_runtime.fs_impl} "
                    f"pair_delta_ms={self._fs_pair_delta_ms:.3f} "
                    f"disp_n={int(disp_valid.size)} depth_n={int(depth_valid.size)} "
                    f"{disp_stats} max_depth={self.fs_max_depth:.2f} valid_pts=0",
                    flush=True,
                )
                self._fs_debug_last_print = self._fs_processed
            self._fs_last_pair_ts = pair_ts
            return

        fx, fy, cx, cy = [float(v) for v in linear.get_projection_params()[:4]]
        x_cam = (xs[valid].astype(np.float32) - cx) * z[valid] / fx
        y_cam = (ys[valid].astype(np.float32) - cy) * z[valid] / fy
        z_cam = z[valid].astype(np.float32)
        pts_rect = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        self._maybe_print_fs_debug_stats(
            baseline=baseline,
            source_focal=source_focal,
            rectified_focal=focal,
            disparity=disparity,
            depth=depth,
            z_valid=z[valid].astype(np.float32),
            pts_rect=pts_rect,
        )
        R_left_rect_mat = np.asarray(R_left_rect.to_matrix(), dtype=np.float32)
        T_world_rect = T_world_device @ T_device_left
        T_world_rect[:3, :3] = T_world_rect[:3, :3] @ R_left_rect_mat
        pts_world = (T_world_rect[:3, :3] @ pts_rect.T + T_world_rect[:3, 3:4]).T
        T_world_rect_raw = T_world_device_raw @ T_device_left
        T_world_rect_raw[:3, :3] = T_world_rect_raw[:3, :3] @ R_left_rect_mat
        pts_world_raw = (
            T_world_rect_raw[:3, :3] @ pts_rect.T + T_world_rect_raw[:3, 3:4]
        ).T.astype(np.float32)
        fs_mark("points")
        self._fs_overlay_pts_world = pts_world_raw
        self._fs_overlay_depths = z[valid].astype(np.float32)
        self._fs_boxer_pts_world = pts_world_raw
        self._fs_boxer_pair_ts = pair_ts
        if self.fs_use_depth_colormap:
            colors = self._depth_jet_colors(z[valid], near=0.1, far=5.0)
        else:
            colors = np.ones((int(valid.sum()), 3), dtype="f4")
        colors = self._color_fs_points_from_obbs(
            pts_world.astype(np.float32), colors.astype(np.float32)
        )
        fs_mark("colors")
        data = np.concatenate([pts_world, colors], axis=1).astype("f4")
        data_bytes = data.tobytes()
        if self.fs_point_vbo is None or self.fs_point_vbo.size < len(data_bytes):
            if self.fs_point_vbo is not None:
                self.fs_point_vbo.release()
            if self.fs_point_vao is not None:
                self.fs_point_vao.release()
            self.fs_point_vbo = self.ctx.buffer(data_bytes)
            self.fs_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self.fs_point_vbo, "3f 3f", "in_position", "in_color")],
            )
        else:
            self.fs_point_vbo.orphan(self.fs_point_vbo.size)
            self.fs_point_vbo.write(data_bytes)
        fs_mark("upload_points")
        self.fs_point_count = len(data)
        self._fs_last_pair_ts = pair_ts
        z_valid = z[valid].astype(np.float32)
        self._fs_processed += 1
        if self.fs_debug_stats and (
            self._fs_processed == 1 or self._fs_processed % 30 == 0
        ):
            disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
            print(
                f"==> fs_points_debug impl={self.fs_runtime.fs_impl} "
                f"pair_delta_ms={self._fs_pair_delta_ms:.3f} "
                f"disp_n={int(disp_valid.size)} pts={int(self.fs_point_count)} "
                f"depth_range=[{float(np.nanmin(z_valid)):.3f},{float(np.nanmax(z_valid)):.3f}]m",
                flush=True,
            )
        self._fs_min_depth = float(np.nanmin(z_valid))
        self._fs_max_depth = float(np.nanmax(z_valid))
        self._fs_mean_depth = float(np.nanmean(z_valid))
        self._fs_median_depth = float(np.nanmedian(z_valid))
        self._fs_last_T_world_device = T_world_device
        self._fs_last_T_world_rect = T_world_rect
        fs_apply_t = time.perf_counter()
        self._fs_last_apply_t = fs_apply_t
        if "start_t" in meta:
            self._fs_last_pipeline_ms = (fs_apply_t - float(meta["start_t"])) * 1000.0
            self._push_timing("fs_pipe", self._fs_last_pipeline_ms)
        if not self._fs_target_inited:
            self._seed_free_orbit_from_follow_view(T_world_rect, T_world_device)
            self._rebuild_world_axes(self.camera_target)
            self._fs_target_inited = True
            self._target_inited = True
        self._update_fs_geometry(pair_ts, T_world_rect, linear, depth, left_rect)
        fs_mark("geometry")
        if self._bench_enabled:
            fs_bench["infer_async"] = float(infer_ms)
            fs_bench["apply_total"] = (time.perf_counter() - fs_total_t0) * 1000.0
            if self._bench_should_print(self._fs_processed, fs_bench["apply_total"]):
                print(
                    "==> bench_fs "
                    f"frame={self._fs_processed} pair_delta={self._fs_pair_delta_ms:.3f}ms "
                    f"pts={self.fs_point_count} async=1 "
                    + self._bench_fmt(fs_bench),
                    flush=True,
                )

    def _maybe_update_fs_scene_async(self) -> None:
        if not self.enable_foundation_stereo:
            self.fs_point_count = 0
            self.fs_trail_count = 0
            self._fs_overlay_pts_world = None
            self._fs_overlay_depths = None
            self._fs_min_depth = float("nan")
            self._fs_max_depth = float("nan")
            self._fs_mean_depth = float("nan")
            self._fs_median_depth = float("nan")
            self._fs_infer_ms = 0.0
            return
        if self.fs_state is None or self.fs_runtime is None or self._fs_executor is None:
            return

        if self._fs_future is not None and self._fs_future.done():
            future = self._fs_future
            meta = self._fs_pending_meta
            self._fs_future = None
            self._fs_pending_meta = None
            self._fs_pending_pair_ts = -1
            try:
                disparity, infer_ms = future.result()
            except Exception as exc:
                print(f"==> bench_fs async inference failed: {exc}", flush=True)
                return
            if meta is not None:
                self._apply_fs_disparity_result(
                    meta,
                    disparity,
                    float(infer_ms),
                    {
                        "queue_wait": (time.perf_counter() - meta["submit_t"]) * 1000.0,
                    },
                )

        if self._fs_future is not None:
            return

        fs_total_t0 = time.perf_counter()
        fs_last = fs_total_t0
        fs_bench = {}

        def fs_mark(name: str) -> None:
            nonlocal fs_last
            if not self._bench_enabled:
                return
            now = time.perf_counter()
            fs_bench[name] = (now - fs_last) * 1000.0
            fs_last = now

        fs_snapshot = self.fs_state.snapshot()
        fs_mark("snapshot")
        left_frame, right_frame, left_calib, right_calib, T_world_device = fs_snapshot
        if (
            left_frame is None
            or right_frame is None
            or left_calib is None
            or right_calib is None
            or T_world_device is None
        ):
            return

        left_img, left_ts = left_frame
        right_img, right_ts = right_frame
        pair_ts = max(left_ts, right_ts)
        if pair_ts == self._fs_last_pair_ts or pair_ts == self._fs_pending_pair_ts:
            return
        pair_delta_ms = abs(left_ts - right_ts) / 1e6
        if pair_delta_ms > 2.0:
            return
        if pair_ts != self._fs_last_seen_pair_ts:
            self._fs_last_seen_pair_ts = int(pair_ts)
            self._fs_pair_seen += 1
        if (self._fs_pair_seen - 1) % max(1, int(self.fsp_every)) != 0:
            self._fs_last_pair_ts = int(pair_ts)
            return

        T_world_device_raw = np.asarray(T_world_device, dtype=np.float32).copy()
        T_world_device = self._maybe_convert_vio_world(T_world_device_raw.copy())
        T_left_device = left_calib.get_transform_device_camera().inverse()
        T_right_device = right_calib.get_transform_device_camera().inverse()
        T_left_right = T_left_device @ T_right_device.inverse()
        T_device_left = np.asarray(T_left_device.inverse().to_matrix(), dtype=np.float32)
        fs_mark("pose_calib")
        ensure_projectaria_fs_repo_on_path()
        from projectaria_tools.core.image import InterpolationMethod
        from stereo_utils import (
            create_scanline_rectified_cameras,
            disparity_to_depth,
            rectify_stereo_pair,
        )
        fs_mark("imports")

        R_left_rect, R_right_rect = create_scanline_rectified_cameras(
            T_left_device, T_right_device
        )
        linear = self._make_fs_linear_calib(left_calib)
        left_rect, right_rect = rectify_stereo_pair(
            left_img,
            right_img,
            left_calib,
            right_calib,
            linear,
            linear,
            R_left_rect,
            R_right_rect,
            interpolation=InterpolationMethod.BILINEAR,
        )
        fs_mark("rectify")
        submit_t = time.perf_counter()
        self._fs_pending_pair_ts = int(pair_ts)
        self._fs_pending_meta = {
            "pair_ts": int(pair_ts),
            "pair_delta_ms": float(pair_delta_ms),
            "T_left_right": T_left_right,
            "T_device_left": T_device_left,
            "T_world_device": T_world_device,
            "T_world_device_raw": T_world_device_raw,
            "R_left_rect": R_left_rect,
            "linear": linear,
            "left_rect": left_rect,
            "left_calib": left_calib,
            "source_focal": float(left_calib.get_projection_params()[0]),
            "disparity_to_depth": disparity_to_depth,
            "start_t": fs_total_t0,
            "submit_t": submit_t,
        }
        self._fs_future = self._fs_executor.submit(
            self._run_fs_disparity_async_job, left_rect, right_rect
        )
        fs_mark("submit")
        if self._bench_enabled:
            fs_bench["submit_total"] = (time.perf_counter() - fs_total_t0) * 1000.0
            if self._bench_should_print(self._fs_processed + 1, fs_bench["submit_total"]):
                print(
                    "==> bench_fs_submit "
                    f"next_frame={self._fs_processed + 1} pair_delta={pair_delta_ms:.3f}ms "
                    + self._bench_fmt(fs_bench),
                    flush=True,
                )

    def _maybe_update_fs_scene(self) -> None:
        if self._fs_async_enabled:
            self._maybe_update_fs_scene_async()
            return

        fs_total_t0 = time.perf_counter()
        fs_last = fs_total_t0
        fs_bench = {}

        def fs_mark(name: str) -> None:
            nonlocal fs_last
            if not self._bench_enabled:
                return
            now = time.perf_counter()
            fs_bench[name] = (now - fs_last) * 1000.0
            fs_last = now

        if not self.enable_foundation_stereo:
            self.fs_point_count = 0
            self.fs_trail_count = 0
            self._fs_overlay_pts_world = None
            self._fs_overlay_depths = None
            self._fs_min_depth = float("nan")
            self._fs_max_depth = float("nan")
            self._fs_mean_depth = float("nan")
            self._fs_median_depth = float("nan")
            self._fs_infer_ms = 0.0
            return
        if self.fs_state is None or self.fs_runtime is None:
            return
        fs_snapshot = self.fs_state.snapshot()
        fs_mark("snapshot")
        left_frame, right_frame, left_calib, right_calib, T_world_device = fs_snapshot
        if (
            left_frame is None
            or right_frame is None
            or left_calib is None
            or right_calib is None
            or T_world_device is None
        ):
            return

        left_img, left_ts = left_frame
        right_img, right_ts = right_frame
        pair_ts = max(left_ts, right_ts)
        if pair_ts == self._fs_last_pair_ts:
            return
        self._fs_pair_delta_ms = abs(left_ts - right_ts) / 1e6
        if self._fs_pair_delta_ms > 2.0:
            return
        if pair_ts != self._fs_last_seen_pair_ts:
            self._fs_last_seen_pair_ts = int(pair_ts)
            self._fs_pair_seen += 1
        if (self._fs_pair_seen - 1) % max(1, int(self.fsp_every)) != 0:
            self._fs_last_pair_ts = int(pair_ts)
            return

        T_world_device_raw = np.asarray(T_world_device, dtype=np.float32).copy()
        T_world_device = self._maybe_convert_vio_world(T_world_device_raw.copy())
        T_left_device = left_calib.get_transform_device_camera().inverse()
        T_right_device = right_calib.get_transform_device_camera().inverse()
        T_left_right = T_left_device @ T_right_device.inverse()
        T_device_left = np.asarray(T_left_device.inverse().to_matrix(), dtype=np.float32)
        fs_mark("pose_calib")
        ensure_projectaria_fs_repo_on_path()
        from projectaria_tools.core.image import InterpolationMethod
        from stereo_utils import (
            create_scanline_rectified_cameras,
            disparity_to_depth,
            rectify_stereo_pair,
        )
        fs_mark("imports")

        R_left_rect, R_right_rect = create_scanline_rectified_cameras(
            T_left_device, T_right_device
        )
        linear = self._make_fs_linear_calib(left_calib)
        left_rect, right_rect = rectify_stereo_pair(
            left_img,
            right_img,
            left_calib,
            right_calib,
            linear,
            linear,
            R_left_rect,
            R_right_rect,
            interpolation=InterpolationMethod.BILINEAR,
        )
        fs_mark("rectify")

        t_infer = time.time()
        disparity = self._infer_fs_disparity(left_rect, right_rect)
        self._fs_infer_ms = (time.time() - t_infer) * 1000.0
        self._push_timing("fsp", self._fs_infer_ms)
        fs_mark("infer")
        if disparity is None:
            return

        baseline = float(np.linalg.norm(T_left_right.translation()))
        source_focal = float(left_calib.get_projection_params()[0])
        focal = float(linear.get_projection_params()[0])
        depth = disparity_to_depth(disparity, baseline, focal)
        fs_mark("depth")
        self._maybe_print_fs_debug_stats(
            baseline=baseline,
            source_focal=source_focal,
            rectified_focal=focal,
            disparity=disparity,
            depth=depth,
            z_valid=depth[np.isfinite(depth) & (depth > 0.0)].astype(np.float32),
            pts_rect=np.zeros((0, 3), dtype=np.float32),
            count_empty=False,
        )
        h, w = depth.shape
        stride = max(1, int(self.fs_point_stride))
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[0:h:stride, 0:w:stride]
        valid = np.isfinite(z) & (z > 0.03) & (z < float(self.fs_max_depth))
        fs_mark("valid_mask")
        if not np.any(valid):
            if self.fs_debug_stats and (
                self._fs_processed == 0
                or self._fs_processed - self._fs_debug_last_print >= 30
            ):
                disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
                disp_finite = disparity[np.isfinite(disparity)]
                depth_valid = depth[np.isfinite(depth) & (depth > 0.0)]
                if disp_finite.size > 0:
                    disp_stats = (
                        f"disp_raw[min,mean,max]="
                        f"[{float(np.nanmin(disp_finite)):.4f},"
                        f"{float(np.nanmean(disp_finite)):.4f},"
                        f"{float(np.nanmax(disp_finite)):.4f}]"
                    )
                else:
                    disp_stats = "disp_raw=none"
                print(
                    f"==> fs_points_debug impl={self.fs_runtime.fs_impl} "
                    f"pair_delta_ms={self._fs_pair_delta_ms:.3f} "
                    f"disp_n={int(disp_valid.size)} depth_n={int(depth_valid.size)} "
                    f"{disp_stats} max_depth={self.fs_max_depth:.2f} valid_pts=0",
                    flush=True,
                )
                self._fs_debug_last_print = self._fs_processed
            return

        fx, fy, cx, cy = [float(v) for v in linear.get_projection_params()[:4]]
        x_cam = (xs[valid].astype(np.float32) - cx) * z[valid] / fx
        y_cam = (ys[valid].astype(np.float32) - cy) * z[valid] / fy
        z_cam = z[valid].astype(np.float32)
        pts_rect = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        self._maybe_print_fs_debug_stats(
            baseline=baseline,
            source_focal=source_focal,
            rectified_focal=focal,
            disparity=disparity,
            depth=depth,
            z_valid=z[valid].astype(np.float32),
            pts_rect=pts_rect,
        )
        R_left_rect_mat = np.asarray(R_left_rect.to_matrix(), dtype=np.float32)
        T_world_rect = T_world_device @ T_device_left
        T_world_rect[:3, :3] = T_world_rect[:3, :3] @ R_left_rect_mat
        pts_world = (T_world_rect[:3, :3] @ pts_rect.T + T_world_rect[:3, 3:4]).T
        T_world_rect_raw = T_world_device_raw @ T_device_left
        T_world_rect_raw[:3, :3] = T_world_rect_raw[:3, :3] @ R_left_rect_mat
        pts_world_raw = (
            T_world_rect_raw[:3, :3] @ pts_rect.T + T_world_rect_raw[:3, 3:4]
        ).T.astype(np.float32)
        fs_mark("points")
        self._fs_overlay_pts_world = pts_world_raw
        self._fs_overlay_depths = z[valid].astype(np.float32)
        self._fs_boxer_pts_world = pts_world_raw
        self._fs_boxer_pair_ts = pair_ts
        if self.fs_use_depth_colormap:
            colors = self._depth_jet_colors(z[valid], near=0.1, far=5.0)
        else:
            colors = np.ones((int(valid.sum()), 3), dtype="f4")
        colors = self._color_fs_points_from_obbs(
            pts_world.astype(np.float32), colors.astype(np.float32)
        )
        fs_mark("colors")
        data = np.concatenate([pts_world, colors], axis=1).astype("f4")
        data_bytes = data.tobytes()
        if self.fs_point_vbo is None or self.fs_point_vbo.size < len(data_bytes):
            if self.fs_point_vbo is not None:
                self.fs_point_vbo.release()
            if self.fs_point_vao is not None:
                self.fs_point_vao.release()
            self.fs_point_vbo = self.ctx.buffer(data_bytes)
            self.fs_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self.fs_point_vbo, "3f 3f", "in_position", "in_color")],
            )
        else:
            self.fs_point_vbo.orphan(self.fs_point_vbo.size)
            self.fs_point_vbo.write(data_bytes)
        fs_mark("upload_points")
        self.fs_point_count = len(data)
        self._fs_last_pair_ts = pair_ts
        z_valid = z[valid].astype(np.float32)
        self._fs_processed += 1
        if self.fs_debug_stats and (
            self._fs_processed == 1 or self._fs_processed % 30 == 0
        ):
            disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
            print(
                f"==> fs_points_debug impl={self.fs_runtime.fs_impl} "
                f"pair_delta_ms={self._fs_pair_delta_ms:.3f} "
                f"disp_n={int(disp_valid.size)} pts={int(self.fs_point_count)} "
                f"depth_range=[{float(np.nanmin(z_valid)):.3f},{float(np.nanmax(z_valid)):.3f}]m",
                flush=True,
            )
        self._fs_min_depth = float(np.nanmin(z_valid))
        self._fs_max_depth = float(np.nanmax(z_valid))
        self._fs_mean_depth = float(np.nanmean(z_valid))
        self._fs_median_depth = float(np.nanmedian(z_valid))
        self._fs_last_T_world_device = T_world_device
        self._fs_last_T_world_rect = T_world_rect
        fs_apply_t = time.perf_counter()
        self._fs_last_apply_t = fs_apply_t
        self._fs_last_pipeline_ms = (fs_apply_t - fs_total_t0) * 1000.0
        self._push_timing("fs_pipe", self._fs_last_pipeline_ms)
        if not self._fs_target_inited:
            self._seed_free_orbit_from_follow_view(T_world_rect, T_world_device)
            self._rebuild_world_axes(self.camera_target)
            self._fs_target_inited = True
            self._target_inited = True
        self._update_fs_geometry(pair_ts, T_world_rect, linear, depth, left_rect)
        fs_mark("geometry")
        if self._bench_enabled:
            fs_bench["total"] = (time.perf_counter() - fs_total_t0) * 1000.0
            if self._bench_should_print(self._fs_processed, fs_bench["total"]):
                print(
                    "==> bench_fs "
                    f"frame={self._fs_processed} pair_delta={self._fs_pair_delta_ms:.3f}ms "
                    f"pts={self.fs_point_count} "
                    + self._bench_fmt(fs_bench),
                    flush=True,
                )

    def _start_recording(self) -> None:
        self._record_dir = tempfile.mkdtemp(prefix="live_boxer_rec_")
        self._record_frame_idx = 0
        self._recording = True
        self._last_record_mp4 = None
        print(f"[REC] Live recording started - frames -> {self._record_dir}", flush=True)

    def _stop_recording(self) -> None:
        self._recording = False
        if self._record_dir is None:
            return
        n = int(self._record_frame_idx)
        print(f"[REC] Live recording stopped - {n} frames captured", flush=True)
        if n > 0:
            output_dir = os.path.expanduser("~/Desktop")
            os.makedirs(output_dir, exist_ok=True)
            base_name = time.strftime("live_boxer_%Y%m%d_%H%M%S")
            output_name = f"{base_name}.mp4"
            out_path = os.path.join(output_dir, output_name)
            suffix = 1
            while os.path.exists(out_path):
                output_name = f"{base_name}_{suffix}.mp4"
                out_path = os.path.join(output_dir, output_name)
                suffix += 1
            self._last_record_mp4 = make_mp4(
                self._record_dir,
                framerate=int(round(self._record_fps)),
                output_dir=output_dir,
                output_name=output_name,
                crf=14,
                preset="slow",
            )
            print(f"[REC] Live video saved to {self._last_record_mp4}", flush=True)
        self._record_dir = None
        self._record_frame_idx = 0

    def _capture_recording_frame(self) -> None:
        if not self._recording or self._record_dir is None:
            return
        t0 = time.perf_counter()
        win_w, win_h = self.wnd.size
        x0 = int(round(self.ui_panel_width))
        capture_w = int(round(win_w - self.ui_panel_width))
        if capture_w <= 0 or win_h <= 0:
            return
        data = self.ctx.screen.read(viewport=(x0, 0, capture_w, win_h), components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(win_h, capture_w, 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (max(1, capture_w // 2), max(1, win_h // 2)))
        path = os.path.join(self._record_dir, f"image_{self._record_frame_idx:05d}.png")
        cv2.imwrite(path, img)
        self._record_frame_idx += 1
        self._render_only_ms = max(
            0.0, self._render_only_ms - (time.perf_counter() - t0) * 1000.0
        )

    def _maybe_capture_ui_debug_frame(self) -> None:
        if (
            self._ui_capture_done
            or not self._ui_capture_path
            or self._render_steps < self._ui_capture_frame
        ):
            return
        t0 = time.perf_counter()
        win_w, win_h = self.wnd.size
        if win_w <= 0 or win_h <= 0:
            return
        data = self.ctx.screen.read(viewport=(0, 0, win_w, win_h), components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(win_h, win_w, 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_path = os.path.abspath(os.path.expanduser(self._ui_capture_path))
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(out_path, img)
        self._ui_capture_done = True
        if ok:
            print(f"[UI_CAPTURE] wrote {out_path} ({win_w}x{win_h})", flush=True)
        else:
            print(f"[UI_CAPTURE] failed to write {out_path}", flush=True)
        self._render_only_ms = max(
            0.0, self._render_only_ms - (time.perf_counter() - t0) * 1000.0
        )

    # -- render --

    def on_render(self, time_val: float, frame_time: float):
        loop_t0 = time.perf_counter()
        self._total_frame_ms = (loop_t0 - self._last_loop_t0) * 1000.0
        self._last_loop_t0 = loop_t0
        fs_update_ms = 0.0
        rgb_update_ms = 0.0
        self._render_steps += 1
        if self.max_steps > 0 and self._render_steps > int(self.max_steps):
            if self._recording:
                self._stop_recording()
            self._shutdown_fs_executor()
            self.wnd.close()
            return
        render_t0 = time.perf_counter()
        super().on_render(time_val, frame_time)
        self._render_only_ms = (time.perf_counter() - render_t0) * 1000.0
        self._capture_recording_frame()
        self._maybe_capture_ui_debug_frame()
        self._push_timing("render", self._render_only_ms)
        if self._prediction_pending_t0 is not None:
            prediction_done_t = time.perf_counter()
            self._prediction_e2e_ms = (
                prediction_done_t - self._prediction_pending_t0
            ) * 1000.0
            self._push_timing("pred_latency", self._prediction_e2e_ms)
            self._prediction_pending_t0 = None
            if self._last_prediction_done_t is not None:
                self._prediction_period_ms = (
                    prediction_done_t - self._last_prediction_done_t
                ) * 1000.0
                self._push_timing("pred_period", self._prediction_period_ms)
            self._last_prediction_done_t = prediction_done_t
            self._prediction_count += 1
            pred_now = prediction_done_t
            if pred_now - self._prediction_count_t0 >= 1.0:
                self._prediction_fps = self._prediction_count / (
                    pred_now - self._prediction_count_t0
                )
                self._prediction_count = 0
                self._prediction_count_t0 = pred_now

        ui_busy = bool(self._resize_drag_active or self._ui_interaction_active)
        if not ui_busy:
            t_fs = time.perf_counter()
            self._maybe_update_fs_scene()
            fs_update_ms = (time.perf_counter() - t_fs) * 1000.0
            t_rgb = time.perf_counter()
            self._maybe_run_inference()
            rgb_update_ms = (time.perf_counter() - t_rgb) * 1000.0
        if not self.enable_owl:
            self._prediction_fps = 0.0
        loop_body_ms = (time.perf_counter() - loop_t0) * 1000.0
        self._fs_update_ms = fs_update_ms
        self._rgb_update_ms = rgb_update_ms
        loop_measured_sum_ms = self._render_only_ms + fs_update_ms + rgb_update_ms
        loop_overhead_ms = loop_body_ms - loop_measured_sum_ms
        loop_component_sum_ms = loop_measured_sum_ms + loop_overhead_ms
        self._loop_timing_sum_ms = loop_component_sum_ms
        self._loop_timing_actual_ms = loop_body_ms
        self._loop_timing_gap_ms = loop_body_ms - loop_component_sum_ms
        self._loop_timing_overhead_ms = loop_overhead_ms
        self._push_timing("loop_sum", self._loop_timing_sum_ms)
        self._push_timing("loop_actual", self._loop_timing_actual_ms)
        self._push_timing("loop_gap", self._loop_timing_gap_ms)
        self._push_timing("loop_overhead", self._loop_timing_overhead_ms)

        self._frame_count += 1
        self._bench_loop_idx += 1
        now = time.perf_counter()
        if now - self._frame_count_t0 >= 1.0:
            self._fps = self._frame_count / (now - self._frame_count_t0)
            self._frame_count = 0
            self._frame_count_t0 = now
        if self._bench_should_print(self._bench_loop_idx, self._total_frame_ms):
            self._bench_last_loop = {
                "period": self._total_frame_ms,
                "body": loop_body_ms,
                "fs_update": fs_update_ms,
                "rgb_update": rgb_update_ms,
                "draw": self._render_only_ms,
                "overhead": loop_overhead_ms,
                "sum": loop_component_sum_ms,
                "actual": loop_body_ms,
                "gap": loop_body_ms - loop_component_sum_ms,
                "pred_e2e": self._prediction_e2e_ms,
                "pred_period": self._prediction_period_ms,
            }
            print(
                "==> bench_loop "
                f"frame={self._bench_loop_idx} render_fps={self._fps:.1f} "
                f"pred_fps={self._prediction_fps:.1f} "
                f"resize_drag={int(self._resize_drag_active)} "
                f"ui_busy={int(ui_busy)} "
                + self._bench_fmt(self._bench_last_loop),
                flush=True,
            )

    def render_3d(self, time_val: float, frame_time: float) -> None:
        full_w, full_h = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = self._get_3d_viewport_origin_x()
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)
        # Clear just the right viewport so the rest of the window stays clean
        bg = self.bg_color_options[self.bg_color_index]
        self.ctx.clear(*bg)

        if self.follow_mode:
            follow_pose = self._get_follow_pose()
            if follow_pose is not None:
                self._apply_follow_view(follow_pose)

        _, _, mvp = self.get_camera_matrices()
        mvp_bytes = np.array(mvp, dtype="f4").tobytes()
        viewport = np.array([w, h], dtype="f4")

        if (
            self.show_obbs_3d
            and self._obb_vao is not None
            and self._obb_count > 0
        ):
            raw_obb_alpha = 0.25 if self.enable_tracker else 1.0
            raw_obb_line_width = (
                max(1.0, float(self.line_width) * 0.65)
                if self.enable_tracker
                else float(self.line_width)
            )
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(raw_obb_line_width, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(raw_obb_alpha, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._obb_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._obb_count
            )

        if (
            self.show_obbs_3d
            and self.enable_tracker
            and self._tracked_obb_vao is not None
            and self._tracked_obb_count > 0
        ):
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(self.tracker_line_width, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._tracked_obb_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._tracked_obb_count
            )
            self.ctx.enable(self.ctx.DEPTH_TEST)

        if (
            self.show_obbs_3d
            and self.enable_tracker
            and self.show_track_assoc_lines
            and self._match_line_vao is not None
            and self._match_line_count > 0
        ):
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(max(3.0, self.tracker_line_width * 0.65), dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(0.95, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._match_line_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._match_line_count
            )
            self.ctx.enable(self.ctx.DEPTH_TEST)

        if (
            self.show_frustum
            and self._frustum_vao is not None
            and self._frustum_count > 0
        ):
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(self.frustum_line_width, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._frustum_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._frustum_count
            )

        if (
            self.show_world_axes
            and self._axis_vao is not None
            and self._axis_count > 0
        ):
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(self.axis_line_width, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._axis_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._axis_count
            )

        if (
            self.show_fs_points
            and self.fs_point_vao is not None
            and self.fs_point_count > 0
        ):
            self.point_prog["mvp"].write(mvp_bytes)
            self.point_prog["point_size"].write(
                np.array(self.fs_point_size, dtype="f4").tobytes()
            )
            self.point_prog["alpha"].write(
                np.array(self.fs_point_alpha, dtype="f4").tobytes()
            )
            self.fs_point_vao.render(
                mode=self.ctx.POINTS, vertices=self.fs_point_count
            )

        if (
            self.show_fs_trajectory
            and self.fs_trail_vao is not None
            and self.fs_trail_count > 0
        ):
            self.ctx.line_width = float(self.fs_line_width)
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self.fs_trail_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.fs_trail_count
            )

        # Restore full viewport for ImGui
        self.ctx.viewport = (0, 0, full_w, full_h)
        self.ctx.scissor = None

    def render_ui(self) -> None:
        self._clamp_panel_widths()
        ui_control_active = False
        prev_drag_active = bool(self._resize_drag_active)
        prev_ui_active = bool(self._ui_interaction_active)
        if not (prev_drag_active or prev_ui_active):
            self._resize_ui_panel_width = float(self.ui_panel_width)
            self._resize_rgb_panel_width = float(self.rgb_panel_width)
            self._resize_viz_panel_width = float(self.viz_panel_width)
        (
            self._resize_ui_panel_width,
            self._resize_rgb_panel_width,
            self._resize_viz_panel_width,
        ) = self._clamp_panel_width_values(
            self._resize_ui_panel_width,
            self._resize_rgb_panel_width,
            self._resize_viz_panel_width,
        )
        self._resize_drag_active = False
        win_w, win_h = self.wnd.size
        ui_panel_width = float(self._resize_ui_panel_width)
        rgb_panel_width = float(self._resize_rgb_panel_width)
        viz_panel_width = float(self._resize_viz_panel_width)
        prompt_bar_h = float(type(self).prompt_bar_height)
        splitter_w = 14.0
        splitter_bar_w = 6.0

        def render_splitter(name: str, center_x: float, on_drag):
            nonlocal ui_control_active
            x0 = int(round(center_x - splitter_w * 0.5))
            splitter_h = max(1, int(round(float(win_h) - prompt_bar_h)))
            imgui.set_next_window_position(x0, 0, imgui.ALWAYS)
            imgui.set_next_window_size(int(splitter_w), splitter_h, imgui.ALWAYS)
            flags = (
                imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_SCROLLBAR
                | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
            )
            imgui.push_style_color(imgui.COLOR_WINDOW_BG, 0.0, 0.0, 0.0, 0.0)
            imgui.begin(name, flags=flags)
            draw_list = imgui.get_window_draw_list()
            win_pos = imgui.get_window_position()
            imgui.set_cursor_pos((0.0, 0.0))
            imgui.invisible_button(
                f"##{name}_drag", imgui.ImVec2(float(splitter_w), float(splitter_h))
            )
            hovered = imgui.is_item_hovered()
            active = imgui.is_item_active()
            active_col = imgui.get_color_u32_rgba(0.92, 0.92, 0.92, 0.95)
            idle_col = imgui.get_color_u32_rgba(0.62, 0.62, 0.62, 0.85)
            col = active_col if hovered or active else idle_col
            bar_x0 = win_pos.x + 0.5 * (splitter_w - splitter_bar_w)
            bar_x1 = bar_x0 + splitter_bar_w
            draw_list.add_rect_filled(
                bar_x0,
                win_pos.y,
                bar_x1,
                win_pos.y + splitter_h,
                col,
            )
            if active:
                ui_control_active = True
                self._resize_drag_active = True
                dx = float(imgui.get_io().mouse_delta.x)
                if abs(dx) > 0.0:
                    on_drag(dx)
                    (
                        self._resize_ui_panel_width,
                        self._resize_rgb_panel_width,
                        self._resize_viz_panel_width,
                    ) = self._clamp_panel_width_values(
                        self._resize_ui_panel_width,
                        self._resize_rgb_panel_width,
                        self._resize_viz_panel_width,
                    )
            imgui.end()
            imgui.pop_style_color()

        # Left: control panel
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(int(ui_panel_width), win_h, imgui.ALWAYS)
        imgui.begin(
            "Live BoxerNet Controls",
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
        )
        def ui_section(label: str, default_open: bool = False) -> bool:
            flags = int(imgui.TreeNodeFlags_.default_open) if default_open else 0
            result = imgui.collapsing_header(label, flags=flags)
            if isinstance(result, tuple):
                return bool(result[0])
            return bool(result)

        label_w = max(110.0, min(180.0, ui_panel_width * 0.42))
        slider_w = max(110.0, float(ui_panel_width) - label_w - 36.0)

        def labeled_slider_float(label, value, min_value, max_value, fmt="%.3f"):
            nonlocal ui_control_active
            imgui.text(label)
            imgui.same_line(label_w)
            imgui.push_item_width(slider_w)
            changed, value = imgui.slider_float(
                f"##{label}", value, min_value, max_value, fmt
            )
            ui_control_active = ui_control_active or bool(changed) or imgui.is_item_active()
            imgui.pop_item_width()
            return changed, value

        def labeled_slider_int(label, value, min_value, max_value):
            nonlocal ui_control_active
            imgui.text(label)
            imgui.same_line(label_w)
            imgui.push_item_width(slider_w)
            changed, value = imgui.slider_int(
                f"##{label}", value, min_value, max_value
            )
            ui_control_active = ui_control_active or bool(changed) or imgui.is_item_active()
            imgui.pop_item_width()
            return changed, value

        if ui_section("Default", default_open=True):
            _, self.thresh2d = labeled_slider_float(
                "2D thr", self.thresh2d, 0.0, 1.0
            )
            _, self.thresh3d = labeled_slider_float(
                "3D thr", self.thresh3d, 0.0, 1.0
            )
            rgb_hz = self.state.stream_hz() if self.state is not None else 0.0
            slam_hz = self.fs_state.stream_hz() if self.fs_state is not None else 0.0
            self._push_timing("rgb_hz", rgb_hz)
            self._push_timing("slam_hz", slam_hz)
            self._push_timing("pred_fps", self._prediction_fps)
            imgui.text(f"RGB Hz: {self._fmt_value_mean30('rgb_hz', rgb_hz)}")
            imgui.text(f"SLAM Hz: {self._fmt_value_mean30('slam_hz', slam_hz)}")
            imgui.text(
                f"Pred FPS: {self._fmt_value_mean30('pred_fps', self._prediction_fps)}"
            )
            imgui.text(
                f"Total: {self._fmt_ms_mean30('pred_latency', self._prediction_e2e_ms)}"
            )
            imgui.text(
                f"FSP: {self._fmt_ms_mean30('fsp', self._fs_infer_ms)}  every {self.fsp_every}"
            )
            imgui.text(f"OWL: {self._fmt_ms_mean30('owl', self._owl_ms)}")
            imgui.text(f"Boxer: {self._fmt_ms_mean30('boxer', self._boxer_ms)}")
            imgui.text(f"Tracker: {self._fmt_ms_mean30('tracker', self._tracker_ms)}")
            imgui.text(f"Render: {self._fmt_ms_mean30('render', self._render_only_ms)}")

        if ui_section("Advanced Timing"):
            self._push_timing("render_fps", self._fps)
            self._push_timing("fs_age", self._prediction_fs_age_ms)
            imgui.text(f"Render FPS: {self._fmt_value_mean30('render_fps', self._fps)}")
            imgui.text(
                f"Pred period: {self._fmt_ms_mean30('pred_period', self._prediction_period_ms)}"
            )
            imgui.text(
                f"FS age: {self._fmt_ms_mean30('fs_age', self._prediction_fs_age_ms)}"
            )
            imgui.text(
                f"FS pipe: {self._fmt_ms_mean30('fs_pipe', self._fs_last_pipeline_ms)}"
            )
            imgui.text(
                f"OWL render: {self._fmt_ms_mean30('owl_render', self._owl_render_ms)}"
            )
            imgui.text(
                f"Boxer render: {self._fmt_ms_mean30('boxer_render', self._boxer_render_ms)}"
            )
            imgui.text(
                f"RGB SUM: {self._fmt_ms_mean30('rgb_sum', self._rgb_timing_sum_ms)}  "
                f"ACTUAL: {self._fmt_ms_mean30('rgb_actual', self._rgb_timing_actual_ms)}"
            )
            imgui.text(
                f"RGB gap: {self._fmt_ms_mean30('rgb_gap', self._rgb_timing_gap_ms, signed=True)}  "
                f"overhead {self._fmt_ms_mean30('rgb_overhead', self._rgb_timing_overhead_ms)}"
            )
            imgui.text(
                f"LOOP SUM: {self._fmt_ms_mean30('loop_sum', self._loop_timing_sum_ms)}  "
                f"ACTUAL: {self._fmt_ms_mean30('loop_actual', self._loop_timing_actual_ms)}"
            )
            imgui.text(
                f"LOOP gap: {self._fmt_ms_mean30('loop_gap', self._loop_timing_gap_ms, signed=True)}  "
                f"overhead {self._fmt_ms_mean30('loop_overhead', self._loop_timing_overhead_ms)}"
            )
            imgui.text(f"2D detections: {self._n_2d}")
            imgui.text(f"3D detections: {self._n_3d}")
            imgui.text(f"Tracked 3DBBs: {self._n_tracks}")
            imgui.text(f"Track matches: {self._n_track_matches}")
            boxer_sdp_count = (
                0
                if self._fs_boxer_pts_world is None
                else min(len(self._fs_boxer_pts_world), int(self.fs_boxer_max_points))
            )
            imgui.text(f"Boxer SDP pts: {boxer_sdp_count}")
            imgui.text(
                f"Boxer SDP patches: {self._boxer_sdp_patch_valid}  median depth: {self._boxer_sdp_patch_median:.3f} m"
            )
        def ui_checkbox(label, value):
            nonlocal ui_control_active
            changed, value = imgui.checkbox(label, value)
            ui_control_active = ui_control_active or bool(changed) or imgui.is_item_active()
            return changed, value

        def ui_button(label):
            nonlocal ui_control_active
            clicked = imgui.button(label)
            ui_control_active = ui_control_active or bool(clicked) or imgui.is_item_active()
            return clicked

        def ui_input_text(label, value, buffer_length):
            nonlocal ui_control_active
            changed, value = imgui.input_text(label, value, buffer_length)
            ui_control_active = ui_control_active or bool(changed) or imgui.is_item_active()
            return changed, value

        def ui_input_text_multiline(label, value, width, height, flags=0):
            nonlocal ui_control_active
            changed, value = imgui.input_text_multiline(
                label, value, width, height, flags
            )
            ui_control_active = ui_control_active or bool(changed) or imgui.is_item_active()
            return changed, value

        if ui_section("Enable", default_open=True):
            if self.fs_state is not None:
                _, self.enable_foundation_stereo = ui_checkbox(
                    "Enable FS", self.enable_foundation_stereo
                )
            _, self.enable_owl = ui_checkbox("Enable OWL", self.enable_owl)
            boxer_enabled = self.enable_boxer
            _, boxer_enabled = ui_checkbox("Enable Boxer", boxer_enabled)
            self.enable_boxer = boxer_enabled and self.enable_owl
            if not self.enable_owl:
                imgui.text("Boxer requires OWL proposals")
            tracker_enabled = self.enable_tracker
            _, tracker_enabled = ui_checkbox("Enable Tracker", tracker_enabled)
            self.enable_tracker = tracker_enabled and self.enable_boxer and self.enable_owl
            if tracker_enabled and not self.enable_boxer:
                imgui.text("Tracker requires Boxer 3DBBs")
            if ui_button("Reset tracker"):
                self._reset_tracker_state()

        if ui_section("Detection"):
            _, self.owl_nms_iou = labeled_slider_float(
                "2D NMS", self.owl_nms_iou, 0.05, 1.0
            )
            _, self.use_fs_for_boxer_sdp = ui_checkbox(
                "Use FS points for Boxer SDP", self.use_fs_for_boxer_sdp
            )
            _, self.rectify_rgb_for_owl_boxes = ui_checkbox(
                "Rectify RGB for OWL", self.rectify_rgb_for_owl_boxes
            )
            _, self.bb3_use_class_colors = ui_checkbox(
                "3DBB class/prompt colors", self.bb3_use_class_colors
            )

        if ui_section("Boxer Model Picker"):
            if ui_button("Refresh models"):
                self.boxernet_ckpts = discover_boxernet_checkpoints(
                    self.current_boxernet_ckpt
                )
                self.boxernet_ckpt_index = min(
                    self.boxernet_ckpt_index, max(0, len(self.boxernet_ckpts) - 1)
                )
            if self.boxernet_ckpts:
                names = [_short_ckpt_name(path) for path in self.boxernet_ckpts]
                imgui.push_item_width(max(160.0, float(ui_panel_width) - 36.0))
                changed, self.boxernet_ckpt_index = imgui.combo(
                    "##boxernet_ckpt_picker",
                    int(self.boxernet_ckpt_index),
                    names,
                )
                ui_control_active = (
                    ui_control_active or bool(changed) or imgui.is_item_active()
                )
                imgui.pop_item_width()
                selected_ckpt = self.boxernet_ckpts[int(self.boxernet_ckpt_index)]
                imgui.text(_short_ckpt_name(selected_ckpt))
                if ui_button("Apply Boxer model"):
                    self._reload_boxernet_checkpoint(selected_ckpt)
            else:
                imgui.text("No ckpts/boxernet_* files found")
            imgui.text(self._boxernet_load_status)

        if ui_section("RGB Overlays"):
            _, self.show_rgb_fs_points = ui_checkbox(
                "Show FS Points", self.show_rgb_fs_points
            )
            _, self.show_rgb_fs = ui_checkbox("Show FS SDP", self.show_rgb_fs)
            _, self.show_rgb_owl = ui_checkbox("Show OWL", self.show_rgb_owl)
            _, self.show_rgb_boxer = ui_checkbox(
                "Show Boxer 3DBB", self.show_rgb_boxer
            )
            _, self.show_rgb_tracker = ui_checkbox(
                "Show Tracked 3DBB", self.show_rgb_tracker
            )
            _, self.split_rgb_overlays = ui_checkbox(
                "Separate RGB copies", self.split_rgb_overlays
            )
            _, self.bb2_line_width = labeled_slider_int(
                "2D lw", self.bb2_line_width, 1, 12
            )
            _, self.bb3_image_line_width = labeled_slider_int(
                "3D img lw", self.bb3_image_line_width, 1, 12
            )

        if ui_section("3D View"):
            _, self.show_obbs_3d = ui_checkbox("Show 3D OBBs", self.show_obbs_3d)
            _, self.show_raw_by_track_match = ui_checkbox(
                "Raw->track color", self.show_raw_by_track_match
            )
            _, self.show_track_assoc_lines = ui_checkbox(
                "Track assoc lines", self.show_track_assoc_lines
            )
            _, self.line_width = labeled_slider_float(
                "3D OBB lw", self.line_width, 1.0, 10.0
            )
            _, self.tracker_line_width = labeled_slider_float(
                "Track lw", self.tracker_line_width, 2.0, 16.0
            )
            _, self.show_frustum = ui_checkbox("Show frustum", self.show_frustum)
            _, self.show_world_axes = ui_checkbox("Show axes", self.show_world_axes)

        if self.fs_state is not None and ui_section("FoundationStereo"):
            _, self.show_fs_points = ui_checkbox("Show FS pts", self.show_fs_points)
            _, self.fs_use_depth_colormap = ui_checkbox(
                "FS jet colors", self.fs_use_depth_colormap
            )
            _, self.fs_color_points_by_obb = ui_checkbox(
                "FS by 3DBB", self.fs_color_points_by_obb
            )
            _, self.fs_point_size = labeled_slider_float(
                "FS pt sz", self.fs_point_size, 1.0, 8.0
            )
            _, self.fs_point_alpha = labeled_slider_float(
                "FS pt a", self.fs_point_alpha, 0.05, 1.0
            )
            follow_clicked = ui_button(
                "Follow view" if not self.follow_mode else "Free orbit"
            )
            if follow_clicked:
                self.follow_mode = not self.follow_mode
                if self.follow_mode:
                    follow_pose = self._get_follow_pose()
                    if follow_pose is not None:
                        self._apply_follow_view(follow_pose)
            _, self.follow_back = labeled_slider_float(
                "F back", self.follow_back, 0.05, 10.0
            )
            _, self.follow_up = labeled_slider_float(
                "F up", self.follow_up, 0.0, 10.0
            )
            _, self.follow_lookahead = labeled_slider_float(
                "Look", self.follow_lookahead, 0.0, 3.0
            )
            _, self.follow_smoothing = labeled_slider_float(
                "F smooth", self.follow_smoothing, 0.0, 1.0
            )

        if ui_section("Layout"):
            _, self._resize_ui_panel_width = labeled_slider_float(
                "UI W", self._resize_ui_panel_width, 260, 700, "%.0f"
            )
            _, self._resize_rgb_panel_width = labeled_slider_float(
                "RGB W", self._resize_rgb_panel_width, 320, 1500, "%.0f"
            )

        if ui_section("Recording"):
            if self._recording:
                if ui_button("Stop video"):
                    self._stop_recording()
                imgui.same_line()
                imgui.text(f"{self._record_frame_idx} frames")
            else:
                if ui_button("Record video"):
                    self._start_recording()
                if self._last_record_mp4:
                    imgui.text(f"Saved: {os.path.basename(self._last_record_mp4)}")
            imgui.text(f"Record FPS: {self._record_fps:.0f}")
        imgui.end()

        # Center-left: RGB + 2DBB overlay panel
        if self._rgb_texture is not None:
            tex_w, tex_h = self._rgb_tex_size
            rgb_x = int(ui_panel_width)
            imgui.set_next_window_position(rgb_x, 0, imgui.ALWAYS)
            imgui.set_next_window_size(int(rgb_panel_width), win_h, imgui.ALWAYS)
            expanded, _ = imgui.begin(
                "RGB",
                flags=imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
            )
            if expanded:
                avail_w, avail_h = imgui.get_content_region_available()
                scale = min(avail_w / tex_w, avail_h / tex_h)
                imgui.image(
                    self._rgb_texture.glo, tex_w * scale, tex_h * scale
                )
                if bool(type(self).rgb_gpu_overlays) and not bool(self.split_rgb_overlays):
                    img_min = imgui.get_item_rect_min()
                    draw_list = imgui.get_window_draw_list()
                    self._draw_rgb_gpu_overlays(
                        draw_list, img_min, tex_w * scale, tex_h * scale
            )
            imgui.end()

        prompt_bar_x = int(ui_panel_width)
        prompt_bar_w = max(1, int(win_w - prompt_bar_x))
        imgui.set_next_window_position(
            prompt_bar_x, int(max(0.0, win_h - prompt_bar_h)), imgui.ALWAYS
        )
        imgui.set_next_window_size(prompt_bar_w, int(prompt_bar_h), imgui.ALWAYS)
        imgui.begin(
            "Prompt Bar",
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_TITLE_BAR,
        )
        prompt_font_size = max(18.0, min(22.0, imgui.get_font_size() * 1.18))
        imgui.push_font(None, prompt_font_size)
        avail_w, avail_h = imgui.get_content_region_available()
        prompt_h = min(
            max(46.0, imgui.get_text_line_height_with_spacing() * 1.55),
            max(38.0, float(avail_h) - 18.0),
        )
        label_col_w = 216.0
        button_w = 126.0
        button_gap = 10.0
        status_text = f"{len(self.text_labels)} active"
        status_w = max(150.0, float(imgui.calc_text_size(status_text).x) + 28.0)
        controls_w = (button_w * 2.0) + (button_gap * 3.0) + status_w + 24.0
        prompt_w = max(160.0, float(avail_w) - label_col_w - controls_w)
        imgui.text("Queries")
        imgui.same_line(label_col_w)
        prompt_submitted, self.prompt_editor_text = ui_input_text_multiline(
            "##live_prompts_bottom",
            self.prompt_editor_text,
            prompt_w,
            prompt_h,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            | imgui.INPUT_TEXT_CTRL_ENTER_FOR_NEW_LINE,
        )
        prompt_active = imgui.is_item_active()
        prompt_enter_pressed = prompt_active and (
            imgui.is_key_pressed(imgui.Key.enter, False)
            or imgui.is_key_pressed(imgui.Key.keypad_enter, False)
        )
        imgui.same_line(0.0, button_gap)
        apply_clicked = imgui.button("Apply", button_w, prompt_h)
        ui_control_active = ui_control_active or bool(apply_clicked) or imgui.is_item_active()
        if apply_clicked or prompt_submitted or prompt_enter_pressed:
            self._apply_prompt_editor()
        imgui.same_line(0.0, button_gap)
        clear_clicked = imgui.button("Clear", button_w, prompt_h)
        ui_control_active = ui_control_active or bool(clear_clicked) or imgui.is_item_active()
        if clear_clicked:
            self.prompt_editor_text = ""
            self._apply_prompt_editor()
        imgui.same_line(0.0, button_gap)
        imgui.text(status_text)
        imgui.pop_font()
        imgui.end()

        render_splitter(
            "##splitter_ui_rgb",
            float(ui_panel_width),
            lambda dx: setattr(
                self, "_resize_ui_panel_width", float(self._resize_ui_panel_width) + dx
            ),
        )
        render_splitter(
            "##splitter_rgb_3d",
            float(ui_panel_width + rgb_panel_width),
            lambda dx: setattr(
                self, "_resize_rgb_panel_width", float(self._resize_rgb_panel_width) + dx
            ),
        )
        if prev_drag_active and not self._resize_drag_active:
            self.ui_panel_width = float(self._resize_ui_panel_width)
            self.rgb_panel_width = float(self._resize_rgb_panel_width)
            self.viz_panel_width = float(self._resize_viz_panel_width)
            self._clamp_panel_widths()
        elif not self._resize_drag_active:
            self.ui_panel_width = float(self._resize_ui_panel_width)
            self.rgb_panel_width = float(self._resize_rgb_panel_width)
            self.viz_panel_width = float(self._resize_viz_panel_width)
            self._clamp_panel_widths()
        self._ui_interaction_active = bool(ui_control_active or self._imgui_ui_busy())

    def on_key_event(self, key, action, modifiers):
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key in (keys.Q, keys.ESCAPE):
            if self._recording:
                self._stop_recording()
            self.wnd.close()


class LiveFoundationStereoViewer(OrbitViewer):
    title = "Live FoundationStereo"
    window_size = (3300, 2100)

    fs_state: LiveFsState = None
    fs_ckpt: str = ""
    fs_impl: str = "foundation"
    fs_hw: int = 256
    fs_valid_iters: int = 16
    fs_point_stride: int = 2
    fs_max_depth: float = 5.0
    consistency: bool = False
    consistency_threshold: float = 1.0

    ui_panel_width = 360

    def init_scene(self) -> None:
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)

        self.point_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_color;
                uniform mat4 mvp;
                uniform float point_size;
                out vec3 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = point_size;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                uniform float alpha;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, alpha);
                }
            """,
        )
        self.line_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_color;
                uniform mat4 mvp;
                out vec3 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                uniform float alpha;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, alpha);
                }
            """,
        )
        self.point_vbo = None
        self.point_vao = None
        self.point_count = 0
        self.trail_vbo = None
        self.trail_vao = None
        self.trail_count = 0
        self.frustum_vbo = None
        self.frustum_vao = None
        self.frustum_count = 0
        self.point_size = 2.0
        self.point_alpha = 1.0
        self.line_width = 2.0
        self.frustum_scale = 0.12
        self.use_depth_colormap = False
        self.show_trajectory = True
        self.show_frustum = True
        self.vio_world_is_y_up = False
        self.follow_mode = False
        self.follow_back = 3.0
        self.follow_up = 3.0
        self.follow_lookahead = 0.20
        self.follow_height_bias = 0.0
        self.follow_smoothing = 0.25
        self._last_pair_ts = -1
        self._processed = 0
        self._fps = 0.0
        self._infer_ms = 0.0
        self._min_depth = float("nan")
        self._max_depth = float("nan")
        self._mean_depth = float("nan")
        self._median_depth = float("nan")
        self._pair_delta_ms = 0.0
        self._t0 = time.time()
        self._target_inited = False
        self._pose_tail: list[tuple[int, np.ndarray]] = []
        self._last_T_world_rect: Optional[np.ndarray] = None
        self._last_linear = None

        self.camera_distance = 1.8
        self.camera_azimuth = -90.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0.0, 1.0, 0.0], dtype="f4")

        self._load_foundation_stereo()

    @staticmethod
    def _world_to_viewer(points: np.ndarray) -> np.ndarray:
        # Render directly in the gravity-aligned world frame so Z stays up.
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _zup_from_yup_matrix() -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _maybe_convert_vio_world(self, T_world_device: np.ndarray) -> np.ndarray:
        if not self.vio_world_is_y_up:
            return T_world_device
        return self._zup_from_yup_matrix() @ T_world_device

    @staticmethod
    def _depth_jet_colors(depths: np.ndarray, near: float = 0.1, far: float = 5.0):
        t = np.clip((depths.astype(np.float32) - near) / (far - near), 0.0, 1.0)
        u8 = (t * 255).astype(np.uint8).reshape(1, -1)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0].astype(np.float32) / 255.0
        return bgr[:, ::-1].astype("f4")

    def _maybe_print_fs_debug_stats(
        self,
        baseline: float,
        source_focal: float,
        rectified_focal: float,
        disparity: np.ndarray,
        z_valid: np.ndarray,
        pts_rect: np.ndarray,
    ) -> None:
        disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
        if disp_valid.size == 0 or z_valid.size == 0 or pts_rect.size == 0:
            return
        if self._processed != 1 and self._processed % 60 != 0:
            return

        disp_q = np.percentile(disp_valid, [10, 50, 90])
        depth_q = np.percentile(z_valid, [10, 50, 90])
        xyz_min = pts_rect.min(axis=0)
        xyz_max = pts_rect.max(axis=0)
        print(
            "==> fs_debug "
            f"baseline={baseline:.4f}m "
            f"f_src={source_focal:.2f}px "
            f"f_rect={rectified_focal:.2f}px "
            f"disp[p10,p50,p90]=[{disp_q[0]:.2f},{disp_q[1]:.2f},{disp_q[2]:.2f}]px "
            f"depth[p10,p50,p90]=[{depth_q[0]:.3f},{depth_q[1]:.3f},{depth_q[2]:.3f}]m "
            f"rect_xyz_min=[{xyz_min[0]:.3f},{xyz_min[1]:.3f},{xyz_min[2]:.3f}]m "
            f"rect_xyz_max=[{xyz_max[0]:.3f},{xyz_max[1]:.3f},{xyz_max[2]:.3f}]m",
            flush=True,
        )

    def _load_foundation_stereo(self) -> None:
        ensure_projectaria_fs_repo_on_path()
        self.fs_runtime = FoundationStereoRuntime(
            self.fs_ckpt,
            self.fs_valid_iters,
            fs_impl=self.fs_impl,
            consistency=self.consistency,
            consistency_threshold=self.consistency_threshold,
        )

    def _upload_line_data(self, attr: str, data: np.ndarray) -> None:
        vbo_name = f"{attr}_vbo"
        vao_name = f"{attr}_vao"
        count_name = f"{attr}_count"
        if data.size == 0:
            setattr(self, count_name, 0)
            return
        data = data.astype("f4")
        data_bytes = data.tobytes()
        vbo = getattr(self, vbo_name)
        vao = getattr(self, vao_name)
        if vbo is None or vbo.size < len(data_bytes):
            if vbo is not None:
                vbo.release()
            if vao is not None:
                vao.release()
            vbo = self.ctx.buffer(data_bytes)
            vao = self.ctx.vertex_array(
                self.line_prog,
                [(vbo, "3f 3f", "in_position", "in_color")],
            )
            setattr(self, vbo_name, vbo)
            setattr(self, vao_name, vao)
        else:
            vbo.orphan(vbo.size)
            vbo.write(data_bytes)
        setattr(self, count_name, len(data))

    def _update_pose_geometry(
        self,
        pair_ts: int,
        T_world_rect: np.ndarray,
        linear,
    ) -> None:
        cam_t_world = T_world_rect[:3, 3].reshape(1, 3).astype(np.float32)
        cam_t_view = self._world_to_viewer(cam_t_world)[0]
        self._pose_tail.append((pair_ts, cam_t_view))
        min_ts = pair_ts - 2_000_000_000
        self._pose_tail = [(ts, p) for ts, p in self._pose_tail if ts >= min_ts]

        if len(self._pose_tail) >= 2:
            trail_positions = [p for _, p in self._pose_tail]
            segments = []
            for p0, p1 in zip(trail_positions[:-1], trail_positions[1:]):
                color = np.array([0.1, 0.9, 1.0], dtype=np.float32)
                segments.append(np.concatenate([p0, color]))
                segments.append(np.concatenate([p1, color]))
            self._upload_line_data("trail", np.asarray(segments, dtype=np.float32))
        else:
            self.trail_count = 0

        w, h = linear.get_image_size()
        fx, fy, cx, cy = [float(v) for v in linear.get_projection_params()[:4]]
        z = float(self.frustum_scale)
        corners_cam = np.array(
            [
                [0.0, 0.0, 0.0],
                [(0.0 - cx) * z / fx, (0.0 - cy) * z / fy, z],
                [(w - cx) * z / fx, (0.0 - cy) * z / fy, z],
                [(w - cx) * z / fx, (h - cy) * z / fy, z],
                [(0.0 - cx) * z / fx, (h - cy) * z / fy, z],
            ],
            dtype=np.float32,
        )
        corners_world = (
            T_world_rect[:3, :3] @ corners_cam.T + T_world_rect[:3, 3:4]
        ).T
        corners = self._world_to_viewer(corners_world)
        color = np.array([1.0, 0.85, 0.1], dtype=np.float32)
        edge_indices = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        lines = []
        for i0, i1 in edge_indices:
            lines.append(np.concatenate([corners[i0], color]))
            lines.append(np.concatenate([corners[i1], color]))
        self._upload_line_data("frustum", np.asarray(lines, dtype=np.float32))

    def _apply_follow_view(self, T_world_rect: np.ndarray) -> None:
        origin_world = np.asarray(T_world_rect[:3, 3], dtype=np.float32)
        R_world_rect = np.asarray(T_world_rect[:3, :3], dtype=np.float32)
        forward_world = R_world_rect @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_world = R_world_rect @ np.array([0.0, -1.0, 0.0], dtype=np.float32)

        camera_world = (
            origin_world
            - forward_world * float(self.follow_back)
            + up_world * float(self.follow_up)
        )
        target_world = origin_world + forward_world * float(self.follow_lookahead)
        target_view = self._world_to_viewer(target_world.reshape(1, 3))[0]
        camera_view = self._world_to_viewer(camera_world.reshape(1, 3))[0]
        delta = camera_view - target_view
        dist = float(np.linalg.norm(delta))
        if dist < 1e-5:
            return

        azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
        elevation = float(np.degrees(np.arcsin(np.clip(delta[2] / dist, -1.0, 1.0))))

        blend = float(np.clip(self.follow_smoothing, 0.0, 1.0))
        if blend <= 0.0:
            self.camera_target = target_view.astype("f4")
            self.camera_distance = dist
            self.camera_azimuth = azimuth
            self.camera_elevation = elevation
            return

        self.camera_target = (
            (1.0 - blend) * self.camera_target + blend * target_view.astype("f4")
        ).astype("f4")
        self.camera_distance = (1.0 - blend) * float(self.camera_distance) + blend * dist
        self.camera_azimuth = (1.0 - blend) * float(self.camera_azimuth) + blend * azimuth
        self.camera_elevation = (
            (1.0 - blend) * float(self.camera_elevation) + blend * elevation
        )

    def get_camera_matrices(self):
        from utils.viewer_3d import _look_at, _perspective_projection

        w, h = self.wnd.size
        aspect_ratio = max(1, w - self.ui_panel_width) / max(1, h)
        projection = _perspective_projection(45.0, aspect_ratio, 0.02, 50.0)
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        camera_pos = self.camera_target + np.array(
            [
                self.camera_distance
                * np.cos(elevation_rad)
                * np.cos(azimuth_rad),
                self.camera_distance
                * np.cos(elevation_rad)
                * np.sin(azimuth_rad),
                self.camera_distance * np.sin(elevation_rad),
            ],
            dtype="f4",
        )
        view = _look_at(tuple(camera_pos), tuple(self.camera_target), (0.0, 0.0, 1.0))
        mvp = np.eye(4, dtype="f4") @ view @ projection
        return projection, view, mvp

    def _make_linear_calib(self, source_calib):
        from projectaria_tools.core import calibration
        from projectaria_tools.core.sophus import SE3

        params = source_calib.get_projection_params()
        src_w, src_h = source_calib.get_image_size()
        scale = min(self.fs_hw / float(src_w), self.fs_hw / float(src_h))
        focal = float(params[0]) * scale * 1.25
        linear_params = np.array(
            [focal, focal, self.fs_hw / 2.0, self.fs_hw / 2.0]
        )
        return calibration.CameraCalibration(
            source_calib.get_label() + f"-linear-{self.fs_hw}",
            calibration.CameraModelType.LINEAR,
            linear_params,
            SE3(),
            self.fs_hw,
            self.fs_hw,
            None,
            source_calib.get_max_solid_angle(),
            source_calib.get_serial_number(),
        )

    def _infer_disparity(self, left_rect, right_rect):
        return self.fs_runtime.infer(left_rect, right_rect)

    def _maybe_update_points(self) -> None:
        ensure_projectaria_fs_repo_on_path()
        from projectaria_tools.core.image import InterpolationMethod
        from stereo_utils import (
            create_scanline_rectified_cameras,
            disparity_to_depth,
            rectify_stereo_pair,
        )

        (
            left_frame,
            right_frame,
            left_calib,
            right_calib,
            T_world_device,
        ) = self.fs_state.snapshot()
        if (
            left_frame is None
            or right_frame is None
            or left_calib is None
            or right_calib is None
            or T_world_device is None
        ):
            return

        left_img, left_ts = left_frame
        right_img, right_ts = right_frame
        pair_ts = max(left_ts, right_ts)
        if pair_ts == self._last_pair_ts:
            return
        self._pair_delta_ms = abs(left_ts - right_ts) / 1e6
        if self._pair_delta_ms > 2.0:
            return

        T_world_device = self._maybe_convert_vio_world(T_world_device)

        T_left_device = left_calib.get_transform_device_camera().inverse()
        T_right_device = right_calib.get_transform_device_camera().inverse()
        T_left_right = T_left_device @ T_right_device.inverse()
        T_device_left = np.asarray(T_left_device.inverse().to_matrix(), dtype=np.float32)
        R_left_rect, R_right_rect = create_scanline_rectified_cameras(
            T_left_device, T_right_device
        )
        linear = self._make_linear_calib(left_calib)
        left_rect, right_rect = rectify_stereo_pair(
            left_img,
            right_img,
            left_calib,
            right_calib,
            linear,
            linear,
            R_left_rect,
            R_right_rect,
            interpolation=InterpolationMethod.BILINEAR,
        )

        t_infer = time.time()
        disparity = self._infer_disparity(left_rect, right_rect)
        self._infer_ms = (time.time() - t_infer) * 1000.0

        baseline = float(np.linalg.norm(T_left_right.translation()))
        source_focal = float(left_calib.get_projection_params()[0])
        focal = float(linear.get_projection_params()[0])
        depth = disparity_to_depth(disparity, baseline, focal)
        h, w = depth.shape
        stride = max(1, int(self.fs_point_stride))
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[0:h:stride, 0:w:stride]
        valid = np.isfinite(z) & (z > 0.03) & (z < float(self.fs_max_depth))
        if not np.any(valid):
            return

        fx, fy, cx, cy = [float(v) for v in linear.get_projection_params()[:4]]
        x_cam = (xs[valid].astype(np.float32) - cx) * z[valid] / fx
        y_cam = (ys[valid].astype(np.float32) - cy) * z[valid] / fy
        z_cam = z[valid].astype(np.float32)

        pts_rect = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        self._maybe_print_fs_debug_stats(
            baseline=baseline,
            source_focal=source_focal,
            rectified_focal=focal,
            disparity=disparity,
            z_valid=z[valid].astype(np.float32),
            pts_rect=pts_rect,
        )
        R_left_rect_mat = np.asarray(R_left_rect.to_matrix(), dtype=np.float32)
        T_world_rect = T_world_device @ T_device_left
        T_world_rect[:3, :3] = T_world_rect[:3, :3] @ R_left_rect_mat
        pts_world = (T_world_rect[:3, :3] @ pts_rect.T + T_world_rect[:3, 3:4]).T
        pts = self._world_to_viewer(pts_world)
        intens = left_rect[0:h:stride, 0:w:stride][valid].astype(np.float32) / 255.0
        if self.use_depth_colormap:
            colors = self._depth_jet_colors(z[valid], near=0.1, far=5.0)
        else:
            colors = np.stack([intens, intens, intens], axis=1).astype("f4")
        data = np.concatenate([pts, colors], axis=1).astype("f4")
        data_bytes = data.tobytes()

        if self.point_vbo is None or self.point_vbo.size < len(data_bytes):
            if self.point_vbo is not None:
                self.point_vbo.release()
            if self.point_vao is not None:
                self.point_vao.release()
            self.point_vbo = self.ctx.buffer(data_bytes)
            self.point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self.point_vbo, "3f 3f", "in_position", "in_color")],
            )
        else:
            self.point_vbo.orphan(self.point_vbo.size)
            self.point_vbo.write(data_bytes)

        self.point_count = len(pts)
        self._last_pair_ts = pair_ts
        self._last_T_world_rect = T_world_rect
        self._last_linear = linear
        self._processed += 1
        self._fps = self._processed / max(time.time() - self._t0, 1e-6)
        z_valid = z[valid].astype(np.float32)
        self._min_depth = float(np.nanmin(z_valid))
        self._max_depth = float(np.nanmax(z_valid))
        self._mean_depth = float(np.nanmean(z_valid))
        self._median_depth = float(np.nanmedian(z_valid))
        if not self._target_inited:
            self.camera_target = np.nanmedian(pts, axis=0).astype("f4")
            self._target_inited = True
        self._update_pose_geometry(pair_ts, T_world_rect, linear)
        print(
            f"==> fs_view[{self._processed}] points={self.point_count} "
            f"infer={self._infer_ms:.1f}ms median_depth={self._median_depth:.3f}m",
            flush=True,
        )

    def render_3d(self, time_val: float, frame_time: float) -> None:
        self._maybe_update_points()
        if self.follow_mode and self._last_T_world_rect is not None:
            self._apply_follow_view(self._last_T_world_rect)
        _, _, mvp = self.get_camera_matrices()
        mvp_bytes = mvp.astype("f4").tobytes()
        if self.point_vao is not None and self.point_count > 0:
            self.point_prog["mvp"].write(mvp_bytes)
            self.point_prog["point_size"].write(
                np.array(self.point_size, dtype="f4").tobytes()
            )
            self.point_prog["alpha"].write(
                np.array(self.point_alpha, dtype="f4").tobytes()
            )
            self.point_vao.render(mode=self.ctx.POINTS, vertices=self.point_count)
        if self.show_trajectory and self.trail_vao is not None and self.trail_count > 0:
            self.ctx.line_width = float(self.line_width)
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.trail_vao.render(mode=self.ctx.LINES, vertices=self.trail_count)
        if self.show_frustum and self.frustum_vao is not None and self.frustum_count > 0:
            self.ctx.line_width = float(self.line_width)
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.frustum_vao.render(mode=self.ctx.LINES, vertices=self.frustum_count)

    def render_ui(self) -> None:
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(self.ui_panel_width, self.wnd.size[1], imgui.ALWAYS)
        imgui.begin("FoundationStereo", None, imgui.WINDOW_NO_RESIZE)
        imgui.text(f"Frames: {self._processed}")
        imgui.text(f"Points: {self.point_count}")
        imgui.text(f"Infer: {self._infer_ms:.1f} ms")
        imgui.text(f"Avg FPS: {self._fps:.2f}")
        imgui.text(f"Pair delta: {self._pair_delta_ms:.3f} ms")
        imgui.text(f"Min depth: {self._min_depth:.3f} m")
        imgui.text(f"Mean depth: {self._mean_depth:.3f} m")
        imgui.text(f"Median depth: {self._median_depth:.3f} m")
        imgui.text(f"Max depth: {self._max_depth:.3f} m")
        _, self.vio_world_is_y_up = imgui.checkbox(
            "VIO world is Y-up", self.vio_world_is_y_up
        )
        _, self.use_depth_colormap = imgui.checkbox(
            "Jet depth colors", self.use_depth_colormap
        )
        _, self.show_frustum = imgui.checkbox("Show frustum", self.show_frustum)
        follow_clicked = imgui.button(
            "Follow view" if not self.follow_mode else "Free orbit"
        )
        if follow_clicked:
            self.follow_mode = not self.follow_mode
            if self.follow_mode and self._last_T_world_rect is not None:
                self._apply_follow_view(self._last_T_world_rect)
        _, self.follow_back = imgui.slider_float(
            "Follow back", self.follow_back, 0.05, 10.0
        )
        _, self.follow_up = imgui.slider_float("Follow up", self.follow_up, 0.0, 10.0)
        _, self.follow_lookahead = imgui.slider_float(
            "Look ahead", self.follow_lookahead, 0.0, 3.0
        )
        _, self.follow_smoothing = imgui.slider_float(
            "Follow smoothing", self.follow_smoothing, 0.0, 1.0
        )
        imgui.text("Gravity convention: world +Z is up")
        _, self.point_size = imgui.slider_float("Point size", self.point_size, 1.0, 8.0)
        _, self.point_alpha = imgui.slider_float(
            "Point alpha", self.point_alpha, 0.05, 1.0
        )
        _, self.line_width = imgui.slider_float("Line width", self.line_width, 1.0, 8.0)
        _, self.frustum_scale = imgui.slider_float(
            "Frustum scale", self.frustum_scale, 0.03, 0.5
        )
        imgui.text("Jet: blue=0.1m, red=5.0m")
        imgui.text("q/Esc quits")
        imgui.end()

    def on_key_event(self, key, action, modifiers):
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key in (keys.Q, keys.ESCAPE):
            self.wnd.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile_name", type=str, default="profile10")
    p.add_argument("--wifi", action="store_true")
    p.add_argument("--ip", type=str, default=None)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--labels", type=str, default="cvpr_demo")
    p.add_argument("--thresh2d", type=float, default=0.25)
    p.add_argument("--thresh3d", type=float, default=0.5)
    p.add_argument("--detector_hw", type=int, default=960)
    p.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Close the live combined viewer after N render steps (0 = no limit).",
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="Print detailed live performance timings for RGB, FS, render, and loop stages.",
    )
    p.add_argument(
        "--bench_every",
        type=int,
        default=30,
        help="Print benchmark timings every N frames, plus first frames and slow frames.",
    )
    p.add_argument(
        "--no_fs_async",
        action="store_true",
        help="Run FoundationStereo synchronously on the render thread.",
    )
    p.add_argument(
        "--cpu_rgb_overlays",
        action="store_true",
        help="Draw RGB 2D/3D overlays into image arrays with OpenCV instead of ImGui draw lists.",
    )
    p.add_argument(
        "--record_fps",
        type=float,
        default=5.0,
        help="FPS used when encoding UI video recordings.",
    )
    p.add_argument(
        "--ui_capture",
        type=str,
        default="",
        help="Write a one-shot full-window UI screenshot PNG to this path.",
    )
    p.add_argument(
        "--ui_capture_frame",
        type=int,
        default=3,
        help="Render frame index for --ui_capture after the viewer opens.",
    )
    p.add_argument(
        "--ui_capture_mock",
        action="store_true",
        help="Open a UI-only mock viewer for --ui_capture without connecting to Aria.",
    )
    p.add_argument(
        "--rectify",
        action="store_true",
        help="Rectify RGB fisheye to pinhole before OWL, then map 2D boxes back to fisheye before Boxer.",
    )
    p.add_argument("--image_hw", type=int, default=None)
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(CKPT_PATH, DEFAULT_BOXERNET_CKPT),
    )
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument(
        "--force_precision", type=str, default=None, choices=["float32", "bfloat16"]
    )
    p.add_argument(
        "--debug_geometry",
        action="store_true",
        help="Print live frame size, camera intrinsics, resize scales, and poses.",
    )
    p.add_argument(
        "--geometry_probe",
        action="store_true",
        help="Print live geometry debug info and exit before loading models.",
    )
    p.add_argument(
        "--live_rotation",
        type=str,
        default="none",
        choices=["none", "cw", "ccw", "cam_cw", "cam_ccw"],
        help=(
            "Live RGB rotation correction before Boxer. Default 'none' matches "
            "AriaLoader(unrotate=True) for Gen2 metadata; 'cam_cw' keeps the live "
            "image upright but rotates only the camera model; 'cw' rotates both "
            "image and camera like the Gen1 unrotate path."
        ),
    )
    p.add_argument(
        "--slam_probe",
        action="store_true",
        help="Receive SLAM image callbacks briefly, print camera ids/shapes, and exit.",
    )
    p.add_argument(
        "--slam_probe_seconds",
        type=float,
        default=6.0,
        help="Duration for --slam_probe.",
    )
    p.add_argument(
        "--fs_point_stride", type=int, default=2
    )
    p.add_argument(
        "--fsp_every",
        type=int,
        default=1,
        help="Run FoundationStereo/FSP on every Nth valid stereo pair (default 1).",
    )
    p.add_argument(
        "--fs_disparity_median",
        type=int,
        default=0,
        help="Apply an odd-sized median filter to FS disparity before depth conversion (0 disables).",
    )
    p.add_argument("--fs_max_depth", type=float, default=5.0)
    p.add_argument("--fs_valid_iters", type=int, default=16)
    p.add_argument(
        "--fs_impl",
        type=str,
        default="auto",
        choices=["auto", "foundation", "fast"],
        help="Foundation stereo backend implementation to use. Default auto-detects from the checkpoint/engine name.",
    )
    p.add_argument(
        "--consistency",
        action="store_true",
        help="Enable FoundationStereo left-right disparity consistency filtering.",
    )
    p.add_argument(
        "--consistency_threshold",
        type=float,
        default=1.0,
        help="LR consistency threshold in pixels (default 1.0).",
    )
    fs_preset_help = "FS model preset shorthand. " + "; ".join(
        f"{name}={FS_MODEL_PRESET_HELP[name]}" for name in sorted(FS_MODEL_PRESET_HELP)
    )
    p.add_argument(
        "--fsm",
        "--fs_model",
        type=str,
        default="f256",
        choices=sorted(FS_MODEL_PRESETS),
        help=fs_preset_help,
    )
    p.add_argument(
        "--fs_ckpt",
        type=str,
        default=None,
        help="Explicit FS .engine/.plan/.onnx/.pth path. Overrides --fsm.",
    )
    p.add_argument(
        "--fs",
        action="store_true",
        help="Enable FoundationStereo in the combined viewer. If any of --fs/--owl/--boxer are passed, only the selected components are enabled.",
    )
    p.add_argument(
        "--owl",
        action="store_true",
        help="Enable OWL in the combined viewer. If any of --fs/--owl/--boxer are passed, only the selected components are enabled.",
    )
    p.add_argument(
        "--boxer",
        action="store_true",
        help="Enable Boxer in the combined viewer. Implies OWL proposals are still used internally.",
    )
    p.add_argument(
        "--tracker",
        action="store_true",
        help="Enable the 3D OBB tracker in the combined viewer. Implies Boxer and OWL are still used internally.",
    )
    return p.parse_args()


def pick_device(force_cpu: bool) -> str:
    if torch.backends.mps.is_available() and not force_cpu:
        return "mps"
    if torch.cuda.is_available() and not force_cpu:
        return "cuda"
    return "cpu"


def start_streaming_with_session_recovery(device_client, device) -> None:
    try:
        device.start_streaming()
        return
    except RuntimeError as err:
        message = str(err)
        if "User session already started" not in message:
            raise

        print(
            "==> Existing Aria streaming session detected; sending stop_streaming and retrying once...",
            flush=True,
        )
        try:
            device.stop_streaming()
            time.sleep(1.0)
        except Exception as stop_err:
            print(f"==> stop_streaming during recovery failed: {stop_err}", flush=True)

        try:
            device.start_streaming()
            print("==> Recovered existing Aria streaming session.", flush=True)
            return
        except RuntimeError:
            try:
                device_client.disconnect(device)
            except Exception as disconnect_err:
                print(
                    f"==> disconnect during recovery failed: {disconnect_err}",
                    flush=True,
                )
            raise


def main():
    args = parse_args()
    if bool(args.ui_capture_mock):
        labels_list = [s for s in args.labels.split(",") if s]
        text_labels = load_text_labels(labels_list)

        class _NoopLock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _MockState:
            lock = _NoopLock()
            frame = None

            @staticmethod
            def stream_hz() -> float:
                return 0.0

        class _MockOwl:
            min_confidence = float(args.thresh2d)
            nms_iou_threshold = 0.5

            def set_text_prompts(self, prompts):
                self.text_prompts = list(prompts)

        sem_name_to_id = {label: i for i, label in enumerate(text_labels)}
        LiveBoxerViewer.state = _MockState()
        LiveBoxerViewer.owl = _MockOwl()
        LiveBoxerViewer.boxernet = None
        LiveBoxerViewer.text_labels = text_labels
        LiveBoxerViewer.sem_name_to_id = sem_name_to_id
        LiveBoxerViewer.sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}
        LiveBoxerViewer.boxernet_ckpt = str(args.ckpt)
        LiveBoxerViewer.initial_prompts_csv = ",".join(text_labels)
        LiveBoxerViewer.HW = int(args.image_hw) if args.image_hw is not None else 960
        LiveBoxerViewer.detector_hw = int(args.detector_hw)
        LiveBoxerViewer.init_thresh3d = float(args.thresh3d)
        LiveBoxerViewer.dev = "cpu"
        LiveBoxerViewer.pdtype = torch.float32
        LiveBoxerViewer.max_steps = int(args.max_steps) if args.max_steps > 0 else 8
        LiveBoxerViewer.bench = bool(args.bench)
        LiveBoxerViewer.bench_every = int(args.bench_every)
        LiveBoxerViewer.fs_async = False
        LiveBoxerViewer.rgb_gpu_overlays = not bool(args.cpu_rgb_overlays)
        LiveBoxerViewer.record_fps = float(args.record_fps)
        LiveBoxerViewer.ui_capture_path = str(args.ui_capture or "")
        LiveBoxerViewer.ui_capture_frame = int(args.ui_capture_frame)
        LiveBoxerViewer.fs_state = None
        LiveBoxerViewer.enable_foundation_stereo = False
        LiveBoxerViewer.enable_owl = False
        LiveBoxerViewer.enable_boxer = False
        LiveBoxerViewer.enable_tracker = False
        print("==> Launching UI capture mock viewer.", flush=True)
        launch_viewer(LiveBoxerViewer)
        return

    ensure_aria_tools_on_path()

    selected_components = bool(args.fs or args.owl or args.boxer or args.tracker)
    enable_fs_requested = bool(args.fs) if selected_components else True
    if enable_fs_requested:
        if args.fs_ckpt is not None and args.fsm is not None:
            print("==> --fs_ckpt overrides --fsm", flush=True)
        if args.fs_ckpt is None and args.fsm is not None:
            args.fs_ckpt = resolve_fs_model_preset(args.fsm)
        if args.fs_ckpt is None:
            args.fs_ckpt = resolve_default_foundation_stereo_model()
        if args.fs_impl == "auto":
            args.fs_impl = infer_fs_impl_from_model_path(args.fs_ckpt)
        args.fs_hw = resolve_fs_hw(args.fs_ckpt, args.fs_impl)
        print(
            f"==> FS model: {args.fs_ckpt} (preset={args.fsm or 'default'}, "
            f"impl={args.fs_impl}, hw={args.fs_hw})",
            flush=True,
        )
    else:
        args.fs_ckpt = args.fs_ckpt or ""
        args.fs_hw = 256
        print("==> FS disabled; skipping FoundationStereo runtime setup", flush=True)

    device_client = sdk_gen2.DeviceClient()
    device_client.set_client_config(sdk_gen2.DeviceClientConfig())

    if args.wifi and not (args.ip or args.serial):
        raise SystemExit(
            "--wifi requires --ip (preferred) or --serial; auto-discovery over WiFi is "
            "not available in this SDK build. Find the device IP in the Mobile "
            "Companion App."
        )

    device, target, target_desc = connect_with_ip_fallback(
        device_client, args.ip, args.serial
    )
    if args.ip:
        save_cached_aria_ip(args.ip)
    print(f"==> Connecting to device ({target_desc})", flush=True)

    sc = sdk_gen2.HttpStreamingConfig()
    sc.profile_name = args.profile_name
    sc.streaming_interface = (
        sdk_gen2.StreamingInterface.WIFI_STA
        if args.wifi
        else sdk_gen2.StreamingInterface.USB_NCM
    )
    print(f"==> Streaming interface: {sc.streaming_interface.name}", flush=True)
    device.set_streaming_config(sc)
    start_streaming_with_session_recovery(device_client, device)

    state = StreamState()
    fs_state = LiveFsState() if enable_fs_requested else None
    device_calib_cb, vio_cb, rgb_cb = make_callbacks(state)
    fs_device_calib_cb = None
    fs_slam_cb = None
    fs_vio_cb = None
    if fs_state is not None:
        fs_device_calib_cb, fs_slam_cb, fs_vio_cb = make_live_fs_callbacks(
            fs_state
        )
    slam_probe_cb = None
    slam_probe_counts = None
    if args.slam_probe:
        slam_probe_cb, slam_probe_counts = make_slam_probe_callback()

    srv = sdk_gen2.HttpServerConfig()
    srv.address = "0.0.0.0"
    srv.port = 6768

    rx = receiver.StreamReceiver(enable_image_decoding=True, enable_raw_stream=False)
    rx.set_rgb_queue_size(1)
    rx.set_vio_queue_size(1)
    if enable_fs_requested or args.slam_probe:
        rx.set_slam_queue_size(2 if enable_fs_requested else 1)
    rx.set_server_config(srv)

    def combined_device_calib_cb(device_calib):
        device_calib_cb(device_calib)
        if fs_device_calib_cb is not None:
            fs_device_calib_cb(device_calib)

    def combined_vio_cb(vio):
        vio_cb(vio)
        if fs_vio_cb is not None:
            fs_vio_cb(vio)

    def combined_slam_cb(image_data, image_record):
        if fs_slam_cb is not None:
            fs_slam_cb(image_data, image_record)
        if slam_probe_cb is not None:
            slam_probe_cb(image_data, image_record)

    if fs_state is not None:
        rx.register_device_calib_callback(combined_device_calib_cb)
        rx.register_slam_callback(combined_slam_cb)
        rx.register_vio_callback(combined_vio_cb)
        rx.register_rgb_callback(rgb_cb)
    else:
        rx.register_device_calib_callback(device_calib_cb)
        if args.slam_probe:
            rx.register_slam_callback(slam_probe_cb)
        else:
            rx.register_vio_callback(vio_cb)
        rx.register_rgb_callback(rgb_cb)
    rx.start_server()

    try:
        if args.slam_probe:
            print(
                f"==> Waiting {args.slam_probe_seconds:.1f}s for SLAM frames ...",
                flush=True,
            )
            time.sleep(args.slam_probe_seconds)
            print(f"==> SLAM counts: {dict(slam_probe_counts or {})}", flush=True)
            return

        print("==> Waiting for device calib + first VIO + first RGB frame ...")
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            frame, T_wr, intr, T_cr, csize = state.snapshot()
            if (
                frame is not None
                and T_wr is not None
                and intr is not None
                and T_cr is not None
                and csize is not None
            ):
                ready = True
                break
            time.sleep(0.05)
        if not ready:
            raise RuntimeError(
                "Timed out waiting for streaming data (calib + VIO + RGB)."
            )
        if args.debug_geometry or args.geometry_probe:
            frame, T_wr, intr, T_cr, csize = state.snapshot()
            arr_rgb, ts_ns = frame
            image_h, image_w = arr_rgb.shape[:2]
            calib_w, calib_h = csize
            print("==> Streaming geometry preflight", flush=True)
            print(
                f"    first RGB ts={ts_ns / 1e9:.6f}s, frame HxW={image_h}x{image_w}, "
                f"calib WxH={calib_w}x{calib_h}",
                flush=True,
            )
            print(
                f"    raw f={intr[0]:.4f}, cx={intr[1]:.4f}, cy={intr[2]:.4f}; "
                f"T_world_rig.t={_fmt_tensor(T_wr.t)}, T_camera_rig.t={_fmt_tensor(T_cr.t)}",
                flush=True,
            )
            if args.geometry_probe:
                probe_hw = int(args.image_hw) if args.image_hw is not None else 960
                cam = build_cam(intr, T_cr, csize, (image_w, image_h), probe_hw)
                probe_img = cv2.resize(
                    arr_rgb, (probe_hw, probe_hw), interpolation=cv2.INTER_LINEAR
                )
                probe_img_torch = (
                    torch.from_numpy(probe_img).permute(2, 0, 1)[None].float()
                    / 255.0
                )
                _, cam, rotated0, rotation_policy = apply_live_rotation(
                    probe_img_torch, cam, args.live_rotation
                )
                log_geometry_debug(
                    arr_rgb,
                    intr,
                    T_wr,
                    T_cr,
                    csize,
                    cam,
                    probe_hw,
                    rotated0,
                    rotation_policy,
                )
                return

        dev = pick_device(args.force_cpu)
        print(f"==> Using device {dev}")

        labels_list = [s for s in args.labels.split(",") if s]
        text_labels = load_text_labels(labels_list)
        taxonomy_name = labels_list[0] if labels_list else "custom"
        print(f"==> {len(text_labels)} text prompts ({taxonomy_name})")

        owl = OwlWrapper(
            dev,
            text_prompts=text_labels,
            min_confidence=args.thresh2d,
            precision=args.force_precision,
        )
        boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=dev)
        HW = int(args.image_hw) if args.image_hw is not None else int(boxernet.hw)
        print(
            f"==> BoxerNet loaded, ckpt hw={int(boxernet.hw)}, using inference hw={HW}"
        )

        if args.force_precision is not None:
            pdtype = (
                torch.bfloat16
                if args.force_precision == "bfloat16"
                else torch.float32
            )
        elif dev == "cuda" and torch.cuda.is_bf16_supported():
            pdtype = torch.bfloat16
        else:
            pdtype = torch.float32

        sem_name_to_id = {label: i for i, label in enumerate(text_labels)}
        sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}

        LiveBoxerViewer.state = state
        LiveBoxerViewer.owl = owl
        LiveBoxerViewer.boxernet = boxernet
        LiveBoxerViewer.text_labels = text_labels
        LiveBoxerViewer.sem_name_to_id = sem_name_to_id
        LiveBoxerViewer.sem_id_to_name = sem_id_to_name
        LiveBoxerViewer.boxernet_ckpt = str(args.ckpt)
        LiveBoxerViewer.initial_prompts_csv = ",".join(text_labels)
        LiveBoxerViewer.HW = HW
        LiveBoxerViewer.detector_hw = args.detector_hw
        LiveBoxerViewer.rectify_rgb_for_owl_boxes = bool(args.rectify)
        LiveBoxerViewer.init_thresh3d = args.thresh3d
        LiveBoxerViewer.dev = dev
        LiveBoxerViewer.pdtype = pdtype
        LiveBoxerViewer.debug_geometry = args.debug_geometry
        LiveBoxerViewer.live_rotation = args.live_rotation
        LiveBoxerViewer.max_steps = int(args.max_steps)
        LiveBoxerViewer.bench = bool(args.bench)
        LiveBoxerViewer.bench_every = int(args.bench_every)
        LiveBoxerViewer.fs_async = not bool(args.no_fs_async)
        LiveBoxerViewer.rgb_gpu_overlays = not bool(args.cpu_rgb_overlays)
        LiveBoxerViewer.record_fps = float(args.record_fps)
        LiveBoxerViewer.ui_capture_path = str(args.ui_capture or "")
        LiveBoxerViewer.ui_capture_frame = int(args.ui_capture_frame)
        LiveBoxerViewer.fs_state = fs_state
        LiveBoxerViewer.fs_ckpt = args.fs_ckpt
        LiveBoxerViewer.fs_impl = args.fs_impl
        LiveBoxerViewer.fs_hw = int(args.fs_hw)
        LiveBoxerViewer.fs_valid_iters = int(args.fs_valid_iters)
        LiveBoxerViewer.consistency = bool(args.consistency)
        LiveBoxerViewer.consistency_threshold = float(args.consistency_threshold)
        LiveBoxerViewer.fsp_every = max(1, int(args.fsp_every))
        LiveBoxerViewer.fs_disparity_median = max(0, int(args.fs_disparity_median))
        LiveBoxerViewer.fs_point_stride = int(args.fs_point_stride)
        LiveBoxerViewer.fs_max_depth = float(args.fs_max_depth)
        if args.fs or args.owl or args.boxer or args.tracker:
            LiveBoxerViewer.enable_foundation_stereo = bool(args.fs)
            LiveBoxerViewer.enable_tracker = bool(args.tracker)
            LiveBoxerViewer.enable_boxer = bool(args.boxer or args.tracker)
            LiveBoxerViewer.enable_owl = bool(args.owl or args.boxer or args.tracker)

        print("==> Launching viewer.")
        launch_viewer(LiveBoxerViewer)
    finally:
        device.stop_streaming()
        rx.stop_server()


if __name__ == "__main__":
    main()
