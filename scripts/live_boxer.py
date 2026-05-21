"""Live Aria Gen2 streaming + BoxerNet demo with interactive 3D viewer.

moderngl-window viewer with three regions:
  * Left:   ImGui control panel (sliders, toggles).
  * Center: Live RGB frame + OWLv2 2D bounding-box overlays.
  * Right:  Interactive 3D scene (orbit camera) with BoxerNet 3D OBBs and
            a camera frustum marker for the current device pose.

Press 'q' or Esc to quit. Right-drag to orbit, left-drag to pan, scroll to zoom.
"""

import argparse
import colorsys
import hashlib
import ipaddress
import os
import platform
import re
import subprocess
import sys
import threading
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

import aria
import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import (
    FrontendOutput,
    ImageData,
    ImageDataRecord,
)

import utils.imgui_compat as imgui
from boxernet.boxernet import BoxerNet
from utils.gravity import gravity_align_T_world_cam
from owl.owl_wrapper import OwlWrapper
from utils.demo_utils import CKPT_PATH, DEFAULT_BOXERNET_CKPT
from utils.image import draw_bb3s, put_text, render_bb2, render_depth_patches, torch2cv2
from utils.taxonomy import BOXY_SEM2NAME, SSI_COLORS_ALT, TEXT2COLORS, load_text_labels
from utils.tw.camera import CameraTW
from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.tw.pose import PoseTW
from utils.viewer_3d import OrbitViewer, launch_viewer


GEN2_CAMERA_ID_TO_LABEL = {
    1: "slam-front-left",
    2: "slam-front-right",
    4: "slam-side-left",
    8: "slam-side-right",
    16: "camera-et-left",
    32: "camera-et-right",
    64: "camera-rgb",
}

ARIA_LAST_IP_PATH = os.path.join(REPO_ROOT, ".aria_last_ip.txt")


def ensure_aria_tools_on_path() -> None:
    aria_dir = os.path.dirname(os.path.abspath(aria.__file__))
    tools_dir = os.path.join(aria_dir, "tools")
    if not os.path.exists(os.path.join(tools_dir, "adb")):
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if tools_dir not in path_parts:
        os.environ["PATH"] = tools_dir + os.pathsep + os.environ.get("PATH", "")


def load_cached_aria_ip() -> Optional[str]:
    try:
        with open(ARIA_LAST_IP_PATH, "r", encoding="ascii") as f:
            ip = f.read().strip()
    except OSError:
        return None
    if not ip:
        return None
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return None
    return ip


def save_cached_aria_ip(ip: str) -> None:
    try:
        with open(ARIA_LAST_IP_PATH, "w", encoding="ascii") as f:
            f.write(ip + "\n")
    except OSError:
        pass


def _iface_priority_for_aria_peer(iface: str) -> int:
    if iface.startswith("enx") or iface.startswith("usb"):
        return 0
    if iface.startswith("eth"):
        return 1
    if iface.startswith("wl"):
        return 10
    return 5


def find_usb_ncm_device_ip() -> Optional[str]:
    """Best-effort discovery of the Aria peer on the USB-NCM link.

    The Gen2 SDK's auto-USB connect path can block in native code on Linux when
    udev/NetworkManager is not configured. The USB-NCM interface still gets a
    normal IPv4 link, so prefer a discovered peer IP when available.
    """
    try:
        addr = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if addr.returncode != 0:
        return None

    candidate_peers: list[tuple[int, str]] = []
    for line in addr.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1]
        cidr = parts[3]
        try:
            network = ipaddress.ip_interface(cidr).network
        except ValueError:
            continue
        if not network.is_private:
            continue

        try:
            neigh = subprocess.run(
                ["ip", "neigh", "show", "dev", iface],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if neigh.returncode != 0:
            continue

        for neigh_line in neigh.stdout.splitlines():
            match = re.match(r"^(\d+\.\d+\.\d+\.\d+)\s+.*\blladdr\b", neigh_line)
            if not match or "FAILED" in neigh_line:
                continue
            peer_ip = match.group(1)
            try:
                if ipaddress.ip_address(peer_ip) in network:
                    candidate_peers.append(
                        (_iface_priority_for_aria_peer(iface), peer_ip)
                    )
            except ValueError:
                continue

    if not candidate_peers:
        return None
    candidate_peers.sort(key=lambda item: item[0])
    return candidate_peers[0][1]


def connect_with_ip_fallback(
    device_client: sdk_gen2.DeviceClient,
    explicit_ip: Optional[str],
    explicit_serial: Optional[str],
) -> tuple[object, sdk_gen2.DeviceTarget, str]:
    if explicit_ip:
        target = sdk_gen2.DeviceTarget(ip=explicit_ip)
        return device_client.connect(target), target, f"ip={explicit_ip}"
    if explicit_serial:
        target = sdk_gen2.DeviceTarget(serial=explicit_serial)
        return device_client.connect(target), target, f"serial={explicit_serial}"

    candidates: list[str] = []
    discovered_ip = find_usb_ncm_device_ip()
    if discovered_ip:
        candidates.append(discovered_ip)
    cached_ip = load_cached_aria_ip()
    if cached_ip and cached_ip not in candidates:
        candidates.append(cached_ip)

    if not candidates:
        raise SystemExit(
            "Could not discover an Aria USB-NCM peer IP and no cached IP was found. "
            "Reconnect the glasses over USB or pass --ip directly."
        )

    last_error = None
    for ip in candidates:
        target = sdk_gen2.DeviceTarget(ip=ip)
        try:
            device = device_client.connect(target)
            save_cached_aria_ip(ip)
            desc = f"cached/discovered USB-NCM ip={ip}"
            return device, target, desc
        except RuntimeError as err:
            last_error = err
            continue

    raise RuntimeError(
        f"Failed to connect using cached/discovered IPs {candidates}: {last_error}"
    )


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


def _stable_label_color_rgb(label: str) -> tuple[float, float, float]:
    key = (label or "unknown").strip().lower()
    if key in TEXT2COLORS:
        color = TEXT2COLORS[key]
        return float(color[0]), float(color[1]), float(color[2])

    compact = key.replace(" ", "")
    for name, color in SSI_COLORS_ALT.items():
        lowered = name.strip().lower()
        if lowered == key or lowered.replace(" ", "") == compact:
            return float(color[0]), float(color[1]), float(color[2])

    digest = hashlib.md5(key.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    sat = 0.68 + 0.20 * (digest[1] / 255.0)
    val = 0.72 + 0.18 * (digest[2] / 255.0)
    r, g, b = colorsys.hsv_to_rgb(float(hue), float(sat), float(val))
    return float(r), float(g), float(b)


def obb_class_color_rgb(label: str, sem_id: int) -> tuple[float, float, float]:
    if sem_id in BOXY_SEM2NAME:
        sem_name = BOXY_SEM2NAME[sem_id]
        color = SSI_COLORS_ALT.get(sem_name)
        if color is not None:
            return float(color[0]), float(color[1]), float(color[2])
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


def get_autocast_dtype_for_cuda():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class StreamState:
    """Thread-safe slots populated by streaming callbacks, read on main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame: Optional[tuple[np.ndarray, int]] = None
        self.T_world_rig: Optional[PoseTW] = None
        self.T_camera_rig: Optional[PoseTW] = None
        self.rgb_image_size: Optional[tuple[int, int]] = None
        self.rgb_intrinsics: Optional[list[float]] = None
        self.debug_geometry_logged = False
        self.rectify_debug_logged = False

    def snapshot(self):
        with self.lock:
            return (
                self.frame,
                self.T_world_rig,
                self.rgb_intrinsics,
                self.T_camera_rig,
                self.rgb_image_size,
            )


class LiveFsState:
    def __init__(self):
        self.lock = threading.Lock()
        self.left_frame: Optional[tuple[np.ndarray, int]] = None
        self.right_frame: Optional[tuple[np.ndarray, int]] = None
        self.left_calib = None
        self.right_calib = None
        self.T_world_device: Optional[np.ndarray] = None

    def snapshot(self):
        with self.lock:
            return (
                self.left_frame,
                self.right_frame,
                self.left_calib,
                self.right_calib,
                None if self.T_world_device is None else self.T_world_device.copy(),
            )


def is_tensorrt_engine_path(path: str) -> bool:
    return path.endswith(".engine") or path.endswith(".plan")


def tensorrt_dtype_to_torch(dtype):
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


class FoundationStereoRuntime:
    def __init__(
        self,
        model_path: str,
        valid_iters: int,
        consistency: bool = False,
        consistency_threshold: float = 1.0,
    ):
        self.model_path = model_path
        self.valid_iters = int(valid_iters)
        self.consistency = bool(consistency)
        self.consistency_threshold = float(consistency_threshold)
        self.kind = "torch"
        self.cfg = None
        self.model = None
        self.supports_consistency_batch2 = True

        fs_repo = "/home/demo/code/projectaria_gen2_depth_from_stereo"
        foundation_path = os.path.join(fs_repo, "FoundationStereo")
        if fs_repo not in sys.path:
            sys.path.insert(0, fs_repo)
        if foundation_path not in sys.path:
            sys.path.insert(0, foundation_path)

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
        return outputs[self.trt_output_names[0]].float()

    def _infer_batch(
        self, left_rect_batch: np.ndarray, right_rect_batch: np.ndarray
    ) -> np.ndarray:
        left_p, right_p, padder = self._prepare_rectified_inputs(
            left_rect_batch, right_rect_batch
        )

        if self.kind == "tensorrt":
            disp_t = self._run_trt_context(self.trt_context, left_p, right_p, stream=None)
            return padder.unpad(disp_t).cpu().numpy()

        autocast_dtype = get_autocast_dtype_for_cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            disp = self.model.forward(
                left_p, right_p, iters=self.cfg.valid_iters, test_mode=True
            )
        return padder.unpad(disp.float()).cpu().numpy()

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
        x_in_right = np.clip(x_in_right_f, 0, w - 1).astype(np.int32)
        rows = np.arange(h)[:, None].repeat(w, axis=1)
        disp_rl_at_match = disp_rl[rows, x_in_right]
        consistent = (
            np.abs(disp_lr - disp_rl_at_match) < self.consistency_threshold
        ) & ~out_of_bounds

        disp_out = disp_lr.astype(np.float32, copy=True)
        disp_out[~consistent] = np.nan
        return disp_out


def make_live_fs_callbacks(state: LiveFsState):
    def device_calib_cb(device_calib):
        left = device_calib.get_camera_calib("slam-front-left")
        right = device_calib.get_camera_calib("slam-front-right")
        if left is None or right is None:
            return
        with state.lock:
            state.left_calib = left
            state.right_calib = right
        print(
            f"==> FS calib received: slam-front-left {left.get_image_size()}, "
            f"slam-front-right {right.get_image_size()}",
            flush=True,
        )

    def slam_cb(image_data: ImageData, image_record: ImageDataRecord):
        camera_id = int(image_record.camera_id)
        if camera_id not in (1, 2):
            return
        arr = image_data.to_numpy_array()
        frame = (arr, int(image_record.capture_timestamp_ns))
        with state.lock:
            if camera_id == 1:
                state.left_frame = frame
            else:
                state.right_frame = frame

    def vio_cb(vio: FrontendOutput):
        M_odo_bi = np.asarray(
            vio.transform_odometry_bodyimu.to_matrix(), dtype=np.float32
        )
        M_bi_dev = np.asarray(
            vio.transform_bodyimu_device.to_matrix(), dtype=np.float32
        )
        with state.lock:
            state.T_world_device = M_odo_bi @ M_bi_dev

    return device_calib_cb, slam_cb, vio_cb


def make_callbacks(state: StreamState):
    def device_calib_cb(device_calib):
        rgb = device_calib.get_camera_calib("camera-rgb")
        T_dev_cam_mat = np.asarray(
            rgb.get_transform_device_camera().to_matrix(), dtype=np.float32
        )
        T_dev_cam = PoseTW.from_matrix(torch.from_numpy(T_dev_cam_mat))
        T_cam_dev = T_dev_cam.inverse().float()
        iw, ih = rgb.get_image_size()
        factory_params = list(rgb.get_projection_params())
        with state.lock:
            state.T_camera_rig = T_cam_dev
            state.rgb_image_size = (int(iw), int(ih))
            state.rgb_intrinsics = factory_params
        print(
            f"==> Device calib received: camera-rgb {iw}x{ih}, "
            f"model={rgb.get_model_name()}, focal={factory_params[0]:.2f}, "
            f"n_params={len(factory_params)}"
        )

    def vio_cb(vio: FrontendOutput):
        M_odo_bi = np.asarray(
            vio.transform_odometry_bodyimu.to_matrix(), dtype=np.float32
        )
        M_bi_dev = np.asarray(
            vio.transform_bodyimu_device.to_matrix(), dtype=np.float32
        )
        M_odo_dev = M_odo_bi @ M_bi_dev
        T_wr = PoseTW.from_matrix(torch.from_numpy(M_odo_dev)).float()
        with state.lock:
            state.T_world_rig = T_wr

    def rgb_cb(image_data: ImageData, image_record: ImageDataRecord):
        arr = image_data.to_numpy_array()
        ts_ns = int(image_record.capture_timestamp_ns)
        with state.lock:
            state.frame = (arr, ts_ns)

    return device_calib_cb, vio_cb, rgb_cb


def make_slam_probe_callback():
    counts: dict[str, int] = {}
    seen: set[str] = set()
    lock = threading.Lock()

    def slam_cb(image_data: ImageData, image_record: ImageDataRecord):
        name = GEN2_CAMERA_ID_TO_LABEL.get(
            int(image_record.camera_id), str(image_record.camera_id)
        )
        arr = image_data.to_numpy_array()
        with lock:
            counts[name] = counts.get(name, 0) + 1
            first = name not in seen
            if first:
                seen.add(name)
        if first:
            print(
                f"==> First {name}: shape={arr.shape}, "
                f"ts={int(image_record.capture_timestamp_ns)}",
                flush=True,
            )

    return slam_cb, counts


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
):
    """Run one OWL+BoxerNet pass on the latest frame.

    Returns dict with keys: viz_2d_bgr, obb_pr_w, T_wr, cam, n_2d, n_3d, ts_ns
    Or None if no frame is available yet.
    """
    frame, T_wr, intr, T_cr, csize = state.snapshot()
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
    sdp_w = boxer_sdp_w if boxer_sdp_w is not None else torch.zeros(0, 3)
    t_start = time.perf_counter()
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

    if use_cuda_timing:
        torch.cuda.synchronize()
    if enable_owl:
        bb2d, scores2d, label_ints, _ = owl.forward(
            owl_img_torch * 255.0,
            rotated=bool(owl_rotated0.item()),
            resize_to_HW=(detector_hw, detector_hw),
        )
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
    if use_cuda_timing:
        torch.cuda.synchronize()
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
    viz_rgb = torch2cv2(img_torch, rotate=rotated_bool, ensure_rgb=True)
    viz_2d = torch2cv2(
        owl_img_torch, rotate=bool(owl_rotated0.item()), ensure_rgb=True
    )
    if bb2d_display.shape[0] > 0:
        bb2_texts = [
            f"{l[:10]} {s:.2f}" for s, l in zip(scores2d_display, labels2d_display)
        ]
        bb2_colors = jet_colors_bgr(scores2d_display)
        viz_2d = render_bb2(
            viz_2d,
            bb2d_display,
            scale=float(bb2_line_width),
            rotated=bool(owl_rotated0.item()),
            texts=bb2_texts,
            clr=bb2_colors,
        )
    owl_title = (
        f"OWLv2 rectified pinhole {detector_hw}x{detector_hw}"
        if rectify_rgb_for_owl_boxes
        else f"OWLv2 {detector_hw}x{detector_hw}"
    )
    put_text(viz_2d, owl_title, scale=0.6, line=0)
    put_text(viz_2d, f"t={ts_ns / 1e9:.3f}s", scale=0.5, line=2)
    viz_3d = viz_rgb.copy()
    t_owl_done = time.perf_counter()

    obb_pr_w = ObbTW(torch.zeros(0, 165))
    scores3d = torch.zeros(0)
    labels3d: list = []
    bb3_rgb_colors = np.zeros((0, 3), dtype=np.float32)
    sdp_patch = None
    sdp_patch_valid = 0
    sdp_patch_median = float("nan")
    n_2d = bb2d.shape[0]
    n_3d = 0
    t_boxer_done = t_owl_done

    if enable_boxer and n_2d > 0:
        if use_cuda_timing:
            torch.cuda.synchronize()
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

        if n_3d > 0:
            sem_ids3d = obb_pr_w.sem_id.squeeze(-1).cpu().numpy().astype(int).tolist()
            bb3_colors, bb3_rgb_colors = get_obb_color_arrays(
                labels3d, sem_ids3d, scores3d, bb3_use_class_colors
            )
            obb_pr_w.set_color(torch.from_numpy(bb3_rgb_colors).float())
            bb3_texts = [
                f"{label[:10]} {float(score):.2f}"
                for label, score in zip(labels3d, scores3d.tolist())
            ]
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
                colors=bb3_colors,
                texts=bb3_texts,
                text_sz=0.35,
                thickness=bb3_line_width,
            )
        if use_cuda_timing:
            torch.cuda.synchronize()
        t_boxer_done = time.perf_counter()

    put_text(viz_3d, "Projected BoxerNet 3DBBs", scale=0.6, line=0)

    return {
        "viz_rgb_bgr": viz_rgb,
        "viz_2d_bgr": viz_2d,
        "viz_3d_bgr": viz_3d,
        "obb_pr_w": obb_pr_w,
        "scores3d": scores3d,
        "labels3d": labels3d,
        "bb3_rgb_colors": bb3_rgb_colors,
        "sdp_patch0": sdp_patch.cpu() if sdp_patch is not None else None,
        "sdp_patch_valid": sdp_patch_valid,
        "sdp_patch_median": sdp_patch_median,
        "owl_ms": (t_owl_done - t_start) * 1000.0,
        "boxer_ms": (t_boxer_done - t_owl_done) * 1000.0,
        "rgb_infer_ms": (t_boxer_done - t_start) * 1000.0,
        "T_wr": T_wr,
        "cam": cam,
        "rotated0": rotated0,
        "n_2d": n_2d,
        "n_3d": n_3d,
        "ts_ns": ts_ns,
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
    HW: int = 960
    detector_hw: int = 960
    init_thresh3d: float = 0.5
    dev: str = "cpu"
    pdtype = torch.float32
    debug_geometry: bool = False
    live_rotation: str = "none"
    fs_state: Optional[LiveFsState] = None
    fs_ckpt: str = ""
    fs_hw: int = 256
    fs_valid_iters: int = 16
    consistency: bool = False
    consistency_threshold: float = 1.0
    fs_point_stride: int = 2
    fs_max_depth: float = 5.0
    vio_world_is_y_up: bool = False
    show_fs_points: bool = True
    show_fs_trajectory: bool = True
    fs_use_depth_colormap: bool = True
    fs_point_size: float = 2.0
    fs_point_alpha: float = 0.85
    fs_line_width: float = 2.0
    fs_frustum_scale: float = 0.12
    enable_owl: bool = True
    enable_boxer: bool = True
    enable_foundation_stereo: bool = True
    rectify_rgb_for_owl_boxes: bool = False
    max_steps: int = 0
    fs_debug_stats: bool = True
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
    show_rgb_boxer: bool = False
    fs_color_points_by_obb: bool = True
    use_fs_for_boxer_sdp: bool = True
    fs_boxer_max_points: int = 12000

    # Layout
    ui_panel_width = 416
    rgb_panel_width = 960
    frustum_scale = 0.12

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
        self._frame_count_t0 = time.time()
        self._render_steps = 0
        self._fps = 0.0
        self._owl_ms = 0.0
        self._boxer_ms = 0.0
        self._total_frame_ms = 0.0
        self._fs_points_in_obbs = 0
        self._boxer_sdp_patch_valid = 0
        self._boxer_sdp_patch_median = float("nan")

        # GL resources
        self._rgb_texture: Optional[moderngl.Texture] = None
        self._rgb_tex_size: Optional[tuple[int, int]] = None
        self._obb_vbo: Optional[moderngl.Buffer] = None
        self._obb_vao: Optional[moderngl.VertexArray] = None
        self._obb_count = 0
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
        self._latest_obbs_3d = ObbTW(torch.zeros(0, 165))
        self._latest_scores_3d = torch.zeros(0)

        # ImGui-controlled state
        self.thresh2d = float(self.owl.min_confidence)
        self.thresh3d = float(self.init_thresh3d)
        self.show_obbs_3d = bool(type(self).show_obbs_3d)
        self.show_frustum = bool(type(self).show_frustum)
        self.show_world_axes = bool(type(self).show_world_axes)
        self.enable_owl = bool(type(self).enable_owl)
        self.enable_boxer = bool(type(self).enable_boxer)
        self.enable_foundation_stereo = bool(type(self).enable_foundation_stereo)
        self.rectify_rgb_for_owl_boxes = bool(type(self).rectify_rgb_for_owl_boxes)
        self.fs_debug_stats = bool(type(self).fs_debug_stats)
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
        self.fs_use_depth_colormap = bool(type(self).fs_use_depth_colormap)
        self.fs_color_points_by_obb = bool(type(self).fs_color_points_by_obb)
        self.use_fs_for_boxer_sdp = bool(type(self).use_fs_for_boxer_sdp)
        self.fs_boxer_max_points = int(type(self).fs_boxer_max_points)
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

        if self.fs_state is not None:
            self._load_foundation_stereo()

    # -- viewport / camera --

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        w, h = self.wnd.size
        vw = max(1, int(w - self.ui_panel_width - self.rgb_panel_width))
        return vw, h

    def _clamp_panel_widths(self) -> None:
        win_w, _ = self.wnd.size
        min_3d_width = 260
        self.ui_panel_width = float(np.clip(self.ui_panel_width, 240, 560))
        self.rgb_panel_width = float(np.clip(self.rgb_panel_width, 320, 1500))
        max_total = max(560, win_w - min_3d_width)
        total = self.ui_panel_width + self.rgb_panel_width
        if total <= max_total:
            return
        overflow = total - max_total
        shrink_rgb = min(overflow, max(0, self.rgb_panel_width - 320))
        self.rgb_panel_width -= shrink_rgb
        overflow -= shrink_rgb
        if overflow > 0:
            self.ui_panel_width = max(240, self.ui_panel_width - overflow)

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

    # -- inference + GPU upload --

    def _maybe_run_inference(self) -> None:
        t_frame_start = time.perf_counter()
        # Cheap snapshot to skip duplicate frames before doing real work
        with self.state.lock:
            frame = self.state.frame
        if frame is None:
            return
        if frame[1] == self._last_ts:
            return

        # Update OWL threshold from the slider before running
        self.owl.min_confidence = float(self.thresh2d)

        boxer_sdp_w = self._get_boxer_sdp_w()
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
        )
        if result is None:
            return

        self._last_ts = result["ts_ns"]
        self._n_2d = result["n_2d"]
        self._n_3d = result["n_3d"]
        self._owl_ms = float(result["owl_ms"])
        self._boxer_ms = float(result["boxer_ms"])
        self._boxer_sdp_patch_valid = int(result["sdp_patch_valid"])
        self._boxer_sdp_patch_median = float(result["sdp_patch_median"])
        self._latest_obbs_3d = self._maybe_convert_obbs_world(result["obb_pr_w"])
        self._latest_scores_3d = result["scores3d"]
        T_world_rgb_cam = result["T_wr"] @ result["cam"].T_camera_rig.inverse()
        self._last_T_world_rgb_cam = (
            T_world_rgb_cam.matrix.detach().cpu().numpy().astype(np.float32)
        )

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
        if self.show_rgb_owl:
            panels.append(result["viz_2d_bgr"])
        if self.show_rgb_boxer:
            panels.append(result["viz_3d_bgr"])
        if not panels:
            panels.append(result["viz_rgb_bgr"])

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
        self._upload_rgb_texture(rgb)

        # Right panel: rebuild 3D line geometry
        self._rebuild_obb_lines(self._latest_obbs_3d, result["scores3d"])
        self._rebuild_frustum(result["cam"], result["T_wr"])

        # FPS counter
        self._frame_count += 1
        now = time.time()
        if now - self._frame_count_t0 >= 1.0:
            self._fps = self._frame_count / (now - self._frame_count_t0)
            self._frame_count = 0
            self._frame_count_t0 = now
        self._total_frame_ms = (time.perf_counter() - t_frame_start) * 1000.0

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

    def _rebuild_obb_lines(self, obbs: ObbTW, scores: torch.Tensor) -> None:
        if self._obb_vbo is not None:
            self._obb_vbo.release()
            self._obb_vbo = None
        if self._obb_vao is not None:
            self._obb_vao.release()
            self._obb_vao = None
        self._obb_count = 0

        N = len(obbs)
        if N == 0:
            return

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
        instance_np = instance.cpu().numpy().astype("f4")
        self._obb_count = len(instance_np)
        self._obb_vbo = self.ctx.buffer(instance_np.tobytes())
        self._obb_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self._obb_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
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
        foundation_path = (
            "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo"
        )
        if foundation_path not in sys.path:
            sys.path.insert(0, foundation_path)
        self.fs_runtime = FoundationStereoRuntime(
            self.fs_ckpt,
            self.fs_valid_iters,
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
        return self.fs_runtime.infer(left_rect, right_rect)

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
        T_world_rect: np.ndarray,
        T_world_device: Optional[np.ndarray] = None,
    ) -> None:
        origin_world = np.asarray(T_world_rect[:3, 3], dtype=np.float32)
        if T_world_device is None:
            T_world_device = T_world_rect
        R_world_device = np.asarray(T_world_device[:3, :3], dtype=np.float32)
        forward_world = R_world_device @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
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
    ) -> None:
        if not self.fs_debug_stats:
            return
        if self._fs_processed != 1 and self._fs_processed - self._fs_debug_last_print < 60:
            return
        self._fs_debug_last_print = self._fs_processed

        disp_valid = disparity[np.isfinite(disparity) & (disparity > 0.0)]
        if disp_valid.size == 0 or z_valid.size == 0 or pts_rect.size == 0:
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

    def _maybe_update_fs_scene(self) -> None:
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

        T_world_device_raw = np.asarray(T_world_device, dtype=np.float32).copy()
        T_world_device = self._maybe_convert_vio_world(T_world_device_raw.copy())
        T_left_device = left_calib.get_transform_device_camera().inverse()
        T_right_device = right_calib.get_transform_device_camera().inverse()
        T_left_right = T_left_device @ T_right_device.inverse()
        T_device_left = np.asarray(T_left_device.inverse().to_matrix(), dtype=np.float32)
        from projectaria_tools.core.image import InterpolationMethod
        from stereo_utils import (
            create_scanline_rectified_cameras,
            disparity_to_depth,
            rectify_stereo_pair,
        )

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

        t_infer = time.time()
        disparity = self._infer_fs_disparity(left_rect, right_rect)
        self._fs_infer_ms = (time.time() - t_infer) * 1000.0
        if disparity is None:
            return

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
        R_left_rect_mat = np.asarray(R_left_rect.to_matrix(), dtype=np.float32)
        T_world_rect = T_world_device @ T_device_left
        T_world_rect[:3, :3] = T_world_rect[:3, :3] @ R_left_rect_mat
        pts_world = (T_world_rect[:3, :3] @ pts_rect.T + T_world_rect[:3, 3:4]).T
        T_world_rect_raw = T_world_device_raw @ T_device_left
        T_world_rect_raw[:3, :3] = T_world_rect_raw[:3, :3] @ R_left_rect_mat
        pts_world_raw = (
            T_world_rect_raw[:3, :3] @ pts_rect.T + T_world_rect_raw[:3, 3:4]
        ).T.astype(np.float32)
        self._fs_overlay_pts_world = pts_world_raw
        self._fs_overlay_depths = z[valid].astype(np.float32)
        self._fs_boxer_pts_world = pts_world_raw
        self._fs_boxer_pair_ts = pair_ts
        intens = left_rect[0:h:stride, 0:w:stride][valid].astype(np.float32) / 255.0
        if self.fs_use_depth_colormap:
            colors = self._depth_jet_colors(z[valid], near=0.1, far=5.0)
        else:
            colors = np.stack([intens, intens, intens], axis=1).astype("f4")
        colors = self._color_fs_points_from_obbs(
            pts_world.astype(np.float32), colors.astype(np.float32)
        )
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
        self.fs_point_count = len(data)
        self._fs_last_pair_ts = pair_ts
        self._fs_processed += 1
        z_valid = z[valid].astype(np.float32)
        self._fs_min_depth = float(np.nanmin(z_valid))
        self._fs_max_depth = float(np.nanmax(z_valid))
        self._fs_mean_depth = float(np.nanmean(z_valid))
        self._fs_median_depth = float(np.nanmedian(z_valid))
        self._fs_last_T_world_device = T_world_device
        self._fs_last_T_world_rect = T_world_rect
        if not self._fs_target_inited:
            self._seed_free_orbit_from_follow_view(T_world_rect, T_world_device)
            self._rebuild_world_axes(self.camera_target)
            self._fs_target_inited = True
            self._target_inited = True
        self._update_fs_geometry(pair_ts, T_world_rect, linear, depth, left_rect)

    # -- render --

    def on_render(self, time_val: float, frame_time: float):
        self._render_steps += 1
        if self.max_steps > 0 and self._render_steps > int(self.max_steps):
            self.wnd.close()
            return
        self._maybe_update_fs_scene()
        self._maybe_run_inference()
        super().on_render(time_val, frame_time)

    def render_3d(self, time_val: float, frame_time: float) -> None:
        full_w, full_h = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)
        # Clear just the right viewport so the rest of the window stays clean
        bg = self.bg_color_options[self.bg_color_index]
        self.ctx.clear(*bg)

        if self.follow_mode and self._fs_last_T_world_rect is not None:
            self._apply_follow_view(
                self._fs_last_T_world_rect, self._fs_last_T_world_device
            )

        _, _, mvp = self.get_camera_matrices()
        mvp_bytes = np.array(mvp, dtype="f4").tobytes()
        viewport = np.array([w, h], dtype="f4")

        if (
            self.show_obbs_3d
            and self._obb_vao is not None
            and self._obb_count > 0
        ):
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["line_width"].write(
                np.array(self.line_width, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["viewport_size"].write(viewport.tobytes())
            self._obb_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._obb_count
            )

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
        win_w, win_h = self.wnd.size
        splitter_w = 8.0

        def render_splitter(name: str, center_x: float, on_drag):
            x0 = int(round(center_x - splitter_w * 0.5))
            imgui.set_next_window_position(x0, 0, imgui.ALWAYS)
            imgui.set_next_window_size(int(splitter_w), win_h, imgui.ALWAYS)
            flags = (
                imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_SCROLLBAR
                | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
            )
            imgui.begin(name, flags=flags)
            draw_list = imgui.get_window_draw_list()
            win_pos = imgui.get_window_position()
            hovered = imgui.is_window_hovered()
            active_col = imgui.get_color_u32_rgba(0.92, 0.92, 0.92, 0.95)
            idle_col = imgui.get_color_u32_rgba(0.62, 0.62, 0.62, 0.85)
            col = active_col if hovered or imgui.is_item_active() else idle_col
            draw_list.add_line(
                win_pos.x + splitter_w * 0.5,
                win_pos.y,
                win_pos.x + splitter_w * 0.5,
                win_pos.y + win_h,
                col,
                2.0,
            )
            imgui.set_cursor_pos((0.0, 0.0))
            imgui.invisible_button(
                f"##{name}_drag", imgui.ImVec2(float(splitter_w), float(win_h))
            )
            if imgui.is_item_active():
                dx = float(imgui.get_io().mouse_delta.x)
                if abs(dx) > 0.0:
                    on_drag(dx)
                    self._clamp_panel_widths()
            imgui.end()

        # Left: control panel
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(int(self.ui_panel_width), win_h, imgui.ALWAYS)
        imgui.begin(
            "Live BoxerNet Controls",
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
        )
        imgui.text(f"FPS: {self._fps:.1f}")
        imgui.text(f"OWL: {self._owl_ms:.1f} ms")
        imgui.text(f"Boxer: {self._boxer_ms:.1f} ms")
        imgui.text(f"FSP: {self._fs_infer_ms:.1f} ms")
        imgui.text(f"Total frame: {self._total_frame_ms:.1f} ms")
        imgui.text(f"2D detections: {self._n_2d}")
        imgui.text(f"3D detections: {self._n_3d}")
        boxer_sdp_count = (
            0
            if self._fs_boxer_pts_world is None
            else min(len(self._fs_boxer_pts_world), int(self.fs_boxer_max_points))
        )
        imgui.text(f"Boxer SDP pts: {boxer_sdp_count}")
        imgui.text(
            f"Boxer SDP patches: {self._boxer_sdp_patch_valid}  median depth: {self._boxer_sdp_patch_median:.3f} m"
        )
        imgui.separator()
        label_w = max(110.0, min(180.0, self.ui_panel_width * 0.42))
        slider_w = max(110.0, float(self.ui_panel_width) - label_w - 36.0)

        def labeled_slider_float(label, value, min_value, max_value, fmt="%.3f"):
            imgui.text(label)
            imgui.same_line(label_w)
            imgui.push_item_width(slider_w)
            changed, value = imgui.slider_float(
                f"##{label}", value, min_value, max_value, fmt
            )
            imgui.pop_item_width()
            return changed, value

        def labeled_slider_int(label, value, min_value, max_value):
            imgui.text(label)
            imgui.same_line(label_w)
            imgui.push_item_width(slider_w)
            changed, value = imgui.slider_int(
                f"##{label}", value, min_value, max_value
            )
            imgui.pop_item_width()
            return changed, value

        _, self.ui_panel_width = labeled_slider_float(
            "UI width", self.ui_panel_width, 240, 560, "%.0f"
        )
        _, self.rgb_panel_width = labeled_slider_float(
            "Image width", self.rgb_panel_width, 320, 1500, "%.0f"
        )
        imgui.separator()
        _, self.thresh2d = labeled_slider_float(
            "2DBB threshold", self.thresh2d, 0.0, 1.0
        )
        _, self.thresh3d = labeled_slider_float(
            "3DBB threshold", self.thresh3d, 0.0, 1.0
        )
        _, self.bb2_line_width = labeled_slider_int(
            "2DBB line width", self.bb2_line_width, 1, 12
        )
        _, self.bb3_image_line_width = labeled_slider_int(
            "3DBB image line width", self.bb3_image_line_width, 1, 12
        )
        _, self.line_width = labeled_slider_float(
            "3D scene OBB line width", self.line_width, 1.0, 10.0
        )
        imgui.separator()
        _, self.enable_owl = imgui.checkbox("Enable OWL", self.enable_owl)
        boxer_enabled = self.enable_boxer
        _, boxer_enabled = imgui.checkbox("Enable Boxer", boxer_enabled)
        self.enable_boxer = boxer_enabled and self.enable_owl
        if not self.enable_owl:
            imgui.text("Boxer requires OWL proposals")
        _, self.use_fs_for_boxer_sdp = imgui.checkbox(
            "Use FS points for Boxer SDP", self.use_fs_for_boxer_sdp
        )
        _, self.rectify_rgb_for_owl_boxes = imgui.checkbox(
            "Rectify RGB for OWL", self.rectify_rgb_for_owl_boxes
        )
        _, self.bb3_use_class_colors = imgui.checkbox(
            "3DBB class/prompt colors", self.bb3_use_class_colors
        )
        if self.fs_state is not None:
            _, self.enable_foundation_stereo = imgui.checkbox(
                "Enable FoundationStereo", self.enable_foundation_stereo
            )
        imgui.separator()
        _, self.show_obbs_3d = imgui.checkbox("Show 3D OBBs", self.show_obbs_3d)
        _, self.show_frustum = imgui.checkbox("Show camera frustum", self.show_frustum)
        _, self.show_world_axes = imgui.checkbox(
            "Show XYZ axes (Z up)", self.show_world_axes
        )
        if self.fs_state is not None:
            imgui.separator()
            _, self.show_fs_points = imgui.checkbox(
                "Show fs points", self.show_fs_points
            )
            _, self.fs_use_depth_colormap = imgui.checkbox(
                "FS jet depth colors", self.fs_use_depth_colormap
            )
            _, self.fs_color_points_by_obb = imgui.checkbox(
                "FS color by containing 3DBB", self.fs_color_points_by_obb
            )
            _, self.fs_point_size = labeled_slider_float(
                "FS point size", self.fs_point_size, 1.0, 8.0
            )
            _, self.fs_point_alpha = labeled_slider_float(
                "FS point alpha", self.fs_point_alpha, 0.05, 1.0
            )
            _, self.fs_boxer_max_points = labeled_slider_int(
                "FS->Boxer max points", self.fs_boxer_max_points, 1000, 50000
            )
            follow_clicked = imgui.button(
                "Follow view" if not self.follow_mode else "Free orbit"
            )
            if follow_clicked:
                self.follow_mode = not self.follow_mode
                if self.follow_mode and self._fs_last_T_world_rect is not None:
                    self._apply_follow_view(self._fs_last_T_world_rect)
            _, self.follow_back = labeled_slider_float(
                "Follow back", self.follow_back, 0.05, 10.0
            )
            _, self.follow_up = labeled_slider_float(
                "Follow up", self.follow_up, 0.0, 10.0
            )
            _, self.follow_lookahead = labeled_slider_float(
                "Look ahead", self.follow_lookahead, 0.0, 3.0
            )
            _, self.follow_smoothing = labeled_slider_float(
                "Follow smoothing", self.follow_smoothing, 0.0, 1.0
            )
        imgui.end()

        # Center: RGB + 2DBB overlay panel
        if self._rgb_texture is not None:
            tex_w, tex_h = self._rgb_tex_size
            imgui.set_next_window_position(int(self.ui_panel_width), 0, imgui.ALWAYS)
            imgui.set_next_window_size(int(self.rgb_panel_width), win_h, imgui.ALWAYS)
            expanded, _ = imgui.begin(
                "RGB",
                flags=imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
            )
            if expanded:
                _, self.show_rgb_fs_points = imgui.checkbox(
                    "Show FS Points", self.show_rgb_fs_points
                )
                _, self.show_rgb_fs = imgui.checkbox("Show FS SDP", self.show_rgb_fs)
                _, self.show_rgb_owl = imgui.checkbox("Show OWL", self.show_rgb_owl)
                _, self.show_rgb_boxer = imgui.checkbox(
                    "Show Boxer 3DBB", self.show_rgb_boxer
                )
                imgui.separator()
                avail_w, avail_h = imgui.get_content_region_available()
                scale = min(avail_w / tex_w, avail_h / tex_h)
                imgui.image(
                    self._rgb_texture.glo, tex_w * scale, tex_h * scale
                )
            imgui.end()

        render_splitter(
            "##splitter_ui_rgb",
            float(self.ui_panel_width),
            lambda dx: setattr(self, "ui_panel_width", float(self.ui_panel_width) + dx),
        )
        render_splitter(
            "##splitter_rgb_3d",
            float(self.ui_panel_width + self.rgb_panel_width),
            lambda dx: setattr(
                self, "rgb_panel_width", float(self.rgb_panel_width) + dx
            ),
        )

    def on_key_event(self, key, action, modifiers):
        super().on_key_event(key, action, modifiers)
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key in (keys.Q, keys.ESCAPE):
            self.wnd.close()


class LiveFoundationStereoViewer(OrbitViewer):
    title = "Live FoundationStereo"
    window_size = (3300, 2100)

    fs_state: LiveFsState = None
    fs_ckpt: str = ""
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
        foundation_path = (
            "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo"
        )
        if foundation_path not in sys.path:
            sys.path.insert(0, foundation_path)
        self.fs_runtime = FoundationStereoRuntime(
            self.fs_ckpt,
            self.fs_valid_iters,
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
        fs_repo = "/home/demo/code/projectaria_gen2_depth_from_stereo"
        if fs_repo not in sys.path:
            sys.path.insert(0, fs_repo)
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
    p.add_argument("--profile_name", type=str, default="profile9")
    p.add_argument("--wifi", action="store_true")
    p.add_argument("--ip", type=str, default=None)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--labels", type=str, default="lvisplus")
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
    p.add_argument("--fs_hw", type=int, default=256)
    p.add_argument(
        "--fs_point_stride", type=int, default=2
    )
    p.add_argument("--fs_max_depth", type=float, default=5.0)
    p.add_argument("--fs_valid_iters", type=int, default=16)
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
    p.add_argument(
        "--fs_ckpt",
        type=str,
        default=(
            "/home/demo/code/projectaria_gen2_depth_from_stereo/FoundationStereo/"
            "pretrained_models/11-33-40/model_best_bp2.pth"
        ),
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
    return p.parse_args()


def pick_device(force_cpu: bool) -> str:
    if torch.backends.mps.is_available() and not force_cpu:
        return "mps"
    if torch.cuda.is_available() and not force_cpu:
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    ensure_aria_tools_on_path()

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
    device.start_streaming()

    state = StreamState()
    fs_state = LiveFsState()
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
        LiveBoxerViewer.HW = HW
        LiveBoxerViewer.detector_hw = args.detector_hw
        LiveBoxerViewer.rectify_rgb_for_owl_boxes = bool(args.rectify)
        LiveBoxerViewer.init_thresh3d = args.thresh3d
        LiveBoxerViewer.dev = dev
        LiveBoxerViewer.pdtype = pdtype
        LiveBoxerViewer.debug_geometry = args.debug_geometry
        LiveBoxerViewer.live_rotation = args.live_rotation
        LiveBoxerViewer.max_steps = int(args.max_steps)
        LiveBoxerViewer.fs_state = fs_state
        LiveBoxerViewer.fs_ckpt = args.fs_ckpt
        LiveBoxerViewer.fs_hw = int(args.fs_hw)
        LiveBoxerViewer.fs_valid_iters = int(args.fs_valid_iters)
        LiveBoxerViewer.consistency = bool(args.consistency)
        LiveBoxerViewer.consistency_threshold = float(args.consistency_threshold)
        LiveBoxerViewer.fs_point_stride = int(args.fs_point_stride)
        LiveBoxerViewer.fs_max_depth = float(args.fs_max_depth)
        if args.fs or args.owl or args.boxer:
            LiveBoxerViewer.enable_foundation_stereo = bool(args.fs)
            LiveBoxerViewer.enable_boxer = bool(args.boxer)
            LiveBoxerViewer.enable_owl = bool(args.owl or args.boxer)

        print("==> Launching viewer.")
        launch_viewer(LiveBoxerViewer)
    finally:
        device.stop_streaming()
        rx.stop_server()


if __name__ == "__main__":
    main()
