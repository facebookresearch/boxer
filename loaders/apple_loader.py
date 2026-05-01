# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Data loader for Apple-style RGB-D captures.

Expects a directory laid out as:
    <seq>/
        cam_K.txt          # 3x3 pinhole intrinsics, shared across frames
        rgb/NNNNNNN.png    # uint8 HxWx3
        depth/NNNNNNN.png  # uint16, depth in millimeters (0 / 65535 = invalid)

No camera trajectory is provided. Per-frame world-from-rig is built from a
gravity estimate (default: GeoCalib) so that boxer's voxel frame (world +Z =
up) is correct. Translation is left at zero — absolute world position is
meaningless without a trajectory, so cross-frame fusion/tracking is disabled.
"""

import os
from typing import Optional, Union

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


def _R_wc_from_up_cam(up_cam: np.ndarray) -> np.ndarray:
    """Build R_wc (cam→world) so that world +Z aligns with up_cam (in cam frame).

    Yaw around the up axis is fixed by projecting cam X onto the horizontal
    plane (or cam Y if cam X is too close to the up axis).
    """
    up = up_cam / (np.linalg.norm(up_cam) + 1e-8)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ up)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_in_cam = ref - (ref @ up) * up
    x_in_cam /= np.linalg.norm(x_in_cam) + 1e-8
    y_in_cam = np.cross(up, x_in_cam)
    R_cw = np.stack([x_in_cam, y_in_cam, up], axis=1)  # columns = world axes in cam
    return R_cw.T.astype(np.float32)


class AppleLoader(BaseLoader):
    def __init__(
        self,
        seq_dir: str,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 1,
        resize: Optional[int] = None,
        max_depth_m: float = 10.0,
        gravity_method: Union[str, tuple, list] = "geocalib",
        geocalib_device: Optional[str] = None,
    ):
        seq_dir = os.path.expanduser(seq_dir)
        if not os.path.isabs(seq_dir) and not os.path.exists(seq_dir):
            from utils.demo_utils import SAMPLE_DATA_PATH

            seq_dir = os.path.join(SAMPLE_DATA_PATH, seq_dir)
        self.seq_dir = seq_dir
        self.seq_name = os.path.basename(seq_dir.rstrip("/"))
        self.camera = "rgb"
        self.device_name = "apple"
        self.resize = resize
        self.max_depth_m = max_depth_m
        self.gravity_method = gravity_method

        # Load shared intrinsics.
        K = np.loadtxt(os.path.join(seq_dir, "cam_K.txt"))
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        # Discover frames from rgb/.
        rgb_dir = os.path.join(seq_dir, "rgb")
        depth_dir = os.path.join(seq_dir, "depth")
        rgb_files = sorted(
            f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg"))
        )
        frame_ids = [os.path.splitext(f)[0] for f in rgb_files]

        # Keep only frames that have a matching depth file.
        frame_ids = [
            fid for fid in frame_ids if os.path.exists(os.path.join(depth_dir, fid + ".png"))
        ]

        # start_frame is 1-indexed for parity with ScanNetLoader.
        frame_ids = frame_ids[start_frame - 1 :: skip_frames]
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]

        self.frame_ids = frame_ids
        self.length = len(frame_ids)
        self.index = 0

        # No GT OBBs in this dataset.
        self.sem_id_to_name = {}
        self.sem_name_to_id = {}

        # Set up gravity estimator.
        self._geocalib = None
        self._geocalib_device = geocalib_device
        self._fixed_up_cam: Optional[np.ndarray] = None
        if gravity_method == "geocalib":
            try:
                from geocalib import GeoCalib
            except ImportError as e:
                raise ImportError(
                    "AppleLoader needs the `geocalib` package for gravity_method='geocalib'. "
                    "Install with `pip install -e ~/code/GeoCalib` or "
                    "`pip install 'geocalib @ git+https://github.com/cvg/GeoCalib'`."
                ) from e
            if self._geocalib_device is None:
                if torch.backends.mps.is_available():
                    self._geocalib_device = "mps"
                elif torch.cuda.is_available():
                    self._geocalib_device = "cuda"
                else:
                    self._geocalib_device = "cpu"
            self._geocalib = GeoCalib(weights="pinhole").to(self._geocalib_device).eval()
        elif gravity_method == "identity":
            print(
                "AppleLoader: gravity_method='identity' — BoxerNet outputs will be "
                "tilted relative to true world up."
            )
        elif isinstance(gravity_method, (tuple, list)) and len(gravity_method) == 3:
            # User supplies gravity (down) direction in cam frame; up = -gravity.
            g = np.array(gravity_method, dtype=np.float64)
            g /= np.linalg.norm(g) + 1e-8
            self._fixed_up_cam = (-g).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown gravity_method: {gravity_method!r}. "
                "Expected 'geocalib', 'identity', or (gx, gy, gz)."
            )

        print(f"AppleLoader: {self.seq_name}, {self.length} frames from {seq_dir}")
        print(f"  gravity_method={gravity_method}")

        self._init_prefetch()

    def _estimate_up_cam(self, img_rgb_native: np.ndarray) -> np.ndarray:
        """Return the unit up-direction in camera frame for one frame."""
        if self.gravity_method == "identity":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self._fixed_up_cam is not None:
            return self._fixed_up_cam

        # GeoCalib: pass the *native-resolution* image (and matching focal prior)
        # so the estimate is in the original camera frame.
        img_t = (
            torch.from_numpy(img_rgb_native).permute(2, 0, 1).float() / 255.0
        ).to(self._geocalib_device)
        focal_prior = torch.tensor((self.fx + self.fy) / 2.0).to(self._geocalib_device)
        result = self._geocalib.calibrate(
            img_t, camera_model="pinhole", priors={"focal": focal_prior}
        )
        up_cam = result["gravity"].vec3d.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return up_cam

    def load(self, idx):
        frame_id = self.frame_ids[idx]
        datum = {}

        # RGB at native resolution (used for gravity estimation before resize).
        rgb_path = os.path.join(self.seq_dir, "rgb", f"{frame_id}.png")
        img_bgr = cv2.imread(rgb_path)
        img_rgb_native = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        HH, WW = img_rgb_native.shape[:2]

        if self.resize is not None:
            resizeH = resizeW = self.resize
            scale_x = resizeW / WW
            scale_y = resizeH / HH
            img_rgb = cv2.resize(
                img_rgb_native, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR
            )
        else:
            resizeH, resizeW = HH, WW
            scale_x = scale_y = 1.0
            img_rgb = img_rgb_native

        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        datum["img0"] = self.img_to_tensor(img_rgb)

        # Depth (uint16 mm) → meters, with sentinel filtering.
        depth_path = os.path.join(self.seq_dir, "depth", f"{frame_id}.png")
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / 1000.0
        # uint16 max (65535) is a saturation/invalid sentinel; also clip far depth.
        depth_m[(depth_raw == 65535) | (depth_m > self.max_depth_m)] = 0.0
        if depth_m.shape[:2] != (resizeH, resizeW):
            depth_m = cv2.resize(
                depth_m, (resizeW, resizeH), interpolation=cv2.INTER_NEAREST
            )

        # Pinhole camera (identity T_camera_rig).
        cam = self.pinhole_from_K(
            resizeW, resizeH, fx, fy, cx, cy, valid_radius=(resizeW, resizeH)
        )
        datum["cam0"] = cam.float()

        # Per-frame gravity → R_wc such that world +Z = up.
        up_cam = self._estimate_up_cam(img_rgb_native)
        R_wc = _R_wc_from_up_cam(up_cam)
        t_wc = np.zeros(3, dtype=np.float32)
        T_wr_data = torch.tensor([*R_wc.flatten(), *t_wc], dtype=torch.float32)
        datum["T_world_rig0"] = PoseTW(T_wr_data)

        # Semi-dense points from depth, transformed by the same R_wc.
        datum["sdp_w"] = self.sdp_from_depth(
            depth_m, fx, fy, cx, cy, R_wc, t_wc,
        )

        # No GT OBBs/2D boxes for this dataset.
        datum["obbs"] = ObbTW(torch.zeros(0, 165))
        datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
        datum["gt_labels"] = []

        # Synthesize a 30fps timestamp from frame index (~33.3ms per frame) so
        # downstream FPS computations in run_boxer behave sensibly.
        datum["time_ns0"] = int(frame_id) * 33_333_333
        datum["rotated0"] = torch.tensor(False).reshape(1)

        return datum
