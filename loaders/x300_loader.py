# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Single-frame loader for x300-style minimal samples.

Expects a directory containing:
    image.jpg          # pinhole RGB
    rt.json            # intrinsics + T_global_cam (4x4)
    sdp_3000.npz       # {"xyz_global": (N, 3) float32} world-frame SDP cloud
    points.ply         # optional dense LiDAR cloud (used when sdp_source="dense")

The pose convention is `p_global = T_global_cam * p_cam` and the world frame
is Z-up, Y-forward — already aligned with BoxerNet's assumed gravity, so no
GeoCalib step is needed.
"""

import json
import os
from typing import Optional, Union

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


def _load_sdp_npz(path: str) -> torch.Tensor:
    data = np.load(path)
    if "xyz_global" not in data:
        raise KeyError(f"{path} missing 'xyz_global' array (got {list(data.keys())})")
    return torch.from_numpy(data["xyz_global"].astype(np.float32))


def _load_sdp_ply(path: str) -> torch.Tensor:
    """Minimal ASCII/binary PLY reader for an x,y,z point cloud."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.strip() == b"end_header":
                break
        header_str = b"".join(header).decode("ascii", errors="replace")
        is_binary = "format binary_little_endian" in header_str
        n_pts = 0
        for line in header_str.splitlines():
            if line.startswith("element vertex"):
                n_pts = int(line.split()[-1])
        if is_binary:
            arr = np.frombuffer(f.read(n_pts * 12), dtype=np.float32).reshape(n_pts, 3)
        else:
            arr = np.loadtxt(f, dtype=np.float32, max_rows=n_pts).reshape(n_pts, 3)
    return torch.from_numpy(arr.astype(np.float32))


class X300Loader(BaseLoader):
    def __init__(
        self,
        seq_dir: str,
        sdp_source: str = "npz",
        resize: Optional[Union[int, tuple]] = None,
        sdp_dense_samples: int = 3000,
    ):
        seq_dir = os.path.expanduser(seq_dir)
        if not os.path.isabs(seq_dir) and not os.path.exists(seq_dir):
            from utils.demo_utils import SAMPLE_DATA_PATH

            seq_dir = os.path.join(SAMPLE_DATA_PATH, seq_dir)
        self.seq_dir = seq_dir
        self.seq_name = os.path.basename(seq_dir.rstrip("/"))
        self.camera = "rgb"
        self.device_name = "x300"
        self.resize = resize
        self.length = 1
        self.index = 0
        self.sem_id_to_name = {}
        self.sem_name_to_id = {}

        # Parse rt.json once.
        with open(os.path.join(seq_dir, "rt.json"), "r") as f:
            rt = json.load(f)
        intr = rt["intrinsics"]
        self.W = int(intr["width"])
        self.H = int(intr["height"])
        self.fx = float(intr["fx"])
        self.fy = float(intr["fy"])
        self.cx = float(intr["cx"])
        self.cy = float(intr["cy"])
        T_wc = np.array(rt["T_global_cam"], dtype=np.float32)
        self.R_wc = T_wc[:3, :3].copy()
        self.t_wc = T_wc[:3, 3].copy()

        # Load SDP cloud (already in world frame).
        if sdp_source == "npz":
            self.sdp_w = _load_sdp_npz(os.path.join(seq_dir, "sdp_3000.npz"))
        elif sdp_source == "ply":
            self.sdp_w = _load_sdp_ply(os.path.join(seq_dir, "sdp_3000.ply"))
        elif sdp_source == "dense":
            dense = _load_sdp_ply(os.path.join(seq_dir, "points.ply"))
            n = dense.shape[0]
            k = min(sdp_dense_samples, n)
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=k, replace=False)
            self.sdp_w = dense[idx]
        else:
            raise ValueError(f"Unknown sdp_source: {sdp_source!r}")

        print(
            f"X300Loader: {self.seq_name}, 1 frame, sdp_source={sdp_source}, "
            f"sdp_pts={self.sdp_w.shape[0]}"
        )

        self._init_prefetch()

    def load(self, idx):
        # idx is always 0 (single-frame loader).
        del idx

        img_bgr = cv2.imread(os.path.join(self.seq_dir, "image.jpg"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        HH, WW = img_rgb.shape[:2]

        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)):
                resizeH, resizeW = int(self.resize[0]), int(self.resize[1])
            else:
                resizeH = resizeW = int(self.resize)
            scale_x = resizeW / WW
            scale_y = resizeH / HH
            img_rgb = cv2.resize(
                img_rgb, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR
            )
        else:
            resizeH, resizeW = HH, WW
            scale_x = scale_y = 1.0

        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        cam = self.pinhole_from_K(
            resizeW, resizeH, fx, fy, cx, cy, valid_radius=(resizeW, resizeH)
        )

        T_wr_data = torch.tensor(
            [*self.R_wc.flatten(), *self.t_wc], dtype=torch.float32
        )

        return {
            "img0": self.img_to_tensor(img_rgb),
            "cam0": cam.float(),
            "T_world_rig0": PoseTW(T_wr_data),
            "sdp_w": self.sdp_w.clone(),
            "obbs": ObbTW(torch.zeros(0, 165)),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "gt_labels": [],
            "time_ns0": 0,
            # `rotated=True` means "image was captured 90° CW from upright" —
            # boxer rotates it back inside DINO and the visualizer. The shipped
            # `image.jpg` is already upright, so False is correct (despite the
            # dataset README mentioning rotated0=True).
            "rotated0": torch.tensor(False).reshape(1),
        }
