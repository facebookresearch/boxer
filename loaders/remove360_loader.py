# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Loader for Remove360 sequences that have been processed with COLMAP.

Directory layout expected:
    <seq_dir>/
        images/           train images (before_*.jpg) — or symlink to train/
        sparse/0/         COLMAP output (cameras.bin, images.bin, points3D.bin)
        masks/            optional per-image object masks (before_*.png)

Camera poses and intrinsics come from COLMAP.  Gravity is estimated from the
average camera up-vector across all frames.  Semi-dense points (SDP) come from
the COLMAP sparse 3D point cloud.
"""

import os
import struct
from typing import Optional

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


# ── Minimal COLMAP binary parser (drop-in replacement for pycolmap.SceneManager) ──
# pycolmap on PyPI doesn't expose scene_manager.SceneManager, and the True-Price
# fork isn't always installable. This in-loader parser reads cameras.bin /
# images.bin / points3D.bin directly and exposes the same .images / .cameras /
# .points3D interface the loader expects.

_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _qvec_wxyz_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


class _ColmapImage:
    def __init__(self, image_id, qvec, tvec, camera_id, name):
        self.id = image_id
        self.qvec = np.array(qvec, dtype=np.float64)   # wxyz
        self.tvec = np.array(tvec, dtype=np.float64)
        self.camera_id = camera_id
        self.name = name
        self._R = _qvec_wxyz_to_R(self.qvec)            # world→camera

    def R(self):
        return self._R

    def C(self):
        # camera center in world coords = -R.T @ tvec
        return -self._R.T @ self.tvec


class _ColmapCamera:
    def __init__(self, cam_id, model_name, width, height, params):
        self.id = cam_id
        self.model_name = model_name
        self.width = int(width)
        self.height = int(height)
        params = list(params)
        if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            self.fx = self.fy = float(params[0])
            self.cx = float(params[1])
            self.cy = float(params[2])
        else:  # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, etc.
            self.fx = float(params[0])
            self.fy = float(params[1])
            self.cx = float(params[2])
            self.cy = float(params[3])


class _ColmapSceneManager:
    """Minimal stand-in for pycolmap.scene_manager.SceneManager."""

    def __init__(self, sparse_dir):
        self.sparse_dir = sparse_dir
        self.images = {}     # image_id → _ColmapImage
        self.cameras = {}    # camera_id → _ColmapCamera
        self.points3D = np.zeros((0, 3), dtype=np.float64)

    def load_cameras(self):
        path = os.path.join(self.sparse_dir, "cameras.bin")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                cam_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<I", f.read(4))[0]
                w = struct.unpack("<Q", f.read(8))[0]
                h = struct.unpack("<Q", f.read(8))[0]
                model_name, n_params = _COLMAP_CAMERA_MODELS[model_id]
                params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
                self.cameras[cam_id] = _ColmapCamera(cam_id, model_name, w, h, params)

    def load_images(self):
        path = os.path.join(self.sparse_dir, "images.bin")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = struct.unpack("<dddd", f.read(32))
                tvec = struct.unpack("<ddd", f.read(24))
                cam_id = struct.unpack("<I", f.read(4))[0]
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name += c
                name = name.decode("utf-8")
                n2d = struct.unpack("<Q", f.read(8))[0]
                # skip 2D-3D correspondences: each is xy(2*d) + point3D_id(q)
                f.seek(n2d * (8 + 8 + 8), 1)
                self.images[image_id] = _ColmapImage(image_id, qvec, tvec, cam_id, name)

    def load_points3D(self):
        path = os.path.join(self.sparse_dir, "points3D.bin")
        if not os.path.exists(path):
            self.points3D = np.zeros((0, 3), dtype=np.float64)
            return
        xyzs = []
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                struct.unpack("<Q", f.read(8))                  # point3D_id
                xyzs.append(struct.unpack("<ddd", f.read(24)))  # xyz
                f.read(3)                                        # rgb
                f.read(8)                                        # error
                tl = struct.unpack("<Q", f.read(8))[0]
                f.seek(tl * (4 + 4), 1)                          # track entries
        self.points3D = np.array(xyzs, dtype=np.float64)


def _quat_wxyz_to_rotation(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] → 3×3 rotation matrix (world→cam)."""
    qw, qx, qy, qz = q_wxyz
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def _estimate_gravity_colmap(images: dict) -> np.ndarray:
    """Estimate world-space gravity direction from COLMAP camera poses.

    COLMAP camera convention: Y axis points *down* in camera space.
    World "down" ≈ average of each camera's down axis in world space:
        down_w = R_wc @ [0, 1, 0]   where R_wc is camera→world rotation.

    Returns:
        (3,) unit vector pointing in the direction of gravity (down) in world space.
    """
    down_vectors = []
    cam_down = np.array([0.0, 1.0, 0.0])  # Y is down in camera space
    for img in images.values():
        R_wc = img.R().T  # camera→world rotation
        down_w = R_wc @ cam_down
        down_vectors.append(down_w)
    mean_down = np.mean(down_vectors, axis=0)
    norm = np.linalg.norm(mean_down)
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0])
    return (mean_down / norm).astype(np.float32)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix R such that R @ a ≈ b (both unit vectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180° rotation around a perpendicular axis
        orth = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(a, orth)
        ax /= np.linalg.norm(ax)
        return (2.0 * np.outer(ax, ax) - np.eye(3)).astype(np.float32)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)
    return R.astype(np.float32)


class Remove360Loader(BaseLoader):
    """Load a Remove360 sequence from COLMAP-reconstructed camera poses.

    Args:
        seq_dir:     Path to sequence directory (with images/ and sparse/0/).
        skip_frames: Load every N-th frame.
        max_frames:  Maximum number of frames to load.
        start_frame: 0-based index of first frame to load.
        use_masks:   Whether to load masks (stored as datum['mask0']).
    """

    camera = "rgb"
    device_name = "Remove360"

    def __init__(
        self,
        seq_dir: str,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        use_masks: bool = False,
    ):
        seq_dir = os.path.expanduser(seq_dir)
        if not os.path.isabs(seq_dir) and not os.path.exists(seq_dir):
            from utils.demo_utils import SAMPLE_DATA_PATH
            seq_dir = os.path.join(SAMPLE_DATA_PATH, seq_dir)

        self.seq_dir = seq_dir
        self.use_masks = use_masks
        self.resize = None

        # ── Locate COLMAP sparse model ─────────────────────────────────────────
        colmap_dir = os.path.join(seq_dir, "sparse", "0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(seq_dir, "sparse")
        if not os.path.exists(colmap_dir):
            raise FileNotFoundError(
                f"COLMAP sparse model not found in {seq_dir}/sparse[/0]. "
                "Run scripts/run_colmap.sh first."
            )

        # ── Load COLMAP cameras and images ─────────────────────────────────────
        manager = _ColmapSceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Sort images by filename for deterministic ordering
        sorted_image_ids = sorted(
            manager.images.keys(),
            key=lambda iid: manager.images[iid].name,
        )
        sorted_image_ids = sorted_image_ids[start_frame::skip_frames]
        if max_frames is not None:
            sorted_image_ids = sorted_image_ids[:max_frames]

        self.image_ids = sorted_image_ids
        self.length = len(self.image_ids)
        self.index = 0
        self.manager = manager

        # ── Locate images directory ────────────────────────────────────────────
        images_dir = os.path.join(seq_dir, "images")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(seq_dir, "train")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found in {seq_dir}")
        self.images_dir = images_dir

        # ── Mask directory ─────────────────────────────────────────────────────
        self.masks_dir = os.path.join(seq_dir, "masks")

        # ── World offset: translate so first camera is near origin ─────────────
        first_img = manager.images[self.image_ids[0]]
        first_R = first_img.R()  # world→camera
        first_t = first_img.tvec
        self.world_offset = (-first_R.T @ first_t).astype(np.float32)  # cam center

        # ── Gravity alignment ─────────────────────────────────────────────────
        # BoxerNet assumes gravity = [0, 0, -1] in world space (VIO convention).
        # We rotate the entire COLMAP world so that the estimated gravity maps to
        # [0, 0, -1], making BoxerNet's default assumption correct.
        gravity_est = _estimate_gravity_colmap(manager.images)
        self.R_fix = _rotation_between(gravity_est, np.array([0.0, 0.0, -1.0]))
        gravity_fixed = self.R_fix @ gravity_est
        print(f"==> Estimated gravity (COLMAP world): {gravity_est}")
        print(f"==> Gravity after R_fix alignment:    {gravity_fixed}  (target [0,0,-1])")

        # ── Sparse 3D points for SDP (recentered + gravity-aligned) ───────────
        if manager.points3D.shape[0] > 0:
            pts = manager.points3D.astype(np.float32) - self.world_offset
            self.points3D_w = (self.R_fix @ pts.T).T
        else:
            self.points3D_w = np.zeros((0, 3), dtype=np.float32)

        print(
            f"Remove360Loader: {os.path.basename(seq_dir)}, "
            f"{self.length} frames, "
            f"{self.points3D_w.shape[0]} sparse 3D points"
        )

        # ── Pre-compute per-frame Ts_wc, cams, timestamps for the viewer ──────
        # build_seq_ctx (utils/viewer_3d.py) needs: timestamp_ns, Ts_wc, cams,
        # sdp_global, and cams[i].T_camera_rig (identity for monocular).
        self.timestamp_ns = []
        self.Ts_wc = []
        self.cams = []
        for image_id in self.image_ids:
            img_info = manager.images[image_id]
            cam = manager.cameras[img_info.camera_id]

            # Synthetic ns timestamp from image_id (monotonic if names are sortable)
            self.timestamp_ns.append(int(image_id) * 1_000_000)

            # Per-frame aligned world→camera pose, stored as PoseTW
            R_wc_np = img_info.R().T.astype(np.float32)
            C_w = img_info.C().astype(np.float32)
            R_wc_aligned = self.R_fix @ R_wc_np
            C_w_aligned = self.R_fix @ (C_w - self.world_offset)
            self.Ts_wc.append(
                PoseTW(
                    torch.tensor(
                        [*R_wc_aligned.flatten(), *C_w_aligned], dtype=torch.float32
                    )
                )
            )

            # Per-frame CameraTW (pinhole_from_K already sets T_camera_rig = identity)
            W = int(cam.width)
            H = int(cam.height)
            boxer_cam = self.pinhole_from_K(
                W, H, float(cam.fx), float(cam.fy),
                float(cam.cx), float(cam.cy), valid_radius=(W, H),
            ).float()
            self.cams.append(boxer_cam)

        # Aligned sparse points as torch tensor (build_seq_ctx expects sdp_global)
        self.sdp_global = (
            torch.from_numpy(self.points3D_w).float()
            if self.points3D_w.shape[0] > 0
            else torch.zeros(0, 3, dtype=torch.float32)
        )

        # Boxer's viewer-side code sometimes peeks at these (Aria-loader habits)
        self.is_nebula = True  # treat as a pre-baked scene, no live streaming
        self.traj = self.Ts_wc  # alias
        self.pose_ts = np.array(self.timestamp_ns, dtype=np.int64)
        self.calibs = [self.cams]  # match Aria's nested-list shape
        self.calib_ts = self.pose_ts

        self._init_prefetch()

    # ── Frame loading ──────────────────────────────────────────────────────────

    def load(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        img_info = self.manager.images[image_id]
        cam = self.manager.cameras[img_info.camera_id]

        datum: dict = {}

        # ── Image ──────────────────────────────────────────────────────────────
        img_path = os.path.join(self.images_dir, img_info.name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = img_rgb.shape[:2]

        if self.resize is not None:
            target = self.resize
            img_rgb = cv2.resize(img_rgb, (target, target), interpolation=cv2.INTER_LINEAR)
            scale_x = target / W_orig
            scale_y = target / H_orig
            HH, WW = target, target
        else:
            scale_x = scale_y = 1.0
            HH, WW = H_orig, W_orig

        datum["img0"] = self.img_to_tensor(img_rgb)

        # ── Camera intrinsics ──────────────────────────────────────────────────
        fx = cam.fx * scale_x
        fy = cam.fy * scale_y
        cx = cam.cx * scale_x
        cy = cam.cy * scale_y
        boxer_cam = self.pinhole_from_K(WW, HH, fx, fy, cx, cy, valid_radius=(WW, HH))
        datum["cam0"] = boxer_cam.float()

        # ── Pose (gravity-aligned world frame, recentered) ────────────────────
        R_wc_np = img_info.R().T.astype(np.float32)        # camera→world
        C_w = img_info.C().astype(np.float32)               # camera center in world
        # Apply gravity-alignment rotation: R_fix rotates world so gravity → [0,0,-1]
        R_wc_aligned = self.R_fix @ R_wc_np
        C_w_aligned = self.R_fix @ (C_w - self.world_offset)

        R_flat = R_wc_aligned.flatten()
        t_vec = C_w_aligned
        datum["T_world_rig0"] = PoseTW(
            torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
        )

        # ── Semi-dense points from COLMAP sparse cloud ─────────────────────────
        # Sample up to 10000 points visible in this frame (or all if fewer)
        if self.points3D_w.shape[0] > 0:
            # Use all reconstructed points (no per-image visibility filtering to
            # keep it simple; Boxer handles noisy / behind-camera points)
            pts = self.points3D_w
            if pts.shape[0] > 10000:
                idx_sample = np.random.choice(pts.shape[0], 10000, replace=False)
                pts = pts[idx_sample]
            datum["sdp_w"] = torch.from_numpy(pts).float()
        else:
            datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)

        # ── Metadata ─────────────────────────────────────────────────────────────
        # Use image_id as a nanosecond-like timestamp
        datum["time_ns0"] = int(image_id) * 1_000_000
        datum["rotated0"] = torch.tensor(False).reshape(1)
        datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
        datum["obbs"] = ObbTW(torch.zeros(0, 165))
        datum["gt_labels"] = []

        # ── Optional mask ──────────────────────────────────────────────────────
        if self.use_masks:
            mask_name = os.path.splitext(img_info.name)[0] + ".png"
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_gray is not None:
                    if self.resize is not None:
                        mask_gray = cv2.resize(
                            mask_gray, (WW, HH), interpolation=cv2.INTER_NEAREST
                        )
                    datum["mask0"] = torch.from_numpy(mask_gray).float() / 255.0

        return datum
