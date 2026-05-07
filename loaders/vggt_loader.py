# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Loader that runs VGGT (https://github.com/facebookresearch/vggt) on an
unposed folder of RGB images to obtain extrinsics, intrinsics, and depth,
then yields boxer datums in the standard format.

Layout expected:
    <seq>/
        *.png|*.jpg|*.jpeg

VGGT predictions are cached to ``<seq>/vggt_cache.npz`` keyed on the
selected files' (basename, size, mtime) and conf threshold; rerunning with
the same inputs skips both the VGGT and GeoCalib passes.

World frame is gravity-aligned: GeoCalib runs on frame 0 to recover the
up-direction in cam0, the VGGT trajectory is rigidly rotated so cam0 sits
at the origin with world +Z = up. Relative geometry between frames is
preserved.
"""

import json
import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from loaders.apple_loader import _R_wc_from_up_cam
from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


VGGT_LOAD_RESOLUTION = 1024
VGGT_INFER_RESOLUTION = 518
CACHE_FILENAME = "vggt_cache.npz"


def _list_images(seq_dir: str):
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(
        f for f in os.listdir(seq_dir) if f.lower().endswith(exts)
    )
    return files


def _cache_key(seq_dir: str, frame_files, conf_thres: float,
               gravity_method: str) -> str:
    parts = []
    for f in frame_files:
        st = os.stat(os.path.join(seq_dir, f))
        parts.append({"name": f, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return json.dumps(
        {"v": 2, "frames": parts, "conf_thres": float(conf_thres),
         "gravity_method": gravity_method},
        sort_keys=True,
    )


def _run_vggt(image_paths, device, dtype):
    """Run VGGT once on a list of image paths.

    Returns:
        extrinsic: (N, 3, 4) np.float32 in OpenCV cam-from-world, in 518-coords
        intrinsic: (N, 3, 3) np.float32 in 518-coords (square frame)
        depth:    (N, 518, 518) np.float32, raw model output
        depth_conf: (N, 518, 518) np.float32
        original_coords: (N, 6) np.float32 — [x1, y1, x2, y2, W, H] at 1024-load-resolution
    """
    try:
        from vggt.models.vggt import VGGT  # noqa: F401
    except ImportError:
        # Fall back to the sibling checkout if it's not installed as a package.
        import sys as _sys
        _candidate = os.path.expanduser("~/code/vggt")
        if os.path.isdir(_candidate) and _candidate not in _sys.path:
            _sys.path.insert(0, _candidate)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print(f"VggtLoader: loading VGGT-1B onto {device} (dtype={dtype})...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    images_1024, original_coords = load_and_preprocess_images_square(
        image_paths, VGGT_LOAD_RESOLUTION
    )
    images_1024 = images_1024.to(device)
    images = F.interpolate(
        images_1024,
        size=(VGGT_INFER_RESOLUTION, VGGT_INFER_RESOLUTION),
        mode="bilinear",
        align_corners=False,
    )

    print(f"VggtLoader: running VGGT on {images.shape[0]} images...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_b = images[None]  # add batch dim
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_b.shape[-2:]
        )
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_b, ps_idx
        )

    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    depth_map = depth_map.squeeze(0).squeeze(-1).float().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).float().cpu().numpy()
    original_coords_np = original_coords.cpu().numpy().astype(np.float32)

    # Free model + autocast tensors before we return.
    del model, aggregated_tokens_list, pose_enc
    if device == "cuda":
        torch.cuda.empty_cache()

    return extrinsic, intrinsic, depth_map, depth_conf, original_coords_np


def _to_original(intrinsic_518, depth_518, depth_conf_518, original_coords_1024,
                 conf_thres):
    """Map a single VGGT prediction back to the original-image resolution.

    Returns intrinsic_orig (3, 3), depth_orig (H, W) np.float32 with low-conf
    pixels zeroed.
    """
    s_load = VGGT_INFER_RESOLUTION / VGGT_LOAD_RESOLUTION
    x1, y1, x2, y2, W, H = original_coords_1024
    x1 *= s_load
    y1 *= s_load
    x2 *= s_load
    y2 *= s_load
    W = int(round(float(W)))
    H = int(round(float(H)))

    # Clip to the valid 518-grid.
    x1i = max(0, int(np.floor(x1)))
    y1i = max(0, int(np.floor(y1)))
    x2i = min(VGGT_INFER_RESOLUTION, int(np.ceil(x2)))
    y2i = min(VGGT_INFER_RESOLUTION, int(np.ceil(y2)))

    fx = float(intrinsic_518[0, 0])
    fy = float(intrinsic_518[1, 1])
    cx = float(intrinsic_518[0, 2])
    cy = float(intrinsic_518[1, 2])
    cx_crop = cx - x1
    cy_crop = cy - y1
    crop_w = max(1.0, x2 - x1)
    crop_h = max(1.0, y2 - y1)
    sx = W / crop_w
    sy = H / crop_h
    intrinsic_orig = np.array(
        [
            [fx * sx, 0.0, cx_crop * sx],
            [0.0, fy * sy, cy_crop * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    depth_crop = depth_518[y1i:y2i, x1i:x2i]
    conf_crop = depth_conf_518[y1i:y2i, x1i:x2i]
    depth_crop = np.where(conf_crop >= conf_thres, depth_crop, 0.0).astype(np.float32)
    depth_orig = cv2.resize(depth_crop, (W, H), interpolation=cv2.INTER_NEAREST)

    return intrinsic_orig, depth_orig


def _gravity_align_extrinsics(extrinsic_cw, up_cam_per_frame):
    """Rotate the VGGT trajectory so world +Z = (averaged) gravity-up.

    Per-view gravity estimates are transformed into VGGT's world frame and
    averaged (then renormalized) before solving for the world rotation.
    Cam0's translation is forced to the origin so the scene stays anchored.

    extrinsic_cw: (N, 3, 4) cam-from-vggt-world (OpenCV).
    up_cam_per_frame: (N, 3) per-view up direction in each cam's frame.

    Returns:
        poses: list of (R_world_cam (3,3), t_world_cam (3,)) in gravity-aligned world.
        up_wV_global: (3,) the averaged up direction in VGGT's world frame
            (useful for diagnostics).
    """
    N = extrinsic_cw.shape[0]
    extr = extrinsic_cw.astype(np.float64)
    ups_cam = np.asarray(up_cam_per_frame, dtype=np.float64)
    assert ups_cam.shape == (N, 3), f"expected ({N},3), got {ups_cam.shape}"

    # Bring each per-view up into the VGGT world frame: up_wV_i = R_wV_ci @ up_cam_i
    # where R_wV_ci is the inverse of the cam-from-world rotation.
    ups_wV = np.empty_like(ups_cam)
    for i in range(N):
        R_ci_wV = extr[i, :3, :3]
        ups_wV[i] = R_ci_wV.T @ ups_cam[i]
        ups_wV[i] /= np.linalg.norm(ups_wV[i]) + 1e-8

    # Robust direction average: simple mean (each vector already unit-norm).
    # All views see roughly the same world-up, so a plain mean is fine; if
    # one estimate flipped, that would show up as a small mean magnitude.
    up_wV = ups_wV.mean(axis=0)
    mean_mag = float(np.linalg.norm(up_wV))
    up_wV /= mean_mag + 1e-8

    # Build R_wG_wV such that R_wG_wV @ up_wV = +Z.
    # Reuse _R_wc_from_up_cam: it builds a rotation taking the given up vector
    # (expressed in the source frame) to +Z in the destination frame.
    R_wG_wV = _R_wc_from_up_cam(up_wV).astype(np.float64)

    # Anchor cam0 at the origin: shift by the cam0 world-position before rotating.
    # T_wV_c0 = inv(T_c0_wV).
    T_c0_wV = np.eye(4); T_c0_wV[:3, :4] = extr[0]
    T_wV_c0 = np.linalg.inv(T_c0_wV)
    cam0_pos_wV = T_wV_c0[:3, 3]

    T_wG_wV = np.eye(4)
    T_wG_wV[:3, :3] = R_wG_wV
    T_wG_wV[:3, 3] = -R_wG_wV @ cam0_pos_wV

    poses = []
    for i in range(N):
        T_ci_wV = np.eye(4); T_ci_wV[:3, :4] = extr[i]
        # T_wG_ci = T_wG_wV @ inv(T_ci_wV)
        T_wG_ci = T_wG_wV @ np.linalg.inv(T_ci_wV)
        R = T_wG_ci[:3, :3].astype(np.float32)
        t = T_wG_ci[:3, 3].astype(np.float32)
        poses.append((R, t))
    return poses, up_wV.astype(np.float32), mean_mag


class VggtLoader(BaseLoader):
    def __init__(
        self,
        seq_dir: str,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 1,
        resize: Optional[Union[int, tuple]] = None,
        conf_thres: float = 5.0,
        gravity_method: str = "geocalib",
        geocalib_device: Optional[str] = None,
        vggt_device: Optional[str] = None,
        force_reload: bool = False,
    ):
        seq_dir = os.path.expanduser(seq_dir)
        if not os.path.isabs(seq_dir) and not os.path.exists(seq_dir):
            from utils.demo_utils import SAMPLE_DATA_PATH

            seq_dir = os.path.join(SAMPLE_DATA_PATH, seq_dir)
        self.seq_dir = seq_dir
        self.seq_name = os.path.basename(seq_dir.rstrip("/"))
        self.camera = "rgb"
        self.device_name = "vggt"
        self.resize = resize
        self.conf_thres = conf_thres
        self.gravity_method = gravity_method

        all_files = _list_images(seq_dir)
        if len(all_files) == 0:
            raise FileNotFoundError(f"No images found in {seq_dir}")
        # start_frame is 1-indexed for parity with other loaders.
        frame_files = all_files[start_frame - 1 :: skip_frames]
        if max_frames is not None:
            frame_files = frame_files[:max_frames]
        if len(frame_files) == 0:
            raise ValueError(
                f"Image selection produced 0 frames (start={start_frame}, "
                f"skip={skip_frames}, max={max_frames})"
            )
        self.frame_files = frame_files
        self.length = len(frame_files)
        self.index = 0

        self.sem_id_to_name = {}
        self.sem_name_to_id = {}

        cache_path = os.path.join(seq_dir, CACHE_FILENAME)
        cache_key = _cache_key(seq_dir, frame_files, conf_thres, gravity_method)
        loaded_from_cache = False
        if os.path.exists(cache_path) and not force_reload:
            try:
                with np.load(cache_path, allow_pickle=False) as npz:
                    if str(npz["cache_key"].item()) == cache_key:
                        extrinsic_518 = npz["extrinsic"]
                        intrinsic_518 = npz["intrinsic"]
                        depth_518 = npz["depth"]
                        depth_conf_518 = npz["depth_conf"]
                        original_coords_1024 = npz["original_coords"]
                        up_cam = npz["up_cam"]
                        loaded_from_cache = True
                        print(f"VggtLoader: loaded cache from {cache_path}")
            except Exception as e:
                print(f"VggtLoader: cache read failed ({e!r}), recomputing")

        if not loaded_from_cache:
            if vggt_device is None:
                vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
            if vggt_device == "cuda":
                cap = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                dtype = torch.float32
            image_paths = [os.path.join(seq_dir, f) for f in frame_files]
            (
                extrinsic_518,
                intrinsic_518,
                depth_518,
                depth_conf_518,
                original_coords_1024,
            ) = _run_vggt(image_paths, vggt_device, dtype)

            up_cam = self._estimate_up_cam_all(
                image_paths,
                intrinsic_518,
                original_coords_1024,
                geocalib_device,
            )

            tmp_path = cache_path + ".tmp.npz"
            np.savez(
                tmp_path,
                extrinsic=extrinsic_518.astype(np.float32),
                intrinsic=intrinsic_518.astype(np.float32),
                depth=depth_518.astype(np.float32),
                depth_conf=depth_conf_518.astype(np.float32),
                original_coords=original_coords_1024.astype(np.float32),
                up_cam=up_cam.astype(np.float32),
                cache_key=np.array(cache_key),
            )
            os.replace(tmp_path, cache_path)
            print(f"VggtLoader: wrote cache to {cache_path}")

        # Map each frame's intrinsic + depth back into native resolution.
        intrinsics_orig = []
        depths_orig = []
        sizes = []
        for i in range(self.length):
            K_orig, D_orig = _to_original(
                intrinsic_518[i],
                depth_518[i],
                depth_conf_518[i],
                original_coords_1024[i],
                conf_thres,
            )
            intrinsics_orig.append(K_orig)
            depths_orig.append(D_orig)
            sizes.append(D_orig.shape)
        self._intrinsics_orig = intrinsics_orig
        self._depths_orig = depths_orig
        self._sizes = sizes

        # Gravity-align the trajectory using per-view averaged gravity.
        self._poses_world_cam, up_wV, up_mean_mag = _gravity_align_extrinsics(
            extrinsic_518, up_cam
        )

        # Sanity check: cam0 sits at origin; agreement of per-view gravity = mean_mag.
        R0, t0 = self._poses_world_cam[0]
        ortho_err = float(np.linalg.norm(R0.T @ R0 - np.eye(3)))
        print(
            f"VggtLoader: {self.seq_name}, {self.length} frames, "
            f"gravity_method={gravity_method}, conf_thres={conf_thres}"
        )
        print(
            f"  cam0 t={t0.tolist()} ||R^T R - I||={ortho_err:.2e}"
        )
        print(
            f"  per-view gravity agreement (1.0=perfect): {up_mean_mag:.4f}, "
            f"avg up in VGGT-world: {up_wV.tolist()}"
        )

        self._init_prefetch()

    def _estimate_up_cam_all(self, image_paths, intrinsic_518_all,
                              original_coords_1024_all, geocalib_device):
        """Recover per-view up-direction (one per frame) using GeoCalib.

        Returns: (N, 3) np.float32 — each row is the gravity-up unit vector in
        the corresponding cam_i frame.
        """
        N = len(image_paths)
        if self.gravity_method == "identity":
            ups = np.zeros((N, 3), dtype=np.float32)
            ups[:, 2] = 1.0
            return ups
        if self.gravity_method != "geocalib":
            raise ValueError(
                f"Unknown gravity_method={self.gravity_method!r}. "
                "Expected 'geocalib' or 'identity'."
            )
        try:
            from geocalib import GeoCalib
        except ImportError as e:
            raise ImportError(
                "VggtLoader needs the `geocalib` package for gravity_method='geocalib'. "
                "Install with `pip install -e ~/code/GeoCalib` or "
                "`pip install 'geocalib @ git+https://github.com/cvg/GeoCalib'`."
            ) from e

        if geocalib_device is None:
            if torch.cuda.is_available():
                geocalib_device = "cuda"
            elif torch.backends.mps.is_available():
                geocalib_device = "mps"
            else:
                geocalib_device = "cpu"

        # Load GeoCalib once, reuse for all frames.
        gc = GeoCalib(weights="pinhole").to(geocalib_device).eval()
        ups = np.empty((N, 3), dtype=np.float32)
        for i, image_path in enumerate(image_paths):
            K_orig, _ = _to_original(
                intrinsic_518_all[i],
                np.zeros((VGGT_INFER_RESOLUTION, VGGT_INFER_RESOLUTION), dtype=np.float32),
                np.zeros((VGGT_INFER_RESOLUTION, VGGT_INFER_RESOLUTION), dtype=np.float32),
                original_coords_1024_all[i],
                conf_thres=-np.inf,
            )
            focal = float((K_orig[0, 0] + K_orig[1, 1]) / 2.0)

            img_bgr = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_t = (
                torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            ).to(geocalib_device)
            focal_t = torch.tensor(focal).to(geocalib_device)
            result = gc.calibrate(
                img_t, camera_model="pinhole", priors={"focal": focal_t}
            )
            up = result["gravity"].vec3d.squeeze(0).detach().cpu().numpy()
            ups[i] = up.astype(np.float32) / (np.linalg.norm(up) + 1e-8)
        del gc
        return ups

    def load(self, idx):
        path = os.path.join(self.seq_dir, self.frame_files[idx])
        img_bgr = cv2.imread(path)
        img_rgb_native = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        HH, WW = img_rgb_native.shape[:2]

        K_orig = self._intrinsics_orig[idx]
        depth_orig = self._depths_orig[idx]
        fx_o = float(K_orig[0, 0])
        fy_o = float(K_orig[1, 1])
        cx_o = float(K_orig[0, 2])
        cy_o = float(K_orig[1, 2])

        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)):
                resizeH, resizeW = int(self.resize[0]), int(self.resize[1])
            else:
                resizeH = resizeW = int(self.resize)
            # Letterbox to (resizeH, resizeW): uniform-scale so the long side
            # fills, pad the short side with black. Avoids the anisotropic
            # squash that was distorting OWL boxes on non-square sources.
            s = min(resizeW / WW, resizeH / HH)
            newW = int(round(WW * s))
            newH = int(round(HH * s))
            pad_left = (resizeW - newW) // 2
            pad_top = (resizeH - newH) // 2
            pad_right = resizeW - newW - pad_left
            pad_bottom = resizeH - newH - pad_top

            img_scaled = cv2.resize(
                img_rgb_native, (newW, newH), interpolation=cv2.INTER_LINEAR
            )
            img_rgb = cv2.copyMakeBorder(
                img_scaled, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            depth_scaled = cv2.resize(
                depth_orig, (newW, newH), interpolation=cv2.INTER_NEAREST
            )
            depth = cv2.copyMakeBorder(
                depth_scaled, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0,
            )
            fx = fx_o * s
            fy = fy_o * s
            cx = cx_o * s + pad_left
            cy = cy_o * s + pad_top
        else:
            resizeH, resizeW = HH, WW
            img_rgb = img_rgb_native
            depth = depth_orig
            fx = fx_o
            fy = fy_o
            cx = cx_o
            cy = cy_o

        cam = self.pinhole_from_K(
            resizeW, resizeH, fx, fy, cx, cy, valid_radius=(resizeW, resizeH)
        )

        R_wc, t_wc = self._poses_world_cam[idx]
        T_wr_data = torch.tensor([*R_wc.flatten(), *t_wc], dtype=torch.float32)

        sdp_w = self.sdp_from_depth(depth, fx, fy, cx, cy, R_wc, t_wc)

        return {
            "img0": self.img_to_tensor(img_rgb),
            "cam0": cam.float(),
            "T_world_rig0": PoseTW(T_wr_data),
            "sdp_w": sdp_w,
            "obbs": ObbTW(torch.zeros(0, 165)),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "gt_labels": [],
            "time_ns0": idx * 33_333_333,
            "rotated0": torch.tensor(False).reshape(1),
        }
