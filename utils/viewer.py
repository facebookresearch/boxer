#!/usr/bin/env python3
# pyre-strict

import argparse
import colorsys
import hashlib
import os
import re
import sys
import tempfile
import time as time_module
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import torch

import imgui
import matplotlib.cm as cm
import moderngl
import moderngl_window as mglw
import numpy as np
from pyrr import Matrix44
from loaders.aria_loader import AriaLoader
from loaders.ca_loader import CALoader
from loaders.scannet_loader import ScanNetLoader
from utils.demo_utils import SAMPLE_DATA_PATH
from utils.tw.camera import CameraTW
from utils.file_io import (
    dump_obbs_adt,
    load_bb2d_csv,
    load_obbs_adt,
    probe_gravity_direction,
    read_obb_csv,
)
from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.orbit_viewer import OrbitViewer, scale_factor
from utils.tw.pose import PoseTW
from utils.taxonomy import BOXY_SEM2NAME, SSI_COLORS_ALT, TEXT2COLORS
from utils.tw.tensor_utils import find_nearest2, tensor2string, unpad_string
from utils.track_3d_boxes import BoundingBox3DTracker
from utils.video import make_mp4
from utils.demo_utils import DEFAULT_SEQ, EVAL_PATH, SAMPLE_DATA_PATH
from loaders.omni_loader import OMNI3D_DATASETS, OmniLoader


# ---------------------------------------------------------------------------
# Shared viewer utilities (formerly view_boxer.py)
# ---------------------------------------------------------------------------


def build_seq_ctx(input_path, dataset_type):
    """Build viewer context from input path (creates loader for traj/calib/RGB)."""
    if dataset_type == "aria":
        loader = AriaLoader(
            input_path,
            camera="rgb",
            with_traj=True,
            with_sdp=True,
            with_img=True,
            with_obb=False,
            restrict_range=True,
            max_n=1_000_000,
            skip_n=1,
            start_n=0,
        )
        rgb_stream_id = loader.stream_id[0]
        rgb_num_frames = loader.provider.get_num_data(rgb_stream_id)
        rgb_timestamps = (
            np.array(loader.pose_ts, dtype=np.int64)
            if getattr(loader, "pose_ts", None) is not None
            and len(loader.pose_ts) > 0
            else np.array([0], dtype=np.int64)
        )
        return {
            "source": "aria",
            "loader": loader,
            "rgb_num_frames": rgb_num_frames,
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,
            "is_nebula": bool(loader.is_nebula),
            "traj": loader.traj,
            "pose_ts": loader.pose_ts,
            "calibs": loader.calibs[0],
            "calib_ts": loader.calib_ts,
            "time_to_uids_slaml": getattr(loader, "time_to_uids_slaml", None),
            "time_to_uids_slamr": getattr(loader, "time_to_uids_slamr", None),
            "uid_to_p3": getattr(loader, "uid_to_p3", None),
        }
    elif dataset_type == "ca1m":
        loader = CALoader(
            input_path,
            start_frame=0,
            skip_frames=1,
            max_frames=1_000_000,
        )
        loader.load_all()
        rgb_timestamps = np.array(loader.timestamp_ns)
        n = len(rgb_timestamps)
        traj = [
            (loader.Ts_wc[i] @ loader.cams[i].T_camera_rig).float() for i in range(n)
        ]
        return {
            "source": "ca1m",
            "rgb_num_frames": n,
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": loader.rgb_images,
            "is_nebula": True,
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": loader.cams,
            "calib_ts": rgb_timestamps,
            "loader": loader,
            "time_to_uids_slaml": None,
            "time_to_uids_slamr": None,
            "uid_to_p3": None,
        }
    elif dataset_type == "scannet":
        loader = ScanNetLoader(
            scene_dir=input_path,
            annotation_path=os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"),
            skip_frames=1,
            max_frames=None,
        )
        frame_ids = list(loader.frame_ids)
        first_fid = frame_ids[0]
        color_path = os.path.join(
            loader.scene_dir, "frames", "color", f"{first_fid}.png"
        )
        if not os.path.exists(color_path):
            color_path = os.path.join(
                loader.scene_dir, "frames", "color", f"{first_fid}.jpg"
            )
        first_bgr = cv2.imread(color_path)
        H, W = first_bgr.shape[:2]
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_data = torch.tensor(
            [W, H, loader.fx, loader.fy, loader.cx, loader.cy, -1, 1e-3, W, H],
            dtype=torch.float32,
        )
        cam_template = CameraTW(torch.cat([cam_data, T_cam_rig])).float()

        rgb_timestamps = np.array([int(fid) for fid in frame_ids], dtype=np.int64)
        traj = []
        calibs = []
        for fid in frame_ids:
            pose_path = os.path.join(loader.scene_dir, "frames", "pose", f"{fid}.txt")
            T_world_cam = np.loadtxt(pose_path).astype(np.float32)
            T_world_cam[:3, 3] -= loader.world_offset.astype(np.float32)
            R_flat = T_world_cam[:3, :3].reshape(-1)
            t_vec = T_world_cam[:3, 3]
            T_wr_data = torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
            traj.append(PoseTW(T_wr_data).float())
            calibs.append(cam_template.clone())
        # Accumulate global SDP from depth maps (cached)
        import pickle

        cache_path = os.path.join(loader.scene_dir, "cache_sdp_global_v2.pkl")
        uid_to_p3 = None
        sdp_global = None
        if os.path.exists(cache_path):
            _startup_log("Loading cached ScanNet SDP...")
            with open(cache_path, "rb") as f:
                sdp_data = pickle.load(f)
            uid_to_p3 = sdp_data["uid_to_p3"]
            sdp_global = sdp_data["sdp_global"]
            _startup_log(f"Loaded {len(sdp_global)} cached SDP points")
        else:
            _startup_log("Building ScanNet global SDP from depth maps...")
            # Load depth intrinsics (depth resolution differs from color)
            depth_K_path = os.path.join(
                loader.scene_dir, "frames", "intrinsic", "intrinsic_depth.txt"
            )
            depth_K = np.loadtxt(depth_K_path)
            depth_fx, depth_fy = depth_K[0, 0], depth_K[1, 1]
            depth_cx, depth_cy = depth_K[0, 2], depth_K[1, 2]

            all_pts = []
            frame_step = max(1, len(frame_ids) // 50)
            for i in range(0, len(frame_ids), frame_step):
                fid = frame_ids[i]
                depth_path = os.path.join(
                    loader.scene_dir, "frames", "depth", f"{fid}.png"
                )
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    continue
                depth_np = depth_raw.astype(np.float32) / 1000.0
                pose_path = os.path.join(
                    loader.scene_dir, "frames", "pose", f"{fid}.txt"
                )
                T_wc = np.loadtxt(pose_path).astype(np.float64)
                T_wc[:3, 3] -= loader.world_offset
                T_wc = T_wc.astype(np.float32)
                pts = ScanNetLoader.sdp_from_depth(
                    depth_np,
                    depth_fx,
                    depth_fy,
                    depth_cx,
                    depth_cy,
                    T_wc[:3, :3],
                    T_wc[:3, 3],
                    num_samples=5000,
                )
                valid = ~torch.isnan(pts[:, 0])
                if valid.any():
                    all_pts.append(pts[valid].numpy())
            if all_pts:
                sdp_global = np.concatenate(all_pts, axis=0).astype(np.float32)
                if len(sdp_global) > 200_000:
                    idx = np.random.choice(len(sdp_global), 200_000, replace=False)
                    sdp_global = sdp_global[idx]
                uid_to_p3 = {i: sdp_global[i] for i in range(len(sdp_global))}
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(
                            {"uid_to_p3": uid_to_p3, "sdp_global": sdp_global}, f
                        )
                except OSError:
                    pass  # skip caching if directory isn't writable
                _startup_log(
                    f"Built and cached {len(sdp_global)} SDP points from "
                    f"{len(all_pts)} frames"
                )

        return {
            "source": "scannet",
            "loader": loader,
            "scannet_scene_dir": loader.scene_dir,
            "scannet_frame_ids": list(frame_ids),
            "rgb_num_frames": len(frame_ids),
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,
            "is_nebula": True,
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": calibs,
            "calib_ts": rgb_timestamps,
            "time_to_uids_slaml": None,
            "time_to_uids_slamr": None,
            "uid_to_p3": uid_to_p3,
            "sdp_global": sdp_global,
        }
    elif dataset_type == "omni3d":
        # Omni3D datasets (SUNRGBD, etc.) — single independent images
        from loaders.omni_loader import load_sunrgbd_extrinsics

        loader = OmniLoader(
            dataset_name=input_path,
            split="val",
            shuffle=True,
            seed=42,
            max_images=500,
        )
        # Build poses and calibs from JSON metadata (no image loading)
        traj = []
        calibs = []
        rgb_timestamps = []
        for img_info in loader.images:
            K = img_info["K"]
            W, H = img_info["width"], img_info["height"]
            fx, fy = K[0][0], K[1][1]
            cx, cy = K[0][2], K[1][2]
            cam = loader.pinhole_from_K(W, H, fx, fy, cx, cy)
            calibs.append(cam.float())
            rgb_timestamps.append(int(img_info["id"]))

            # Build T_world_rig (same logic as OmniLoader.load)
            if input_path == "SUNRGBD":
                R_ext = load_sunrgbd_extrinsics(
                    loader.data_root, img_info["file_path"]
                )
                if R_ext is not None:
                    R_yz = np.array(
                        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
                    )
                    R_flat = (R_yz @ R_ext).flatten()
                    T_wr_data = torch.tensor(
                        [*R_flat, 0, 0, 0], dtype=torch.float32
                    )
                else:
                    T_wr_data = torch.tensor(
                        [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                        dtype=torch.float32,
                    )
            else:
                T_wr_data = torch.tensor(
                    [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], dtype=torch.float32
                )
            traj.append(PoseTW(T_wr_data).float())

        rgb_timestamps = np.array(rgb_timestamps, dtype=np.int64)
        print(f"Loaded {len(traj)} {input_path} images (metadata)")

        return {
            "source": "omni3d",
            "loader": loader,
            "rgb_num_frames": len(traj),
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,  # loaded on demand via loader
            "is_nebula": True,  # no Aria rotation
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": calibs,
            "calib_ts": rgb_timestamps,
            "time_to_uids_slaml": None,
            "time_to_uids_slamr": None,
            "uid_to_p3": None,
            "sdp_global": None,
        }
    else:
        return None


def subsample_timed_obbs(timed_obbs, skip_n=1, start_n=0, max_n=0):
    """Subsample/slice timed_obbs dict by skip_n, start_n, max_n."""
    sorted_ts = sorted(timed_obbs.keys())
    if start_n > 0:
        sorted_ts = sorted_ts[start_n:]
    if max_n > 0 and len(sorted_ts) > max_n:
        sorted_ts = sorted_ts[:max_n]
    if skip_n > 1:
        sorted_ts = sorted_ts[::skip_n]
    return {ts: timed_obbs[ts] for ts in sorted_ts}


def resolve_input(input_str):
    """Resolve input string to (input_path, dataset_type, seq_name)."""
    if bool(re.search(r"scene\d{4}_\d{2}", input_str)) or "/scannet/" in input_str:
        return input_str, "scannet", os.path.basename(input_str.rstrip("/"))
    elif input_str in OMNI3D_DATASETS:
        return input_str, "omni3d", input_str
    elif input_str.startswith("ca1m"):
        return input_str, "ca1m", input_str
    else:
        input_path = input_str
        if not os.path.isabs(input_path) and not os.path.exists(input_path):
            sample = os.path.join(SAMPLE_DATA_PATH, input_path)
            legacy = os.path.expanduser(os.path.join("~/boxy_data", input_path))
            if os.path.exists(sample):
                input_path = sample
            elif os.path.exists(legacy):
                input_path = legacy
        seq_name = input_path.rstrip("/").split("/")[-1]
        return input_path, "aria", seq_name


def load_view_file(log_dir, load_view_arg):
    """Resolve and load camera view file. Returns (view_path, load_view_data)."""
    view_path = os.path.join(log_dir, "camera_view.pt")
    if load_view_arg is None:
        return view_path, None
    target = view_path if load_view_arg == "DEFAULT" else load_view_arg
    if os.path.exists(target):
        data = torch.load(target, weights_only=False)
        print(f"==> Loaded camera view from {target}")
        return view_path, data
    return view_path, None


def resolve_bb2d_csv(log_dir, bb2d_csv_arg, write_name):
    """Resolve 2D BB CSV path. Raises IOError if not found."""
    path = os.path.join(log_dir, bb2d_csv_arg) if bb2d_csv_arg else os.path.join(log_dir, "owl_2dbbs.csv")
    if not os.path.exists(path):
        raise IOError(f"2D BB CSV not found: {path}")
    return path


def add_common_args(parser):
    """Add arguments shared between fusion and tracker viewers."""
    # fmt: off
    parser.add_argument("--input", type=str, default=DEFAULT_SEQ, help="path to the sequence folder")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Where CSVs live (default: ~/viz_boxer)")
    parser.add_argument("--write_name", default="boxer", type=str, help="CSV prefix (default: boxer)")
    parser.add_argument("--skip_n", type=int, default=1, help="subsample loaded OBBs")
    parser.add_argument("--start_n", type=int, default=0, help="start from n-th OBB frame")
    parser.add_argument("--max_n", type=int, default=0, help="max OBB frames to load")
    parser.add_argument("--load_view", type=str, nargs="?", const="DEFAULT", default=None)
    parser.add_argument("--window_w", type=int, default=0, help="Initial window width (0 = default)")
    parser.add_argument("--window_h", type=int, default=0, help="Initial window height (0 = default)")
    parser.add_argument("--init_color_mode", type=str, default=None, help="Initial 3DBB color mode")
    parser.add_argument("--init_rgb_text_scale", type=float, default=None, help="Initial RGB label text scale")
    parser.add_argument("--init_image_panel_width", type=float, default=None, help="Initial image panel width fraction")
    parser.add_argument("--scannet_scene", type=str, default=None, help="Path to ScanNet scene directory")
    parser.add_argument("--scannet_annotation_path", type=str, default=os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"))
    # fmt: on


def load_common(args):
    """Shared loading logic. Returns (input_path, dataset_type, seq_name, log_dir, view_path, load_view_data)."""
    input_path, dataset_type, seq_name = resolve_input(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    view_path, load_view_data = load_view_file(log_dir, args.load_view)
    return input_path, dataset_type, seq_name, log_dir, view_path, load_view_data


def launch_viewer(ViewerClass):
    """Run moderngl viewer, protecting sys.argv."""
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    try:
        mglw.run_window_config(ViewerClass)
    finally:
        sys.argv = saved_argv


# Color mode constants
COLOR_MODE_PCA = 0
COLOR_MODE_PROBABILITY = 1
COLOR_MODE_RANDOM = 2


def _normalize_color_mode_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _fuse_color_mode_from_name(name: str) -> Optional[int]:
    key = _normalize_color_mode_name(name)
    mapping = {
        "pca": COLOR_MODE_PCA,
        "prob": COLOR_MODE_PROBABILITY,
        "probability": COLOR_MODE_PROBABILITY,
        "random": COLOR_MODE_RANDOM,
    }
    return mapping.get(key)


def _track_color_mode_from_name(name: str) -> Optional[int]:
    key = _normalize_color_mode_name(name)
    mapping = {
        "confidence": 0,
        "health": 0,
        "boxy": 1,
        "boxy_color": 1,
        "boxy_alt": 2,
        "boxyalt": 2,
        "boxy_alt_color": 2,
        "text_pca": 3,
        "pca": 3,
        "random": 4,
    }
    return mapping.get(key)


def _startup_log(msg: str) -> None:
    print(f"[STARTUP] {msg}")


@dataclass
class FrameObbSets:
    raw: Optional[ObbTW] = None
    tracked_all: Optional[ObbTW] = None
    tracked_visible: Optional[ObbTW] = None


def _load_sequence_context_auto(
    seq_name: str,
    *,
    scannet_scene: str | None = None,
    scannet_annotation_path: str = os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"),
    with_sdp: bool = False,
    start_frame: int = 0,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Load sequence context via AriaLoader (default), CALoader, or ScanNetLoader.

    CA1M always uses pinhole mode in this viewer.
    """
    t0 = time_module.perf_counter()
    _startup_log(
        f"Loading sequence context for '{seq_name}' "
        f"(scannet={scannet_scene is not None}, with_sdp={with_sdp}, "
        f"start_frame={start_frame}, max_frames={max_frames})"
    )
    if scannet_scene:
        t_sc0 = time_module.perf_counter()
        # Load full ScanNet timeline — do NOT slice by start_frame/max_frames
        # here because the OBB CSV may have gaps (frames with zero detections
        # are omitted), so index-based slicing diverges from the CSV's timestamp
        # set.  Instead, keep all valid frames and let find_nearest2 map each
        # OBB timestamp to the correct RGB/pose frame.
        loader = ScanNetLoader(
            scene_dir=scannet_scene,
            annotation_path=scannet_annotation_path,
            skip_frames=1,
            max_frames=None,
        )
        frame_ids = list(loader.frame_ids)
        if not frame_ids:
            raise RuntimeError(f"ScanNetLoader returned 0 frames for scene {seq_name}")
        _startup_log(
            f"ScanNetLoader init complete in {(time_module.perf_counter() - t_sc0):.2f}s ({len(frame_ids)} frames)"
        )

        t_cam0 = time_module.perf_counter()
        # Build a single pinhole camera from the first frame dimensions.
        first_fid = frame_ids[0]
        color_path = os.path.join(
            loader.scene_dir, "frames", "color", f"{first_fid}.png"
        )
        if not os.path.exists(color_path):
            color_path = os.path.join(
                loader.scene_dir, "frames", "color", f"{first_fid}.jpg"
            )
        first_bgr = cv2.imread(color_path)
        if first_bgr is None:
            raise RuntimeError(f"Failed to read ScanNet color image: {color_path}")
        H, W = first_bgr.shape[:2]
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_data = torch.tensor(
            [W, H, loader.fx, loader.fy, loader.cx, loader.cy, -1, 1e-3, W, H],
            dtype=torch.float32,
        )
        cam_template = CameraTW(torch.cat([cam_data, T_cam_rig])).float()

        rgb_timestamps = np.array([int(fid) for fid in frame_ids], dtype=np.int64)
        traj: list[PoseTW] = []
        calibs: list[CameraTW] = []
        for fid in frame_ids:
            pose_path = os.path.join(loader.scene_dir, "frames", "pose", f"{fid}.txt")
            T_world_cam = np.loadtxt(pose_path).astype(np.float32)
            T_world_cam[:3, 3] -= loader.world_offset.astype(np.float32)
            R_flat = T_world_cam[:3, :3].reshape(-1)
            t_vec = T_world_cam[:3, 3]
            T_wr_data = torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
            traj.append(PoseTW(T_wr_data).float())
            calibs.append(cam_template.clone())
        _startup_log(
            f"ScanNet traj/calib precompute complete in {(time_module.perf_counter() - t_cam0):.2f}s"
        )
        data = {
            "source": "scannet",
            "loader": loader,
            "scannet_scene_dir": loader.scene_dir,
            "scannet_frame_ids": list(frame_ids),
            "rgb_num_frames": len(frame_ids),
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,  # lazy-loaded on demand
            "is_nebula": True,  # no Aria Gen1 rotation path for ScanNet
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": calibs,
            "calib_ts": rgb_timestamps,
        }
        if with_sdp:
            data["time_to_uids_slaml"] = None
            data["time_to_uids_slamr"] = None
            data["uid_to_p3"] = None
        _startup_log(
            f"Context ready (ScanNet) in {(time_module.perf_counter() - t0):.2f}s"
        )
        return data

    if seq_name.startswith("ca1m"):
        t_ca0 = time_module.perf_counter()
        ca_max_frames = max_frames if max_frames > 0 else 1_000_000
        ca = CALoader(
            seq_name,
            start_frame=max(0, int(start_frame)),
            skip_frames=1,
            max_frames=ca_max_frames,
            num_samples=1000,
            remove_structure=False,
            remove_large=False,
        )
        ca.load_all()
        rgb_timestamps = np.array(ca.timestamp_ns)
        n = len(rgb_timestamps)
        # Use the same world-rig pose convention as CALoader.__next__:
        #   datum["T_world_rig0"] = T_wc @ cam.T_camera_rig
        traj = [(ca.Ts_wc[i] @ ca.cams[i].T_camera_rig).float() for i in range(n)]
        data: dict[str, Any] = {
            "source": "ca1m",
            "rgb_num_frames": n,
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": ca.rgb_images,
            "is_nebula": True,
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": ca.cams,
            "calib_ts": rgb_timestamps,
            "loader": ca,
        }
        if with_sdp:
            data["time_to_uids_slaml"] = None
            data["time_to_uids_slamr"] = None
            data["uid_to_p3"] = None
        _startup_log(
            f"Context ready (CA) in {(time_module.perf_counter() - t_ca0):.2f}s"
        )
        return data

    # Default path: AriaLoader for boxy_data / VRS sequences.
    t_aria0 = time_module.perf_counter()
    root = os.path.join(os.path.expanduser("~"), "boxy_data", seq_name)
    # IMPORTANT: don't cap by detection-frame count; CSV timestamps can be
    # sparse over long recordings, and truncating the RGB timeline can misalign
    # image/pose lookup.
    loader_max_frames = 1_000_000
    loader = AriaLoader(
        remote_root=root,
        camera="rgb",
        with_img=True,
        with_traj=True,
        with_sdp=with_sdp,
        with_obb=False,
        restrict_range=True,
        max_n=loader_max_frames,
        skip_n=1,
        start_n=max(0, int(start_frame)),
    )
    rgb_stream_id = loader.stream_id[0]
    rgb_num_frames = loader.provider.get_num_data(rgb_stream_id)
    if rgb_num_frames <= 0:
        raise RuntimeError(f"AriaLoader returned 0 frames for {seq_name}")
    # Keep startup fast: don't scan all frame timestamps here.
    # For Aria RGB fetch we resolve frame index on-demand via _find_frame_by_timestamp.
    rgb_timestamps = (
        np.array(loader.pose_ts, dtype=np.int64)
        if getattr(loader, "pose_ts", None) is not None and len(loader.pose_ts) > 0
        else np.array([0], dtype=np.int64)
    )
    data: dict[str, Any] = {
        "source": "aria",
        "loader": loader,
        "rgb_num_frames": rgb_num_frames,
        "rgb_timestamps": rgb_timestamps,
        "rgb_images": None,  # lazy-loaded from aria img loader
        "is_nebula": bool(loader.is_nebula),
        "traj": loader.traj,
        "pose_ts": loader.pose_ts,
        "calibs": loader.calibs[0],  # camera='rgb'
        "calib_ts": loader.calib_ts,
    }
    if with_sdp:
        data["time_to_uids_slaml"] = getattr(loader, "time_to_uids_slaml", None)
        data["time_to_uids_slamr"] = getattr(loader, "time_to_uids_slamr", None)
        data["uid_to_p3"] = getattr(loader, "uid_to_p3", None)
    _startup_log(f"Context ready (Aria) in {(time_module.perf_counter() - t_aria0):.2f}s")
    return data


def _extract_ca_sdp_positions(ca_loader: Any) -> Optional[np.ndarray]:
    """Build a global semidense point cloud from CALoader.sdp_ws."""
    sdp_ws = getattr(ca_loader, "sdp_ws", None)
    if sdp_ws is None:
        return None
    pts_all: list[torch.Tensor] = []
    for sdp in sdp_ws:
        if sdp is None:
            continue
        pts = sdp.float().reshape(-1, 3).cpu()
        if len(pts) == 0:
            continue
        valid = torch.isfinite(pts).all(dim=1)
        nonzero = pts.abs().sum(dim=1) > 1e-6
        pts = pts[valid & nonzero]
        if len(pts) > 0:
            pts_all.append(pts)
    if not pts_all:
        return None
    # Dedup with light quantization to keep memory bounded and stable.
    pts_cat = torch.cat(pts_all, dim=0)
    pts_quant = torch.round(pts_cat * 1000.0) / 1000.0
    pts_unique = torch.unique(pts_quant, dim=0)
    return pts_unique.numpy().astype(np.float32)


def _infer_fps_from_timestamps_ns(
    timestamps: np.ndarray,
    fallback: float = 10.0,
    source: str | None = None,
) -> float:
    """Infer FPS from timestamps.

    For true nanosecond timelines, uses 1e9 / median_delta.
    For index-like timelines (e.g. ScanNet frame ids, delta~1), falls back to
    source-specific defaults to avoid absurd FPS values.
    """
    if timestamps is None or len(timestamps) < 2:
        return float(fallback)
    deltas = np.diff(np.asarray(timestamps, dtype=np.int64))
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return float(fallback)
    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        return float(fallback)

    # Non-nanosecond timeline guard (e.g. frame ids: 0,1,2,...)
    if median_delta < 1_000.0:
        if source == "scannet":
            return 30.0
        if source == "ca1m":
            return 10.0
        return float(fallback)

    fps = float(1e9 / median_delta)
    if not np.isfinite(fps) or fps <= 0.0:
        return float(fallback)
    # Guard against corrupted tiny deltas producing unrealistic FPS.
    if fps > 240.0:
        if source == "scannet":
            return 30.0
        if source == "ca1m":
            return 10.0
        return float(fallback)
    return fps


class OBBViewer(OrbitViewer):
    """Viewer for rendering 3D oriented bounding boxes."""

    title = "OBB Fusion Viewer"
    window_size = (2250 * scale_factor, 1100 * scale_factor)

    def _empty_obbs_like(self, obbs: Optional[ObbTW] = None) -> ObbTW:
        if obbs is not None and hasattr(obbs, "_data"):
            d = int(obbs._data.shape[-1])
        else:
            d = 165
        return ObbTW(torch.zeros(0, d))

    def _subset_obbs(self, obbs: ObbTW, mask: torch.Tensor) -> ObbTW:
        if len(obbs) == 0 or mask.numel() == 0 or not bool(mask.any()):
            return self._empty_obbs_like(obbs)
        return ObbTW(obbs._data[mask])

    def _set_frame_obb_sets(
        self,
        *,
        raw: Optional[ObbTW] = None,
        tracked_all: Optional[ObbTW] = None,
        tracked_visible: Optional[ObbTW] = None,
    ) -> None:
        self.frame_obb_sets = FrameObbSets(
            raw=raw,
            tracked_all=tracked_all,
            tracked_visible=tracked_visible,
        )

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        """Return (width, height) of 3D viewport (right of controls panel)."""
        w, h = self.wnd.size
        return w, h

    def get_camera_matrices(self):
        """Use active 3D viewport aspect so camera matches split-screen view."""
        vw, vh = self._get_3d_viewport_size()
        aspect_ratio = vw / vh
        projection = Matrix44.perspective_projection(45.0, aspect_ratio, 0.1, 100.0)

        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        camera_x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        camera_y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        camera_z = self.camera_distance * np.sin(elevation_rad)
        camera_pos = self.camera_target + np.array([camera_x, camera_y, camera_z])

        view = Matrix44.look_at(
            tuple(camera_pos),
            tuple(self.camera_target),
            (0.0, 0.0, 1.0),
        )
        mvp = projection * view * Matrix44.identity()
        return projection, view, mvp

    def __init__(
        self,
        all_obbs: ObbTW,
        root_path: str,
        timed_obbs: dict[int, ObbTW] | None = None,
        init_color_mode: Optional[str] = None,
        init_image_panel_width: Optional[float] = None,
        skip_precompute: bool = False,
        load_view_data: dict | None = None,
        view_save_path: str = "",
        seq_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize viewer with a single ObbTW tensor containing all detections.

        Args:
            all_obbs: ObbTW tensor containing all detections
            root_path: Root path for the sequence
            skip_precompute: Skip semantic embedding precomputation and initial
                geometry build. Used by TrackerViewer which builds its own geometry
                per frame and doesn't need embeddings at startup.
            load_view_data: Optional dict with saved camera state to restore
            view_save_path: Path to save camera view .pt files
        """
        t_init0 = time_module.perf_counter()
        _startup_log(f"OBBViewer init start (seq={seq_name})")
        self.all_obbs = all_obbs  # Single ObbTW tensor with all detections
        self.total_detections = len(all_obbs)  # Cache total count
        self.alpha = 0.2  # Alpha transparency for OBBs
        self.show_debug = False  # Debug info visibility
        # Color mode: default to theme-aware Random
        self.color_mode = COLOR_MODE_RANDOM
        if init_color_mode is not None:
            parsed_mode = _fuse_color_mode_from_name(init_color_mode)
            if parsed_mode is not None:
                self.color_mode = parsed_mode
            else:
                print(
                    f"Warning: unknown --init_color_mode='{init_color_mode}' "
                    "(valid: pca, prob, random)"
                )
        self.prob_threshold = 0.55  # Probability threshold for filtering
        self.root_path = root_path  # Root path for saving results
        self.seq_name = seq_name  # Sequence name for boxy_data lookup
        if timed_obbs is not None:
            self.timed_obbs = timed_obbs
        elif not hasattr(self, "timed_obbs"):
            self.timed_obbs = {}
        self.sorted_timestamps = sorted(self.timed_obbs.keys())
        self.total_frames = len(self.sorted_timestamps)
        self.current_frame_idx = 0
        self.is_playing = False
        self.playback_fps = 10.0
        self._last_step_time = 0.0
        self._last_render_mvp: Optional[np.ndarray] = None

        # Fusion state
        self.tracked_all_instances = []  # List of FusedInstance
        self.show_raw_set = False
        self.show_tracked_all_set = True
        self.show_tracked_visible_set = True
        # Unified visual theme: 0=Light, 1=Dark
        self.visual_theme_mode = 0
        self.overlay_text_rgb: tuple[float, float, float] = (0.25, 0.25, 0.25)
        self.overlay_text_bg_rgba: tuple[float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            0.6,
        )
        self.show_text_labels = False  # Toggle text labels on fused boxes
        self.show_dimensions = False  # Toggle dimensions display on text labels
        self.frame_obb_sets = FrameObbSets()
        self.tracked_all_vbo = None  # Separate VBO for tracked-all boxes
        self.tracked_all_vao = None  # Separate VAO for tracked-all boxes
        self.tracked_all_vertex_count = 0
        self.tracked_all_text_labels = []  # Text labels for each fused instance
        self.tracked_all_label_positions = []  # 3D positions for text labels
        self.tracked_all_label_colors = []  # Per-instance RGB colors for label backgrounds

        # Fusion parameters (adjustable in UI)
        self.fusion_iou_threshold = 0.3  # IOU threshold for fusion
        self.fusion_min_detections = 4  # Minimum detections to create instance
        self.fusion_confidence_weighting = "robust"  # Confidence weighting mode
        self.fusion_samp_per_dim = (
            8  # Number of samples per dimension for IoU calculation
        )
        self.fusion_semantic_threshold = (
            0.7  # Minimum semantic similarity to allow merging (hard cutoff)
        )
        self.fusion_enable_nms = True  # Enable NMS on fused boxes
        self.fusion_nms_iou_threshold = 0.6  # IoU threshold for NMS (>0.7 = redundant)

        # Line rendering parameters
        self.raw_line_width = 2  # Line width for raw detections
        self.tracked_all_line_width = 5  # Line width for fused boxes
        self.show_axis_lines = False  # Toggle RGB axis lines on 3D boxes

        # Axis lines rendering buffers (initialized early, built in init_scene)
        self.axis_instance_vbo = None
        self.axis_instance_vao = None
        self.axis_instance_count = 0

        # UI panel width for text label culling
        self.ui_panel_width = 450  # Labels with x < this are hidden

        # Track last fusion parameters to detect when they're out of date
        self._last_fusion_iou_threshold: Optional[float] = None
        self._last_fusion_min_detections: Optional[int] = None
        self._last_fusion_confidence_weighting: Optional[str] = None
        self._last_fusion_samp_per_dim: Optional[int] = None
        self._last_fusion_semantic_threshold: Optional[float] = None
        self._last_fusion_enable_nms: Optional[bool] = None
        self._last_fusion_nms_iou_threshold: Optional[float] = None
        self._last_fusion_prob_threshold: Optional[float] = (
            None  # Track prob threshold too
        )

        # Cache for filtered detections (to avoid recomputing every frame)
        self._cached_filtered_obbs: Optional[ObbTW] = None
        self._cached_prob_threshold: Optional[float] = None
        self._cached_filtered_indices: Optional[torch.Tensor] = None  # Track indices

        # Camera view save/load
        self.view_save_path = view_save_path
        self._load_view_data = load_view_data  # Applied after super().__init__

        # Semantic embeddings cache (computed once for ALL detections during init)
        self._semantic_embeddings: Optional[torch.Tensor] = None
        # No longer need _embedding_model since we use the standalone function

        # PCA model cache (fitted once on all detections, reused for consistency)
        self._pca_model: Optional[object] = None
        self._all_embeddings: Optional[torch.Tensor] = None

        # Compute semantic embeddings for ALL detections upfront BEFORE initializing scene
        # (needed because init_scene() -> _build_geometry_cache() uses embeddings)
        self._skip_precompute = skip_precompute
        if not skip_precompute:
            from utils.fuse_3d_boxes import precompute_semantic_embeddings

            self._semantic_embeddings = precompute_semantic_embeddings(self.all_obbs)
        else:
            self._semantic_embeddings = None

        # Initialize camera to focus on scene center
        super().__init__(**kwargs)
        _startup_log(
            f"OrbitViewer/OpenGL init done in {(time_module.perf_counter() - t_init0):.2f}s"
        )

        # Override defaults for white-background-friendly viewing
        self.bg_color_index = 0  # White background
        self._apply_visual_theme()

        if self.total_frames > 0:
            self._step_to_frame(self.current_frame_idx)

        # Apply loaded camera view or auto-focus on scene
        if self._load_view_data is not None:
            self._apply_camera_view(self._load_view_data)
        else:
            self._focus_on_scene()
        _startup_log(
            f"OBBViewer init complete in {(time_module.perf_counter() - t_init0):.2f}s"
        )

    def _apply_visual_theme(self) -> None:
        """Apply the unified Light/Dark visual theme across bg/points/text."""
        # Light
        if self.visual_theme_mode == 0:
            self.bg_color_index = 0  # white
            if hasattr(self, "sdp_color_index"):
                self.sdp_color_index = 3  # dark grey points
                if hasattr(self, "_rebuild_point_vbo_color"):
                    self._rebuild_point_vbo_color()
            self.overlay_text_rgb = (0.25, 0.25, 0.25)
            self.overlay_text_bg_rgba = (1.0, 1.0, 1.0, 0.6)
            return

        # Dark
        self.bg_color_index = 3  # black
        if hasattr(self, "sdp_color_index"):
            self.sdp_color_index = 1  # light grey points
            if hasattr(self, "_rebuild_point_vbo_color"):
                self._rebuild_point_vbo_color()
        self.overlay_text_rgb = (0.8, 0.8, 0.8)
        self.overlay_text_bg_rgba = (0.0, 0.0, 0.0, 0.6)

        # Random color modes are theme-dependent; refresh views when theme changes.
        if getattr(self, "color_mode", None) == COLOR_MODE_RANDOM:
            if hasattr(self, "_build_geometry_cache"):
                self._build_geometry_cache()
            if getattr(self, "tracked_all_instances", None) and hasattr(self.tracked_all_instances[0], "obb"):
                self._build_tracked_all_geometry()
            if hasattr(self, "_rgb_tracked_all_color_cache"):
                self._rgb_tracked_all_color_cache = None
            if hasattr(self, "_rebuild_rgb_projections"):
                self._rebuild_rgb_projections()
        if hasattr(self, "track_color_mode") and self.track_color_mode == 4:
            if (
                hasattr(self, "_rebuild_current_view")
                and getattr(self, "total_frames", 0) > 0
            ):
                self._rebuild_current_view()

    def _focus_on_scene(self) -> None:
        """Center camera on the bounding box of filtered detections."""
        # Get filtered OBBs based on probability threshold
        filtered_obbs, _ = self._get_filtered_obbs()

        if len(filtered_obbs) == 0:
            return

        # Get all corners from filtered boxes
        corners = filtered_obbs.bb3corners_world  # (N, 8, 3)

        # Compute bounding box of entire scene
        all_corners = corners.reshape(-1, 3)  # (N*8, 3)
        min_bounds = all_corners.min(dim=0).values.cpu().numpy()  # (3,)
        max_bounds = all_corners.max(dim=0).values.cpu().numpy()  # (3,)

        # Compute scene center
        scene_center = (min_bounds + max_bounds) / 2.0

        # Compute scene extent (diagonal of bounding box)
        scene_extent = np.linalg.norm(max_bounds - min_bounds)

        # Set camera target to scene center
        self.camera_target = scene_center

        # Set camera distance to fit entire scene in view
        # Using a factor of 1.5 to add some margin
        self.camera_distance = scene_extent * 1.5

        print("\n=== Camera Auto-Focused on Scene ===")
        print(
            f"Focused on {len(filtered_obbs)} filtered detections (threshold: {self.prob_threshold:.2f})"
        )
        print(
            f"Scene center: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})"
        )
        print(f"Scene extent: {scene_extent:.2f}")
        print(f"Camera distance: {self.camera_distance:.2f}")
        print("====================================\n")

    def _label_to_theme_random_color(self, label: str) -> np.ndarray:
        """Deterministic random color tuned to be visible in Light/Dark themes."""
        digest = hashlib.md5(label.encode("utf-8")).digest()
        hue = digest[0] / 255.0
        sat = 0.60 + 0.35 * (digest[1] / 255.0)
        if self.visual_theme_mode == 1:  # Dark background
            val = 0.72 + 0.23 * (digest[2] / 255.0)
        else:  # Light background
            # Slightly brighter palette for visibility over RGB image.
            val = 0.58 + 0.24 * (digest[2] / 255.0)
            # Avoid hard-to-see very light pastel yellows/pinks on bright imagery.
            is_yellow = 0.11 <= hue <= 0.20
            is_pink = hue >= 0.86 or hue <= 0.02
            if is_yellow or is_pink:
                val = min(val, 0.62)
                sat = max(sat, 0.82)
        r, g, b = colorsys.hsv_to_rgb(float(hue), float(sat), float(val))
        return np.array([r, g, b], dtype=np.float32)

    def _remap_label(self, label: str) -> str:
        """Apply display-name overrides. Subclasses may extend."""
        return label

    def _obbs_random_colors(self, obbs: ObbTW) -> torch.Tensor:
        """Theme-aware random colors per semantic class (stable across instances)."""
        sem_ids = obbs.sem_id.squeeze(-1).cpu().numpy().astype(int).tolist()
        labels = obbs.text_string()
        if isinstance(labels, str):
            labels = [labels]
        class_keys: list[str] = []
        for i, sid in enumerate(sem_ids):
            if sid in BOXY_SEM2NAME:
                class_keys.append(f"sem:{sid}:{BOXY_SEM2NAME[sid]}")
            else:
                # Fallback for unknown sem_id values: normalized label text.
                lbl = str(labels[i]) if i < len(labels) else "unknown"
                class_keys.append(f"lbl:{self._remap_label(lbl).strip().lower()}")
        colors_np = np.array(
            [self._label_to_theme_random_color(key) for key in class_keys],
            dtype=np.float32,
        )
        return torch.from_numpy(colors_np).float().to(obbs.device)

    def _save_camera_view(self) -> None:
        """Save current camera state to a .pt file."""
        view_data = {
            "camera_distance": self.camera_distance,
            "camera_azimuth": self.camera_azimuth,
            "camera_elevation": self.camera_elevation,
            "camera_target": torch.from_numpy(self.camera_target.copy()),
        }
        torch.save(view_data, self.view_save_path)
        print(f"Saved camera view to {self.view_save_path}")


    def _save_screenshot(self) -> None:
        """Save the current framebuffer as a PNG image."""
        from PIL import Image

        # Read pixels from the default framebuffer
        w, h = self.wnd.size
        pixel_ratio = self.wnd.pixel_ratio
        fb_w, fb_h = int(w * pixel_ratio), int(h * pixel_ratio)
        data = self.ctx.fbo.read(components=3)
        img = Image.frombytes("RGB", (fb_w, fb_h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Save with incrementing filename
        i = 0
        while True:
            path = os.path.join(self.root_path, f"screenshot_{i:03d}.jpg")
            if not os.path.exists(path):
                break
            i += 1
        img.save(path)
        print(f"Saved screenshot to {path}")

    def _apply_camera_view(self, view_data: dict) -> None:
        """Apply saved camera state from a dict."""
        self.camera_distance = float(view_data["camera_distance"])
        self.camera_azimuth = float(view_data["camera_azimuth"])
        self.camera_elevation = float(view_data["camera_elevation"])
        self.camera_target = view_data["camera_target"].numpy().astype(np.float32)
        print(
            f"Applied camera view: dist={self.camera_distance:.2f}, "
            f"az={self.camera_azimuth:.1f}, el={self.camera_elevation:.1f}"
        )

    def init_scene(self) -> None:
        """Initialize instanced quad rendering for thick lines (macOS compatible)."""

        # Shader for instanced quad rendering
        # Each line segment is an instance that draws a screen-aligned quad
        vertex_shader = """
            #version 330

            // Per-vertex attributes (quad template - 6 vertices forming 2 triangles)
            in vec2 in_quad_pos;  // Quad corner positions in normalized space

            // Per-instance attributes (one set per line segment)
            in vec3 start_pos;    // Line start in world space
            in vec3 end_pos;      // Line end in world space
            in vec3 line_color;   // RGB color
            in float line_prob;   // Probability for filtering

            // Uniforms
            uniform mat4 mvp;
            uniform float line_width;     // Width in pixels
            uniform vec2 viewport_size;   // Screen resolution

            // Outputs to fragment shader
            out vec3 v_color;
            out float v_prob;

            void main() {
                // Transform endpoints to clip space
                vec4 clip_start = mvp * vec4(start_pos, 1.0);
                vec4 clip_end = mvp * vec4(end_pos, 1.0);

                // Perspective divide to get normalized device coordinates
                vec2 ndc_start = clip_start.xy / clip_start.w;
                vec2 ndc_end = clip_end.xy / clip_end.w;

                // Convert to screen pixel coordinates
                vec2 screen_start = (ndc_start * 0.5 + 0.5) * viewport_size;
                vec2 screen_end = (ndc_end * 0.5 + 0.5) * viewport_size;

                // Compute line direction and perpendicular in screen space
                vec2 line_vec = screen_end - screen_start;
                float line_length = length(line_vec);
                vec2 line_dir = line_length > 0.0 ? line_vec / line_length : vec2(1.0, 0.0);
                vec2 line_perp = vec2(-line_dir.y, line_dir.x);

                // Expand quad corners in screen space
                // in_quad_pos.x: -1 (start) to +1 (end)
                // in_quad_pos.y: -1 (bottom) to +1 (top)
                float t = in_quad_pos.x * 0.5 + 0.5;  // [0, 1] along line
                vec2 center = mix(screen_start, screen_end, t);
                vec2 offset = line_perp * (line_width * 0.5) * in_quad_pos.y;

                // Final screen position
                vec2 screen_pos = center + offset;

                // Convert back to NDC
                vec2 ndc_pos = (screen_pos / viewport_size) * 2.0 - 1.0;

                // Interpolate depth between start and end
                float depth = mix(clip_start.z / clip_start.w, clip_end.z / clip_end.w, t);
                float w = mix(clip_start.w, clip_end.w, t);

                gl_Position = vec4(ndc_pos * w, depth * w, w);

                v_color = line_color;
                v_prob = line_prob;
            }
        """

        fragment_shader = """
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

        self.line_prog = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        # Shader for point cloud rendering (GL_POINTS with per-vertex color)
        point_vertex_shader = """
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
        """

        point_fragment_shader = """
            #version 330
            in vec3 v_color;
            uniform float alpha;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, alpha);
            }
        """

        self.point_prog = self.ctx.program(
            vertex_shader=point_vertex_shader, fragment_shader=point_fragment_shader
        )

        # Enable programmable point size for point cloud rendering
        self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)

        # Create quad template (shared by all instances)
        # Two triangles forming a rectangle: 6 vertices
        quad_vertices = np.array(
            [
                # First triangle
                -1.0,
                -1.0,  # Bottom-left
                1.0,
                -1.0,  # Bottom-right
                1.0,
                1.0,  # Top-right
                # Second triangle
                -1.0,
                -1.0,  # Bottom-left
                1.0,
                1.0,  # Top-right
                -1.0,
                1.0,  # Top-left
            ],
            dtype=np.float32,
        )

        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())

        # Enable blending for transparency
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

        # Line width parameters
        self.raw_line_width = 1
        self.tracked_all_line_width = 3

        # Cache for instance data
        self.cached_instance_vbo = None
        self.cached_instance_vao = None
        self.cached_instance_count = 0

        self.tracked_all_instance_vbo = None
        self.tracked_all_instance_vao = None
        self.tracked_all_instance_count = 0

        if not self._skip_precompute:
            self._build_geometry_cache()

    def _get_filtered_obbs(self) -> tuple[ObbTW, torch.Tensor]:
        """Get filtered OBBs based on probability threshold (with caching).

        Returns:
            Tuple of (filtered_obbs, indices) where indices are the positions in all_obbs
        """
        # Return cached result if threshold hasn't changed
        if (
            self._cached_filtered_obbs is not None
            and self._cached_prob_threshold == self.prob_threshold
        ):
            return self._cached_filtered_obbs, self._cached_filtered_indices

        # Filter using vectorized tensor operation (much faster than Python loops!)
        mask = (self.all_obbs.prob >= self.prob_threshold).reshape(-1)
        filtered_obbs = self.all_obbs[mask]
        indices = torch.where(mask)[0]  # Get actual indices

        # Update cache
        self._cached_filtered_obbs = filtered_obbs
        self._cached_prob_threshold = self.prob_threshold
        self._cached_filtered_indices = indices

        return filtered_obbs, indices

    def _get_semantic_embeddings(
        self, obbs: ObbTW, indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get semantic embeddings for OBBs from cache.

        Args:
            obbs: ObbTW tensor to get embeddings for
            indices: Optional indices in all_obbs (if None, assumes obbs == all_obbs)

        Returns:
            Embeddings tensor of shape (N, 384) normalized to unit length
        """
        if self._semantic_embeddings is None:
            raise RuntimeError(
                "Semantic embeddings not precomputed! Call _precompute_semantic_embeddings first."
            )

        # If requesting embeddings for all detections, return full cache
        if indices is None and len(obbs) == len(self.all_obbs):
            return self._semantic_embeddings

        # Extract subset using indices
        if indices is not None:
            return self._semantic_embeddings[indices]

        # Fallback: should not happen
        raise RuntimeError("Must provide indices to extract embeddings from cache")

    def _run_fusion(self) -> None:
        """Run fusion on all detections and build fused geometry."""
        # Get filtered detections (uses cache)
        filtered_obbs, filtered_indices = self._get_filtered_obbs()

        if len(filtered_obbs) == 0:
            print("No detections above threshold to fuse")
            return

        # Lazily compute embeddings if not precomputed (e.g. TrackerViewer with skip_precompute)
        if self._semantic_embeddings is None and len(self.all_obbs) > 0:
            from utils.fuse_3d_boxes import precompute_semantic_embeddings
            self._semantic_embeddings = precompute_semantic_embeddings(self.all_obbs)

        # Get semantic embeddings from cache using indices
        semantic_embeddings = self._get_semantic_embeddings(
            filtered_obbs, filtered_indices
        )

        # Run fusion
        print(
            f"Running fusion with {len(filtered_obbs)} out of {self.total_detections} obbs"
        )
        print(f"  IOU threshold: {self.fusion_iou_threshold:.3f}")
        print(f"  Min detections: {self.fusion_min_detections}")
        print(f"  Semantic threshold: {self.fusion_semantic_threshold:.3f}")

        # Create fuser with UI-controlled config
        # NOTE: Set conf_threshold=0.0 because we already filtered via _get_filtered_obbs()
        from utils.fuse_3d_boxes import BoundingBox3DFuser

        fuser = BoundingBox3DFuser(
            iou_threshold=self.fusion_iou_threshold,
            min_detections=self.fusion_min_detections,
            confidence_weighting=self.fusion_confidence_weighting,
            samp_per_dim=self.fusion_samp_per_dim,
            semantic_threshold=self.fusion_semantic_threshold,
            enable_nms=self.fusion_enable_nms,
            nms_iou_threshold=self.fusion_nms_iou_threshold,
            conf_threshold=0.0,  # Disable internal filtering (already done by viewer)
        )

        # Run fusion on filtered detections with semantic embeddings
        self.tracked_all_instances = fuser.fuse(
            filtered_obbs, semantic_embeddings=semantic_embeddings
        )

        # Build fused geometry for rendering
        self._build_tracked_all_geometry()

        # Update last fusion parameters
        self._last_fusion_iou_threshold = self.fusion_iou_threshold
        self._last_fusion_min_detections = self.fusion_min_detections
        self._last_fusion_confidence_weighting = self.fusion_confidence_weighting
        self._last_fusion_samp_per_dim = self.fusion_samp_per_dim
        self._last_fusion_semantic_threshold = self.fusion_semantic_threshold
        self._last_fusion_enable_nms = self.fusion_enable_nms
        self._last_fusion_nms_iou_threshold = self.fusion_nms_iou_threshold
        self._last_fusion_prob_threshold = self.prob_threshold  # Track prob threshold

        print(f"Fusion complete: {len(self.tracked_all_instances)} instances created")

    def _build_tracked_all_geometry(self) -> None:
        """Build per-instance line segment data for fused boxes."""
        if not self.tracked_all_instances:
            return

        # Clear text labels, positions, and colors
        self.tracked_all_text_labels = []
        self.tracked_all_label_positions = []
        self.tracked_all_label_colors = []

        # Stack all tracked-all OBBs into single tensor for batch color computation
        tracked_all_obbs_list = [
            instance.obb for instance in self.tracked_all_instances
        ]
        tracked_all_obbs = torch.stack(tracked_all_obbs_list)

        # Get colors based on color mode (SAME logic as raw detections)
        if self.color_mode == COLOR_MODE_PCA:
            # Fused instances are new OBBs not in original cache, so compute embeddings directly
            # Use standalone function to compute embeddings
            from utils.fuse_3d_boxes import precompute_semantic_embeddings

            embeddings = precompute_semantic_embeddings(tracked_all_obbs)
            # Use cached PCA model for consistent colors with raw detections
            colors = self._create_pca_colors_from_embeddings(
                embeddings, use_cached_pca=True
            )
            colors = colors.to(tracked_all_obbs.device)
        elif self.color_mode == COLOR_MODE_RANDOM:
            colors = self._obbs_random_colors(tracked_all_obbs)
        else:  # COLOR_MODE_PROBABILITY
            # Use probability-based jet colormap
            probs_np = tracked_all_obbs.prob.cpu().numpy().squeeze()
            colors_rgba = cm.jet(probs_np)  # (N, 4) RGBA
            colors_np = colors_rgba[:, :3]  # (N, 3) RGB
            colors = (
                torch.from_numpy(colors_np).float().to(tracked_all_obbs.device)
            )  # (N, 3)

        # Collect line segments from all fused instances
        all_start_points = []
        all_end_points = []
        all_colors = []
        all_probs = []

        for idx, instance in enumerate(self.tracked_all_instances):
            obb = instance.obb
            corners = obb.bb3corners_world.cpu().numpy().squeeze()  # (8, 3)

            # Get text label for this instance using safe extraction
            try:
                text_label = obb.reshape(1, -1).text_string()[0]
            except Exception:
                text_label = "Unknown"
            self.tracked_all_text_labels.append(text_label)

            # Compute center of the box for text placement
            box_center = corners.mean(axis=0)  # (3,)
            self.tracked_all_label_positions.append(box_center)

            # Get color for this instance from computed colors
            color = colors[idx].cpu().numpy()
            self.tracked_all_label_colors.append(color)
            prob = 1.0  # Always show fused boxes

            # Generate 12 line segments (one per edge)
            for i, j in BB3D_LINE_ORDERS:
                all_start_points.append(corners[i])
                all_end_points.append(corners[j])
                all_colors.append(color)
                all_probs.append(prob)

        if not all_start_points:
            return

        # Convert to numpy arrays
        start_points = np.array(all_start_points, dtype=np.float32)  # (M, 3)
        end_points = np.array(all_end_points, dtype=np.float32)  # (M, 3)
        colors = np.array(all_colors, dtype=np.float32)  # (M, 3)
        probs = np.array(all_probs, dtype=np.float32)[:, None]  # (M, 1)

        # Create per-instance data: [start_pos (3), end_pos (3), color (3), prob (1)] = 10 floats
        instance_data = np.concatenate(
            [start_points, end_points, colors, probs], axis=1
        )  # (M, 10)

        self.tracked_all_instance_count = len(instance_data)
        print(f"Created {self.tracked_all_instance_count} fused line segment instances")
        print(f"Created {len(self.tracked_all_text_labels)} text labels")

        # Create GPU buffers
        if self.tracked_all_instance_vbo is not None:
            self.tracked_all_instance_vbo.release()
        if self.tracked_all_instance_vao is not None:
            self.tracked_all_instance_vao.release()

        self.tracked_all_instance_vbo = self.ctx.buffer(instance_data.tobytes())

        # Create VAO with quad template + per-instance attributes
        self.tracked_all_instance_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),  # Per-vertex (shared quad)
                (
                    self.tracked_all_instance_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),  # Per-instance
            ],
        )

    def _build_geometry_cache(self) -> None:
        """Build and cache per-instance line segment data for instanced rendering."""
        N = len(self.all_obbs)
        if N == 0:
            return

        print("\n=== Building Instance Data for Thick Lines ===")

        # Get corners and probabilities as tensors
        corners = self.all_obbs.bb3corners_world  # (N, 8, 3)
        probs = self.all_obbs.prob.squeeze()  # (N,)

        # Get colors based on color mode
        if self.color_mode == COLOR_MODE_PCA:
            # Use PCA color mapping with cached embeddings
            embeddings = self._get_semantic_embeddings(self.all_obbs)
            colors = self._create_pca_colors_from_embeddings(embeddings)
            colors = colors.to(corners.device)
        elif self.color_mode == COLOR_MODE_RANDOM:
            colors = self._obbs_random_colors(self.all_obbs).to(corners.device)
        else:  # COLOR_MODE_PROBABILITY
            # Use probability-based jet colormap
            probs_np = probs.cpu().numpy()
            colors_rgba = cm.jet(probs_np)  # (N, 4) RGBA
            colors_np = colors_rgba[:, :3]  # (N, 3) RGB
            colors = torch.from_numpy(colors_np).float().to(corners.device)  # (N, 3)

        # Define edges as tensor (12 edges per box)
        edge_indices = torch.tensor(
            BB3D_LINE_ORDERS, dtype=torch.long, device=corners.device
        )  # (12, 2)

        # Create batch indices for all boxes and edges
        batch_indices = torch.arange(N, device=corners.device)[:, None].expand(
            N, 12
        )  # (N, 12)
        start_idx = edge_indices[:, 0][None, :].expand(N, 12)  # (N, 12)
        end_idx = edge_indices[:, 1][None, :].expand(N, 12)  # (N, 12)

        # Get all start and end points for all edges
        start_points = corners[batch_indices, start_idx]  # (N, 12, 3)
        end_points = corners[batch_indices, end_idx]  # (N, 12, 3)

        # Expand colors and probs to match edges
        colors_expanded = colors[:, None, :].expand(N, 12, 3)  # (N, 12, 3)
        probs_expanded = probs[:, None].expand(N, 12)  # (N, 12)

        # Create per-instance data: [start_pos (3), end_pos (3), color (3), prob (1)] = 10 floats
        instance_data = torch.cat(
            [
                start_points,  # (N, 12, 3)
                end_points,  # (N, 12, 3)
                colors_expanded,  # (N, 12, 3)
                probs_expanded.unsqueeze(-1),  # (N, 12, 1)
            ],
            dim=2,
        )  # (N, 12, 10)

        # Flatten to (N*12, 10) - one row per line segment instance
        instance_array = instance_data.reshape(-1, 10).cpu().numpy().astype("f4")

        self.cached_instance_count = len(instance_array)
        print(f"Created {self.cached_instance_count} line segment instances")

        # Create GPU buffers
        if self.cached_instance_vbo is not None:
            self.cached_instance_vbo.release()
        if self.cached_instance_vao is not None:
            self.cached_instance_vao.release()

        self.cached_instance_vbo = self.ctx.buffer(instance_array.tobytes())

        # Create VAO with quad template + per-instance attributes
        self.cached_instance_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),  # Per-vertex (shared quad)
                (
                    self.cached_instance_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),  # Per-instance (/i = instanced)
            ],
        )

        # Build axis line geometry (RGB for XYZ) after main geometry
        self._build_axis_geometry(corners, probs)

    def _build_axis_geometry(self, corners: torch.Tensor, probs: torch.Tensor) -> None:
        """Build per-instance axis line data for RGB axis visualization.

        Args:
            corners: (N, 8, 3) tensor of box corners
            probs: (N,) tensor of probabilities
        """
        N = corners.shape[0]
        if N == 0:
            return

        # Compute center of each box
        centers = corners.mean(dim=1)  # (N, 3)

        # Compute axis directions from corner layout:
        # Assuming standard OBB corner ordering:
        #   X direction: corners[1] - corners[0]
        #   Y direction: corners[3] - corners[0]
        #   Z direction: corners[4] - corners[0]
        x_axis = corners[:, 1, :] - corners[:, 0, :]  # (N, 3)
        y_axis = corners[:, 3, :] - corners[:, 0, :]  # (N, 3)
        z_axis = corners[:, 4, :] - corners[:, 0, :]  # (N, 3)

        # Scale axes to half length for visualization
        x_half = x_axis * 0.5
        y_half = y_axis * 0.5
        z_half = z_axis * 0.5

        # Compute axis endpoints from center
        x_end = centers + x_half  # (N, 3)
        y_end = centers + y_half  # (N, 3)
        z_end = centers + z_half  # (N, 3)

        # Define RGB colors for XYZ axes
        red = torch.tensor([1.0, 0.0, 0.0], device=corners.device)  # X axis
        green = torch.tensor([0.0, 1.0, 0.0], device=corners.device)  # Y axis
        blue = torch.tensor([0.0, 0.0, 1.0], device=corners.device)  # Z axis

        # Build instance data: 3 lines per box (X, Y, Z axes)
        # Format: [start_pos (3), end_pos (3), color (3), prob (1)] = 10 floats
        axis_data = []

        # X axis (Red)
        x_colors = red.unsqueeze(0).expand(N, 3)  # (N, 3)
        x_probs = probs.unsqueeze(-1)  # (N, 1)
        x_data = torch.cat([centers, x_end, x_colors, x_probs], dim=1)  # (N, 10)
        axis_data.append(x_data)

        # Y axis (Green)
        y_colors = green.unsqueeze(0).expand(N, 3)  # (N, 3)
        y_data = torch.cat([centers, y_end, y_colors, x_probs], dim=1)  # (N, 10)
        axis_data.append(y_data)

        # Z axis (Blue)
        z_colors = blue.unsqueeze(0).expand(N, 3)  # (N, 3)
        z_data = torch.cat([centers, z_end, z_colors, x_probs], dim=1)  # (N, 10)
        axis_data.append(z_data)

        # Concatenate all axis lines
        all_axis_data = torch.cat(axis_data, dim=0)  # (N*3, 10)
        axis_array = all_axis_data.cpu().numpy().astype("f4")

        self.axis_instance_count = len(axis_array)
        print(f"Created {self.axis_instance_count} axis line instances")

        # Create GPU buffers
        if self.axis_instance_vbo is not None:
            self.axis_instance_vbo.release()
        if self.axis_instance_vao is not None:
            self.axis_instance_vao.release()

        self.axis_instance_vbo = self.ctx.buffer(axis_array.tobytes())

        # Create VAO with quad template + per-instance attributes
        self.axis_instance_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),  # Per-vertex (shared quad)
                (
                    self.axis_instance_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),  # Per-instance
            ],
        )

    def _create_pca_colors_from_embeddings(
        self, embeddings: torch.Tensor, use_cached_pca: bool = False
    ) -> torch.Tensor:
        """Create PCA-based colors from semantic embeddings.

        Args:
            embeddings: (N, D) tensor of normalized embeddings
            use_cached_pca: If True, use cached PCA model for transformation

        Returns:
            (N, 3) tensor of RGB colors in [0, 1] range
        """
        print("\n=== Creating PCA Colors from Embeddings ===")

        # Convert to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()

        if use_cached_pca and self._pca_model is not None:
            # Use cached PCA model for consistent colors
            print("Using cached PCA model for consistent coloring")
            pca = self._pca_model
            pca_embeddings = pca.transform(embeddings_np)

            # Use cached min/max from all embeddings for consistent normalization
            pca_min = self._all_embeddings.min(axis=0)
            pca_max = self._all_embeddings.max(axis=0)
        else:
            # Fit new PCA model
            print("Fitting new PCA model")
            from sklearn.decomposition import PCA

            pca = PCA(n_components=3)
            pca_embeddings = pca.fit_transform(embeddings_np)

            # Cache the model and transformed embeddings for future use
            self._pca_model = pca
            self._all_embeddings = pca_embeddings

            # Compute min/max from current data
            pca_min = pca_embeddings.min(axis=0)
            pca_max = pca_embeddings.max(axis=0)

        # Normalize to [0, 1] range for RGB
        pca_normalized = (pca_embeddings - pca_min) / (pca_max - pca_min + 1e-8)

        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.3f}")
        print(f"  PC3: {pca.explained_variance_ratio_[2]:.3f}")
        print("==========================================\n")

        # Convert back to torch tensor
        colors = torch.from_numpy(pca_normalized.astype(np.float32))

        return colors

    def _project_to_screen(
        self, pos_3d: np.ndarray, mvp: np.ndarray, viewport_size: tuple
    ) -> tuple:
        """Project 3D world position to 2D screen coordinates.

        Args:
            pos_3d: 3D position in world space (3,)
            mvp: Model-view-projection matrix (4, 4) as numpy array
            viewport_size: (width, height) of viewport

        Returns:
            (screen_x, screen_y, is_visible) where is_visible indicates if the point is in front of camera
        """
        # Ensure mvp is a numpy array (convert from Matrix44 if needed)
        if not isinstance(mvp, np.ndarray):
            mvp = np.asarray(mvp, dtype=np.float32)

        # Convert to homogeneous coordinates
        pos_4d = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0], dtype=np.float32)

        # Transform to clip space using column-major order (OpenGL style)
        # Transpose the MVP matrix to handle pyrr's Matrix44 format correctly
        clip_pos = mvp.T @ pos_4d

        # Check if behind camera (negative w or z)
        if clip_pos[3] <= 0 or clip_pos[2] < 0:
            return (0, 0, False)

        # Perspective divide to get normalized device coordinates
        ndc = clip_pos[:3] / clip_pos[3]

        # Check if outside NDC cube [-1, 1]
        if abs(ndc[0]) > 1 or abs(ndc[1]) > 1:
            return (0, 0, False)

        # Convert to screen coordinates
        screen_x = (ndc[0] * 0.5 + 0.5) * viewport_size[0]
        screen_y = (1.0 - (ndc[1] * 0.5 + 0.5)) * viewport_size[
            1
        ]  # Flip Y for screen coords

        return (screen_x, screen_y, True)



    def _step_to_frame(self, target_idx: int) -> None:
        """Seek to frame index and update frame OBB sets."""
        if self.total_frames == 0:
            return
        target_idx = max(0, min(target_idx, self.total_frames - 1))
        self.current_frame_idx = target_idx
        ts = self.sorted_timestamps[target_idx]
        self._set_frame_obb_sets(raw=self.timed_obbs.get(ts))


    def _step_forward(self) -> None:
        if self.current_frame_idx < self.total_frames - 1:
            self._step_to_frame(self.current_frame_idx + 1)









    def on_key_event(self, key, action, modifiers):
        """Keyboard shortcuts for RGB playback."""
        super().on_key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return

        if key == self.wnd.keys.SPACE:
            if self.current_frame_idx >= self.total_frames - 1:
                self._step_to_frame(0)
                self.is_playing = True
            else:
                self.is_playing = not self.is_playing
            self._last_step_time = time_module.time()
        elif key == self.wnd.keys.RIGHT:
            self.is_playing = False
            self._step_forward()
        elif key == self.wnd.keys.LEFT:
            self.is_playing = False
            self._step_to_frame(self.current_frame_idx - 1)

    def _render_text_labels(self) -> None:
        """Render text labels for fused boxes in screen space."""
        if not self.show_text_labels or not self.tracked_all_label_positions:
            return

        # Get camera matrices and viewport
        _projection, _view, mvp = self.get_camera_matrices()
        full_w, full_h = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w  # viewport x offset (panels on left)

        # Convert Matrix44 to numpy array
        mvp_array = np.array(mvp, dtype=np.float32)

        text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        draw_list = imgui.get_foreground_draw_list()

        # Render each text label at each fused box position
        for i, pos_3d in enumerate(self.tracked_all_label_positions):
            # Get the text label for this fused instance
            text = self.tracked_all_text_labels[i]

            # Project 3D position to screen
            screen_x, screen_y, is_visible = self._project_to_screen(
                np.array(pos_3d), mvp_array, (w, h)
            )

            if not is_visible:
                continue

            # Offset screen_x by viewport origin (panels on left)
            screen_x += vp_x

            # Skip if label would overlap with UI controls panel
            if screen_x < vp_x + self.ui_panel_width:
                continue

            # Add small offset so text doesn't overlap with box center
            text_x = screen_x + 10
            text_y = screen_y - 10

            # Compute dimensions if enabled
            if self.show_dimensions and i < len(self.tracked_all_instances):
                obb = self.tracked_all_instances[i].obb
                corners = obb.bb3corners_world.cpu().numpy().squeeze()  # (8, 3)
                x_dim = np.linalg.norm(corners[1] - corners[0])  # Width
                y_dim = np.linalg.norm(corners[3] - corners[0])  # Height
                z_dim = np.linalg.norm(corners[4] - corners[0])  # Depth
                display_text = f"{text} ({y_dim:.2f}x{x_dim:.2f}x{z_dim:.2f})"
            else:
                display_text = text

            # Background color: darkened OBB color (matches 2D BB style)
            if i < len(self.tracked_all_label_colors):
                c = self.tracked_all_label_colors[i]
                bg_col = imgui.get_color_u32_rgba(
                    float(c[0]) * 0.4, float(c[1]) * 0.4, float(c[2]) * 0.4, 0.6
                )
            else:
                br, bg_, bb, ba = self.overlay_text_bg_rgba
                bg_col = imgui.get_color_u32_rgba(br, bg_, bb, ba)

            tw, th = imgui.calc_text_size(display_text)
            draw_list.add_rect_filled(
                text_x - 2, text_y - 1, text_x + tw + 2, text_y + th + 1, bg_col
            )
            draw_list.add_text(text_x, text_y, text_col, display_text)

    def render_3d(self, time: float, frame_time: float) -> None:
        """Render all OBBs using instanced quad rendering for thick lines."""
        # Playback auto-advance
        if self.is_playing and self.total_frames > 0:
            now = time_module.time()
            if now - self._last_step_time >= 1.0 / max(self.playback_fps, 0.1):
                self._last_step_time = now
                self._step_forward()

        full_w, h = self.wnd.size

        # Split viewport: panels on left, 3D on right
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)

        # Get camera matrices and viewport (match active 3D area)
        _projection, _view, mvp = self.get_camera_matrices()
        self._last_render_mvp = np.asarray(mvp, dtype=np.float32).copy()

        # Disable depth testing for line rendering (lines render on top)
        self.ctx.disable(self.ctx.DEPTH_TEST)

        # Render original detections with instanced quads
        if self.show_raw_set and self.cached_instance_vao is not None:
            self.line_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.line_prog["alpha"].write(np.array(self.alpha, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(
                np.array(self.prob_threshold, dtype="f4").tobytes()
            )
            self.line_prog["line_width"].value = float(self.raw_line_width)
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self.cached_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.cached_instance_count
            )

        # Render fused instances with instanced quads
        if self.show_tracked_all_set and self.tracked_all_instance_vao is not None:
            self.line_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(np.array(0.0, dtype="f4").tobytes())
            self.line_prog["line_width"].value = float(self.tracked_all_line_width)
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self.tracked_all_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.tracked_all_instance_count
            )

        # Render visible tracked subset with thicker lines
        if self.show_tracked_visible_set and getattr(self, "outline_instance_vao", None) is not None:
            self.line_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(np.array(0.0, dtype="f4").tobytes())
            self.line_prog["line_width"].value = float(
                getattr(self, "visible_line_width", self.tracked_all_line_width + 2)
            )
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self.outline_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.outline_instance_count
            )

        # Render axis lines (RGB for XYZ) when enabled
        if self.show_axis_lines and self.axis_instance_vao is not None:
            self.line_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(
                np.array(self.prob_threshold, dtype="f4").tobytes()
            )
            self.line_prog["line_width"].value = float(self.tracked_all_line_width)
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self.axis_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.axis_instance_count
            )

        # Restore full viewport for imgui rendering
        self.ctx.viewport = (0, 0, full_w, h)
        self.ctx.scissor = None

    def render_ui(self) -> None:
        """Render ImGui UI panel."""
        # Render text labels for fused boxes (before UI panels so they appear behind)
        if self.show_tracked_all_set:
            self._render_text_labels()

        # Get current window size dynamically
        w, h = self.wnd.size

        imgui.set_next_window_position(0, 0, imgui.ONCE)
        imgui.set_next_window_size(self.ui_panel_width, h, imgui.ALWAYS)  # Always match window height

        imgui.begin("OBB Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        # Render all the main controls
        self._render_main_controls()
        imgui.end()

    def _render_common_visual_controls(
        self,
        *,
        raw_checkbox_label: str = "Raw",
        tracked_all_checkbox_label: str = "Show Tracked All",
        tracked_all_line_label: str = "Tracked All Line Width",
        show_visible_checkbox: bool = True,
        show_visible_line_width: bool = False,
        show_sets_header: bool = True,
    ) -> None:
        """Render visualization controls for pure 3D box rendering."""
        imgui.push_item_width(200)
        _theme_changed, self.visual_theme_mode = imgui.combo(
            "Theme", self.visual_theme_mode, ["Light", "Dark"]
        )
        if _theme_changed:
            self._apply_visual_theme()
            if self.color_mode == COLOR_MODE_RANDOM:
                self._build_geometry_cache()
                if self.tracked_all_instances:
                    self._build_tracked_all_geometry()
        imgui.pop_item_width()

        if show_sets_header:
            imgui.text("OBB Sets")
        _changed, self.show_raw_set = imgui.checkbox(raw_checkbox_label, self.show_raw_set)
        _changed, self.show_tracked_all_set = imgui.checkbox(
            tracked_all_checkbox_label, self.show_tracked_all_set
        )
        if show_visible_checkbox:
            _changed, self.show_tracked_visible_set = imgui.checkbox(
                "Tracked Visible", self.show_tracked_visible_set
            )
        _changed, self.show_text_labels = imgui.checkbox(
            "Show Labels", self.show_text_labels
        )

        imgui.push_item_width(200)
        _changed, self.alpha = imgui.slider_float(
            "Detection Alpha", self.alpha, 0.0, 1.0
        )
        if imgui.tree_node("Line Widths"):
            _changed, self.raw_line_width = imgui.slider_int(
                "Raw Line Width", self.raw_line_width, 1, 10
            )
            _changed, self.tracked_all_line_width = imgui.slider_int(
                tracked_all_line_label, self.tracked_all_line_width, 1, 10
            )
            if show_visible_line_width and hasattr(self, "visible_line_width"):
                _changed, self.visible_line_width = imgui.slider_int(
                    "Visible Line Width", self.visible_line_width, 1, 10
                )
            imgui.tree_pop()
        imgui.pop_item_width()
        imgui.separator()

    def _section_header(self, title: str) -> None:
        """Draw a consistent section header in the left control panel."""
        imgui.separator()
        imgui.text(title)
        imgui.separator()

    def _render_fusion_controls(self) -> None:
        """Render fusion section UI (reusable by both OBBViewer and TrackerViewer)."""
        self._section_header("Fusion")

        # Check if fusion parameters are out of date
        fusion_out_of_date = (
            self._last_fusion_iou_threshold != self.fusion_iou_threshold
            or self._last_fusion_min_detections != self.fusion_min_detections
            or self._last_fusion_confidence_weighting
            != self.fusion_confidence_weighting
            or self._last_fusion_samp_per_dim != self.fusion_samp_per_dim
            or self._last_fusion_semantic_threshold != self.fusion_semantic_threshold
            or self._last_fusion_enable_nms != self.fusion_enable_nms
            or self._last_fusion_nms_iou_threshold != self.fusion_nms_iou_threshold
            or self._last_fusion_prob_threshold
            != self.prob_threshold
        )

        button_label = (
            "RUN FUSION (out of date)" if fusion_out_of_date else "RUN FUSION"
        )
        button_width = 400
        button_height = 40
        if fusion_out_of_date:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.15, 0.15, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.1, 0.1, 1.0)
        if imgui.button(button_label, width=button_width, height=button_height):
            self._run_fusion()
            if hasattr(self, "_rgb_tracked_all_cache_epoch"):
                self._rgb_tracked_all_cache_epoch += 1
                self._rgb_tracked_all_color_cache = None
                self._rgb_tracked_all_color_mode_cache = None
            if hasattr(self, "_rebuild_rgb_projections"):
                self._rebuild_rgb_projections()
        if fusion_out_of_date:
            imgui.pop_style_color(3)

        imgui.separator()

        imgui.push_item_width(200)
        prob_changed, new_prob = imgui.slider_float(
            "3DBB Conf Thresh", self.prob_threshold, 0.0, 1.0
        )
        if prob_changed:
            self.prob_threshold = new_prob
            self._cached_filtered_obbs = None
        imgui.pop_item_width()

        imgui.push_item_width(200)
        _changed, self.fusion_semantic_threshold = imgui.slider_float(
            "Semantic Merge Thresh", self.fusion_semantic_threshold, 0.0, 1.0
        )
        imgui.pop_item_width()

        imgui.separator()

        if self.fusion_enable_nms:
            imgui.push_item_width(200)
            _changed, self.fusion_nms_iou_threshold = imgui.slider_float(
                "NMS IoU Thresh", self.fusion_nms_iou_threshold, 0.5, 1.0
            )
            imgui.pop_item_width()

        imgui.text("Detection Statistics")
        filtered_obbs, _ = self._get_filtered_obbs()
        current_detections = len(filtered_obbs)
        imgui.text(f"Shown: {current_detections} / {self.total_detections}")
        if self.total_detections > 0:
            percentage = (current_detections / self.total_detections) * 100
            imgui.text(f"  ({percentage:.1f}%)")

        imgui.push_item_width(200)
        color_mode_names = ["PCA", "Prob", "Random"]
        changed, self.color_mode = imgui.combo(
            "Color Mode", self.color_mode, color_mode_names
        )
        if changed:
            self._build_geometry_cache()
            if self.tracked_all_instances:
                self._build_tracked_all_geometry()
            if hasattr(self, "_rgb_tracked_all_color_cache"):
                self._rgb_tracked_all_color_cache = None
                self._rgb_tracked_all_color_mode_cache = None
            if hasattr(self, "_rebuild_rgb_projections"):
                self._rebuild_rgb_projections()

        if self.color_mode == COLOR_MODE_PCA:
            imgui.text_colored("  PCA of text embeddings", 0.3, 1.0, 0.3)
        elif self.color_mode == COLOR_MODE_PROBABILITY:
            imgui.text_colored("  Jet colormap by probability", 1.0, 0.5, 0.0)

        imgui.separator()

        imgui.text("Fusion Parameters")
        imgui.push_item_width(200)
        _changed, self.fusion_iou_threshold = imgui.slider_float(
            "IOU Threshold", self.fusion_iou_threshold, 0.0, 1.0
        )
        _changed, self.fusion_min_detections = imgui.slider_int(
            "Min Detections", self.fusion_min_detections, 1, 10
        )
        _changed, self.fusion_samp_per_dim = imgui.slider_int(
            "IoU Samples", self.fusion_samp_per_dim, 1, 32
        )
        imgui.pop_item_width()

        imgui.push_item_width(200)
        weighting_modes = ["uniform", "linear", "quadratic", "robust"]
        current_idx = weighting_modes.index(self.fusion_confidence_weighting)
        weighting_changed, new_idx = imgui.combo(
            "Confidence Weighting", current_idx, weighting_modes
        )
        if weighting_changed:
            self.fusion_confidence_weighting = weighting_modes[new_idx]
        imgui.pop_item_width()

        imgui.separator()

        _changed, self.fusion_enable_nms = imgui.checkbox(
            "Enable Fused NMS", self.fusion_enable_nms
        )
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text("Remove redundant fused boxes with:")
            imgui.text("  • IoU > threshold")
            imgui.text("  • Semantic similarity > threshold")
            imgui.end_tooltip()

        if self.tracked_all_instances:
            valid_instances = [inst for inst in self.tracked_all_instances if hasattr(inst, "support_count")]
            imgui.text(f"Fused: {len(valid_instances)} instances")
            total_detections = sum(
                inst.support_count for inst in valid_instances
            )
            imgui.text(f"  From: {total_detections} detections")

    def _render_playback_controls(self) -> None:
        """Render playback controls (play/pause, frame slider, FPS)."""
        self._section_header("Playback")

        if imgui.button(
            "Play" if not self.is_playing else "Pause", width=90, height=28
        ):
            self.is_playing = not self.is_playing
            self._last_step_time = time_module.time()
        imgui.same_line()
        if imgui.button("<", width=30, height=28):
            self.is_playing = False
            self._step_to_frame(self.current_frame_idx - 1)
        imgui.same_line()
        if imgui.button(">", width=30, height=28):
            self.is_playing = False
            self._step_forward()

        if self.total_frames > 0:
            imgui.push_item_width(300)
            changed, new_frame = imgui.slider_int(
                "Frame",
                self.current_frame_idx,
                0,
                max(0, self.total_frames - 1),
            )
            if changed:
                self.is_playing = False
                self._step_to_frame(new_frame)
            imgui.pop_item_width()

            imgui.push_item_width(200)
            _changed, self.playback_fps = imgui.slider_float(
                "Playback FPS", self.playback_fps, 0.5, 60.0
            )
            imgui.pop_item_width()
            imgui.text("Space: play/pause, Left/Right: step")

    def _render_main_controls(self) -> None:
        """Render the main control panel UI."""
        self._render_playback_controls()
        self._render_fusion_controls()

        self._section_header("Visualization")
        self._render_common_visual_controls()

        self._section_header("Camera")
        if imgui.button("Focus on Scene"):
            self._focus_on_scene()
        imgui.same_line()
        if imgui.button("Screenshot"):
            self._save_screenshot()
        imgui.same_line()
        if imgui.button("Save View"):
            self._save_camera_view()

        self._section_header("Export")
        # Disable export button if no fused instances
        has_tracked_all = len(self.tracked_all_instances) > 0
        if not has_tracked_all:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.3, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.3, 0.3, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)

        if imgui.button("Export Fused OBBs to ADT"):
            if has_tracked_all:
                self._export_tracked_all_obbs_adt()

        if not has_tracked_all:
            imgui.pop_style_color(4)
            if imgui.is_item_hovered(imgui.HOVERED_ALLOW_WHEN_DISABLED):
                imgui.begin_tooltip()
                imgui.text("Run fusion first to enable export")
                imgui.end_tooltip()

        self._section_header("Scene Data")

    def _export_tracked_all_obbs_adt(self) -> None:
        """Export fused OBBs in ADT format using dump_obbs_adt."""
        if not self.tracked_all_instances:
            print("No fused instances to export. Run fusion first.")
            return

        # Stack all tracked-all OBBs into a single tensor
        tracked_all_obbs_list = [
            instance.obb for instance in self.tracked_all_instances
        ]
        tracked_all_obbs = torch.stack(tracked_all_obbs_list)

        # Assign unique instance IDs to each fused box
        ids = torch.arange(len(tracked_all_obbs), dtype=torch.int32)
        tracked_all_obbs.set_inst_id(ids)

        # Create timed_obbs dict with a single timestamp (static scene)
        # Use timestamp -1 for static fused boxes
        timed_obbs = {-1: tracked_all_obbs}

        # Export to ADT format
        output_path = os.path.join(self.root_path, "tracked_all_adt_export")
        os.makedirs(output_path, exist_ok=True)

        print(
            f"\n=== Exporting {len(tracked_all_obbs)} Tracked-All OBBs to ADT Format ==="
        )
        print(f"Output path: {output_path}")

        dump_obbs_adt(output_path, timed_obbs)

        print("=== Export Complete ===\n")


class SequenceOBBViewer(OBBViewer):
    """OBBViewer with sequence-aware features: RGB panel, trajectory, frustum, calibrations.

    Adds sequence context loading, RGB panel rendering, trajectory/frustum overlays,
    camera-based visibility culling, and 2D projection of 3D boxes onto images.
    """

    title = "OBB Sequence Viewer"

    def __init__(
        self,
        all_obbs: ObbTW,
        root_path: str,
        timed_obbs: dict[int, ObbTW] | None = None,
        loader_max_frames: int = 0,
        loader_start_frame: int = 0,
        init_rgb_text_scale: Optional[float] = None,
        init_color_mode: Optional[str] = None,
        init_image_panel_width: Optional[float] = None,
        skip_precompute: bool = False,
        load_view_data: dict | None = None,
        view_save_path: str = "",
        seq_name: str = "",
        scannet_scene: str | None = None,
        scannet_annotation_path: str = os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"),
        seq_ctx: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self._prebuilt_seq_ctx = seq_ctx
        self.scannet_scene = scannet_scene
        self.scannet_annotation_path = scannet_annotation_path
        self._loader_max_frames = int(loader_max_frames)
        self._loader_start_frame = int(loader_start_frame)

        # RGB video panel state
        self._rgb_texture = None
        self._rgb_tex_size = (0, 0)
        self.show_rgb = True
        self.rgb_panel_max_frac = 0.45
        if init_image_panel_width is not None:
            self.rgb_panel_max_frac = float(init_image_panel_width)
        if not hasattr(self, "_data_source"):
            self._data_source = "ca1m" if seq_name.startswith("ca1m") else "aria"
        if not hasattr(self, "_scannet_scene_dir"):
            self._scannet_scene_dir: Optional[str] = None
        if not hasattr(self, "_scannet_frame_ids"):
            self._scannet_frame_ids: Optional[list[str]] = None
        self._rgb_lru_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._rgb_lru_max_items = 32
        if not hasattr(self, "_loader"):
            self._loader = None
        self._rgb_img_scale: float = 1.0
        self._rgb_vrs_h: int = 0
        self._rgb_vrs_w: int = 0
        if not hasattr(self, "_vrs_is_nebula"):
            self._vrs_is_nebula = False
        if not hasattr(self, "_rgb_timestamps"):
            self._rgb_timestamps = np.array([])
        self.show_rgb_obbs = True
        self.show_rgb_raw = False
        self.show_rgb_tracked_all = False
        self.show_rgb_tracked_visible = True
        self.rgb_obb_thickness = 3.0
        self.show_rgb_labels = True
        self.rgb_text_scale = 1.4
        if init_rgb_text_scale is not None:
            self.rgb_text_scale = float(init_rgb_text_scale)
        self._rgb_label_fonts: dict[float, Any] = {}
        self._rgb_projected_raw_lines: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self._rgb_projected_tracked_all_lines: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self._rgb_projected_tracked_visible_lines: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self._rgb_projected_labels: list[tuple] = []
        self._rgb_projected_tracked_all_labels: list[tuple[float, float, str]] = []
        self._rgb_projected_tracked_visible_labels: list[tuple[float, float, str]] = []

        # Pose/calibration for projecting 3DBBs into RGB image
        if not hasattr(self, "traj"):
            self.traj = None
        if not hasattr(self, "pose_ts"):
            self.pose_ts = np.array([])
        if not hasattr(self, "calibs"):
            self.calibs = None
        if not hasattr(self, "calib_ts"):
            self.calib_ts = np.array([])
        # RGB overlay color cache
        self._rgb_tracked_all_color_cache: Optional[np.ndarray] = None
        self._rgb_tracked_all_color_mode_cache: Optional[int] = None
        self._rgb_tracked_all_cache_epoch = 0
        self._rgb_tracked_all_cache_epoch_used = -1

        # Navigation overlays (trajectory + frustum)
        self.show_trajectory = True
        self.traj_alpha = 1.0
        self.traj_tail_secs = 3.0
        self.show_frustum = True
        self.frustum_scale = 0.1
        self.traj_instance_vbo = None
        self.traj_instance_vao = None
        self.traj_instance_count = 0
        self._traj_all_segments = None
        self._traj_seg_ts = None
        self.frustum_instance_vbo = None
        self.frustum_instance_vao = None
        self.frustum_instance_count = 0
        self.outline_instance_vbo = None
        self.outline_instance_vao = None
        self.outline_instance_count = 0

        super().__init__(
            all_obbs=all_obbs,
            root_path=root_path,
            timed_obbs=timed_obbs,
            init_color_mode=init_color_mode,
            init_image_panel_width=init_image_panel_width,
            skip_precompute=skip_precompute,
            load_view_data=load_view_data,
            view_save_path=view_save_path,
            seq_name=seq_name,
            **kwargs,
        )

        # Load sequence context (RGB/traj/calib)
        has_rgb_timeline = len(getattr(self, "_rgb_timestamps", np.array([]))) > 0
        has_preloaded_context = (
            has_rgb_timeline and (self.traj is not None) and (self.calibs is not None)
        )
        skip_context = self._prebuilt_seq_ctx is not None and len(self._prebuilt_seq_ctx) == 0
        if self.total_frames > 0 and not has_preloaded_context and not skip_context:
            try:
                t_ctx0 = time_module.perf_counter()
                if self._prebuilt_seq_ctx is not None:
                    seq_ctx_data = self._prebuilt_seq_ctx
                else:
                    seq_ctx_data = _load_sequence_context_auto(
                        seq_name=seq_name,
                        scannet_scene=self.scannet_scene,
                        scannet_annotation_path=self.scannet_annotation_path,
                        with_sdp=False,
                        start_frame=max(0, self._loader_start_frame),
                        max_frames=(
                            self._loader_max_frames
                            if self._loader_max_frames > 0
                            else self.total_frames
                        ),
                    )
                self._data_source = seq_ctx_data.get("source", "aria")
                source_name = {
                    "ca1m": "CALoader",
                    "scannet": "ScanNetLoader",
                    "omni3d": "OmniLoader",
                }.get(self._data_source, "AriaLoader")
                print("Data source: " + source_name)
                self._loader = seq_ctx_data.get("loader", None)
                self._scannet_scene_dir = seq_ctx_data.get("scannet_scene_dir", None)
                self._rgb_num_frames = seq_ctx_data["rgb_num_frames"]
                self._rgb_timestamps = seq_ctx_data["rgb_timestamps"]
                self._rgb_images = seq_ctx_data.get("rgb_images", None)
                self._vrs_is_nebula = seq_ctx_data["is_nebula"]
                self.traj = seq_ctx_data["traj"]
                self.pose_ts = seq_ctx_data["pose_ts"]
                self.calibs = seq_ctx_data["calibs"]
                self.calib_ts = seq_ctx_data["calib_ts"]
                ts0 = self.sorted_timestamps[0]
                rgb0 = self._load_rgb_for_timestamp(ts0)
                if rgb0 is not None:
                    self._upload_rgb_texture(rgb0)
                self._rebuild_rgb_projections()
                _startup_log(
                    f"SequenceOBBViewer context applied in {(time_module.perf_counter() - t_ctx0):.2f}s"
                )
            except Exception as e:
                print(f"Warning: failed to initialize RGB panel: {e}")
                self._rgb_images = None
        if self.show_rgb and self._rgb_texture is None and self.total_frames > 0:
            ts0 = self.sorted_timestamps[0]
            rgb0 = self._load_rgb_for_timestamp(ts0)
            if rgb0 is not None:
                self._upload_rgb_texture(rgb0)

        # Build trajectory overlay geometry
        self._build_trajectory_geometry()
        if self.total_frames > 0:
            self._step_to_frame(self.current_frame_idx)

    def _compute_rgb_panel_width(self, win_w: int, panel_h: int) -> int:
        """Compute right RGB panel width as a fraction of window width."""
        if self._rgb_texture is None or not self.show_rgb:
            return 0
        return int(win_w * self.rgb_panel_max_frac)

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        """Return (width, height) of 3D viewport (right of controls + RGB panels)."""
        w, h = self.wnd.size
        ui_w = self.ui_panel_width
        panel_w = self._compute_rgb_panel_width(w, h)
        if panel_w > 0:
            return max(1, w - ui_w - panel_w), h
        return w, h

    def _get_rgb_label_font(self) -> Any:
        """Return cached ImGui font for current RGB text scale (larger glyphs)."""
        scale = max(0.5, float(self.rgb_text_scale))
        key = round(scale, 2)
        if key in self._rgb_label_fonts:
            return self._rgb_label_fonts[key]
        size_px = max(10.0, 16.0 * key)
        io = imgui.get_io()
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        for p in font_paths:
            if os.path.exists(p):
                try:
                    fnt = io.fonts.add_font_from_file_ttf(p, size_px)
                    self.imgui.refresh_font_texture()
                    self._rgb_label_fonts[key] = fnt
                    return fnt
                except Exception:
                    continue
        return None

    def _load_rgb_for_timestamp(self, ts_ns: int) -> Optional[np.ndarray]:
        """Load nearest RGB frame for the given timestamp."""

        def _to_scannet_fid(val: Any) -> str:
            try:
                as_int = int(val)
                return str(as_int)
            except Exception:
                return str(val)

        if (
            len(self._rgb_timestamps) == 0
            and getattr(self, "_data_source", None) != "aria"
        ):
            return None
        if getattr(self, "_data_source", None) == "aria":
            if len(self._rgb_timestamps) > 0:
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                ts_key = int(self._rgb_timestamps[idx])
            else:
                idx = -1
                ts_key = int(ts_ns)
        elif getattr(self, "_data_source", None) == "ca1m":
            idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
            ts_key = int(self._rgb_timestamps[idx])
        else:
            idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
            ts_key = int(self._rgb_timestamps[idx])
        if ts_key in self._rgb_lru_cache:
            img = self._rgb_lru_cache[ts_key]
            self._rgb_lru_cache.move_to_end(ts_key)
        elif (
            getattr(self, "_data_source", None) == "aria"
            and self._loader is not None
        ):
            frame_idx = int(self._loader._find_frame_by_timestamp(int(ts_ns)))
            stream_id = self._loader.stream_id[0]
            calibs = self._loader.calibs[0]
            out = self._loader._single(frame_idx, stream_id, calibs)
            if out is False or "img" not in out:
                return None
            img_t = out["img"][0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img_t * 255.0, 0, 255).astype(np.uint8)
            self._rgb_lru_cache[ts_key] = img
            if len(self._rgb_lru_cache) > self._rgb_lru_max_items:
                self._rgb_lru_cache.popitem(last=False)
        elif getattr(self, "_data_source", None) == "scannet":
            scene_dir = getattr(self, "_scannet_scene_dir", None)
            if scene_dir is None:
                if getattr(self, "_scannet_debug_points", True):
                    print(f"[ScanNet RGB] ts={ts_ns}: missing scene_dir")
                return None
            frame_ids = getattr(self, "_scannet_frame_ids", None)
            if frame_ids is not None and idx < len(frame_ids):
                frame_id = _to_scannet_fid(frame_ids[idx])
            else:
                frame_id = _to_scannet_fid(self._rgb_timestamps[idx])
            if frame_id in self._rgb_lru_cache:
                img = self._rgb_lru_cache[frame_id]
                self._rgb_lru_cache.move_to_end(frame_id)
            else:
                color_path = os.path.join(
                    scene_dir, "frames", "color", f"{frame_id}.png"
                )
                if not os.path.exists(color_path):
                    color_path = os.path.join(
                        scene_dir, "frames", "color", f"{frame_id}.jpg"
                    )
                img_bgr = cv2.imread(color_path)
                if img_bgr is None:
                    if getattr(self, "_scannet_debug_points", True):
                        print(
                            f"[ScanNet RGB] ts={ts_ns} idx={idx} frame_id={frame_id}: failed read {color_path}"
                        )
                    return None
                if getattr(self, "_scannet_debug_points", True):
                    print(
                        f"[ScanNet RGB] ts={ts_ns} idx={idx} frame_id={frame_id}: loaded {color_path}"
                    )
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self._rgb_lru_cache[frame_id] = img
                if len(self._rgb_lru_cache) > self._rgb_lru_max_items:
                    self._rgb_lru_cache.popitem(last=False)
        elif getattr(self, "_data_source", None) == "omni3d" and self._loader is not None:
            datum = self._loader.load(idx)
            img_t = datum["img0"][0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img_t * 255.0, 0, 255).astype(np.uint8)
            self._rgb_lru_cache[ts_key] = img
            if len(self._rgb_lru_cache) > self._rgb_lru_max_items:
                self._rgb_lru_cache.popitem(last=False)
        elif getattr(self, "_rgb_images", None) is not None:
            img = self._rgb_images[idx]
            if img is None:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self._rgb_lru_cache[ts_key] = img
            if len(self._rgb_lru_cache) > self._rgb_lru_max_items:
                self._rgb_lru_cache.popitem(last=False)
        else:
            return None

        self._rgb_vrs_h, self._rgb_vrs_w = img.shape[:2]
        if self.calibs is not None and not getattr(
            self, "_logged_calib_vrs_mismatch", False
        ):
            if isinstance(self.calibs, list):
                cam0 = self.calibs[0]
            elif hasattr(self.calibs, "dim") and self.calibs.dim() >= 2:
                cam0 = self.calibs[0]
            else:
                cam0 = self.calibs
            cam_w = cam0.size[..., 0].item()
            cam_h = cam0.size[..., 1].item()
            if abs(cam_w - self._rgb_vrs_w) > 1 or abs(cam_h - self._rgb_vrs_h) > 1:
                print(
                    f"[RGB] Calib/VRS resolution mismatch: calib=({cam_w:.0f}x{cam_h:.0f}), "
                    f"VRS=({self._rgb_vrs_w}x{self._rgb_vrs_h}) — scaling projection camera"
                )
            else:
                print(f"[RGB] Calib/VRS resolution match: {cam_w:.0f}x{cam_h:.0f}")
            self._logged_calib_vrs_mismatch = True
        if not self._vrs_is_nebula:
            img = np.rot90(img, k=3).copy()

        h, w = img.shape[:2]
        target_h = 1200
        scale = target_h / h
        self._rgb_img_scale = scale
        target_w = int(w * scale)
        img = cv2.resize(img, (target_w, target_h))
        return img

    def _upload_rgb_texture(self, img: np.ndarray) -> None:
        """Upload RGB image into GL texture for imgui rendering."""
        h, w = img.shape[:2]
        if self._rgb_texture is None or self._rgb_tex_size != (w, h):
            if self._rgb_texture is not None:
                self.imgui.remove_texture(self._rgb_texture)
                self._rgb_texture.release()
            self._rgb_texture = self.ctx.texture((w, h), 3, img.tobytes())
            self._rgb_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.imgui.register_texture(self._rgb_texture)
            self._rgb_tex_size = (w, h)
        else:
            self._rgb_texture.write(img.tobytes())

    def _step_to_frame(self, target_idx: int) -> None:
        """Seek to frame index, update RGB texture, frustum, and trajectory."""
        super()._step_to_frame(target_idx)
        if self.total_frames == 0:
            return
        ts = self.sorted_timestamps[self.current_frame_idx]
        rgb = self._load_rgb_for_timestamp(ts)
        if rgb is not None:
            self._upload_rgb_texture(rgb)
        elif getattr(self, "_data_source", None) == "scannet" and getattr(
            self, "_scannet_debug_points", True
        ):
            print(f"[ScanNet RGB] frame_idx={target_idx} ts={ts}: no RGB image")
        nav_ts = self._get_navigation_timestamp(self.current_frame_idx, ts)
        cam, T_wr = self._get_cam_and_pose(nav_ts)
        if cam is not None and T_wr is not None:
            self._build_frustum_geometry(cam, T_wr)
            self._update_trajectory_tail(nav_ts)
            self._build_visible_outline_geometry(cam, T_wr)
        self._rebuild_rgb_projections()

    def _build_visible_outline_geometry(self, cam: CameraTW, T_wr: PoseTW) -> None:
        """Build currently visible subset geometry for tracked-all set in fuse mode."""
        self.outline_instance_count = 0
        if self.outline_instance_vbo is not None:
            self.outline_instance_vbo.release()
            self.outline_instance_vbo = None
        if self.outline_instance_vao is not None:
            self.outline_instance_vao.release()
            self.outline_instance_vao = None

        if not self.tracked_all_instances:
            self._set_frame_obb_sets(
                raw=self.frame_obb_sets.raw,
                tracked_all=self._empty_obbs_like(),
                tracked_visible=self._empty_obbs_like(),
            )
            return

        tracked_all = torch.stack([inst.obb for inst in self.tracked_all_instances])
        _, bb2_valid = tracked_all.get_pseudo_bb2(
            cam.unsqueeze(0),
            T_wr.unsqueeze(0),
            num_samples_per_edge=10,
            valid_ratio=0.16667,
        )
        visible_mask = bb2_valid.squeeze(0)
        tracked_visible = self._subset_obbs(tracked_all, visible_mask)
        self._set_frame_obb_sets(
            raw=self.frame_obb_sets.raw,
            tracked_all=tracked_all,
            tracked_visible=tracked_visible,
        )
        if len(tracked_visible) == 0:
            return

        corners = tracked_visible.bb3corners_world
        M = len(tracked_visible)
        edge_indices = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long)
        batch_idx = torch.arange(M)[:, None].expand(M, 12)
        start_pts = corners[batch_idx, edge_indices[:, 0][None, :].expand(M, 12)]
        end_pts = corners[batch_idx, edge_indices[:, 1][None, :].expand(M, 12)]
        vis_color = torch.tensor([1.0, 0.6, 0.0], dtype=torch.float32)
        colors = vis_color.reshape(1, 1, 3).expand(M, 12, 3)
        probs = torch.ones(M, 12, 1, dtype=torch.float32)
        outline_data = (
            torch.cat([start_pts, end_pts, colors, probs], dim=2)
            .reshape(-1, 10)
            .numpy()
            .astype("f4")
        )
        self.outline_instance_count = len(outline_data)
        self.outline_instance_vbo = self.ctx.buffer(outline_data.tobytes())
        self.outline_instance_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self.outline_instance_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )

    def _build_trajectory_geometry(self) -> None:
        """Build full trajectory segment array for trajectory tail rendering."""
        if self.traj is None:
            return
        if hasattr(self.traj, "t"):
            positions = self.traj.t.cpu().float()
        else:
            positions = torch.stack(
                [pose.t.reshape(3).cpu().float() for pose in self.traj], dim=0
            )
        N = len(positions)
        if N < 2:
            return

        starts = positions[:-1]
        ends = positions[1:]
        num_segs = N - 1
        color = torch.tensor([0.0, 0.8, 0.8], dtype=torch.float32)
        colors = color.unsqueeze(0).expand(num_segs, 3)
        probs = torch.ones(num_segs, 1)
        traj_data = torch.cat([starts, ends, colors, probs], dim=1)
        self._traj_all_segments = traj_data.numpy().astype("f4")
        if len(self.pose_ts) >= 2:
            self._traj_seg_ts = self.pose_ts[:-1]
        else:
            self._traj_seg_ts = None

    def _update_trajectory_tail(self, ts_ns: int) -> None:
        """Upload trajectory segments within trailing time window."""
        if self._traj_all_segments is None or self._traj_seg_ts is None:
            self.traj_instance_count = 0
            return
        tail_ns = self.traj_tail_secs * 1e9
        start_ns = ts_ns - tail_ns
        mask = (self._traj_seg_ts >= start_ns) & (self._traj_seg_ts <= ts_ns)
        tail_data = self._traj_all_segments[mask]
        if len(tail_data) == 0:
            self.traj_instance_count = 0
            return

        self.traj_instance_count = len(tail_data)
        data_bytes = tail_data.tobytes()
        if self.traj_instance_vbo is not None and self.traj_instance_vbo.size >= len(
            data_bytes
        ):
            self.traj_instance_vbo.orphan(self.traj_instance_vbo.size)
            self.traj_instance_vbo.write(data_bytes)
        else:
            if self.traj_instance_vbo is not None:
                self.traj_instance_vbo.release()
            if self.traj_instance_vao is not None:
                self.traj_instance_vao.release()
            self.traj_instance_vbo = self.ctx.buffer(data_bytes)
            self.traj_instance_vao = self.ctx.vertex_array(
                self.line_prog,
                [
                    (self.quad_vbo, "2f", "in_quad_pos"),
                    (
                        self.traj_instance_vbo,
                        "3f 3f 3f 1f /i",
                        "start_pos",
                        "end_pos",
                        "line_color",
                        "line_prob",
                    ),
                ],
            )

    def _build_frustum_geometry(self, cam: CameraTW, T_wr: PoseTW) -> None:
        """Build frustum line geometry for current frame camera pose."""
        if self.frustum_instance_vbo is not None:
            self.frustum_instance_vbo.release()
            self.frustum_instance_vbo = None
        if self.frustum_instance_vao is not None:
            self.frustum_instance_vao.release()
            self.frustum_instance_vao = None
        self.frustum_instance_count = 0

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
        t_wc = origin
        pts_world = (R_wc @ pts_cam.T).T + t_wc

        segments = []
        color = torch.tensor([0.0, 0.8, 0.8], dtype=torch.float32)
        for i in range(4):
            segments.append(torch.cat([origin, pts_world[i], color, torch.ones(1)]))
        for i in range(4):
            j = (i + 1) % 4
            segments.append(
                torch.cat([pts_world[i], pts_world[j], color, torch.ones(1)])
            )
        frustum_data = torch.stack(segments).numpy().astype("f4")
        self.frustum_instance_count = len(frustum_data)
        self.frustum_instance_vbo = self.ctx.buffer(frustum_data.tobytes())
        self.frustum_instance_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self.frustum_instance_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )

    def _get_cam_and_pose(
        self, ts_ns: int
    ) -> tuple[Optional[CameraTW], Optional[PoseTW]]:
        """Get nearest camera calibration and world pose for timestamp."""
        if self.traj is None or self.calibs is None:
            return None, None
        if len(self.pose_ts) == 0 or len(self.calib_ts) == 0:
            return None, None
        if getattr(self, "_data_source", None) == "ca1m":
            if len(self._rgb_timestamps) > 0:
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                pose_idx = max(0, min(idx, len(self.traj) - 1))
                calib_idx = max(0, min(idx, len(self.calibs) - 1))
            else:
                pose_idx = int(find_nearest2(self.pose_ts, ts_ns))
                calib_idx = int(find_nearest2(self.calib_ts, ts_ns))
            T_wr = self.traj[pose_idx].float()
            cam = self.calibs[calib_idx].float()
            return cam, T_wr
        pose_idx = int(find_nearest2(self.pose_ts, ts_ns))
        calib_idx = int(find_nearest2(self.calib_ts, ts_ns))
        T_wr = self.traj[pose_idx].float()
        cam = self.calibs[calib_idx].float()
        return cam, T_wr

    def _get_navigation_timestamp(self, frame_idx: int, fallback_ts: int) -> int:
        """Timestamp used for camera/frustum/trajectory overlays."""
        return int(fallback_ts)

    def _project_obbs_for_rgb(
        self,
        obbs: ObbTW,
        cam: CameraTW,
        T_wr: PoseTW,
        colors: np.ndarray,
        labels: Optional[list[str]] = None,
    ) -> tuple[
        list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        list[tuple[float, float, str, np.ndarray]],
    ]:
        """Project OBB wireframes into RGB image coordinates."""
        if len(obbs) == 0:
            return [], []

        N_SUB = 10
        corners = obbs.bb3corners_world
        edge_idx = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long)
        p0 = corners[:, edge_idx[:, 0], :]
        p1 = corners[:, edge_idx[:, 1], :]
        t_interp = torch.linspace(0, 1, N_SUB + 1)
        edge_pts = (
            p0[:, :, None, :] * (1 - t_interp[None, None, :, None])
            + p1[:, :, None, :] * t_interp[None, None, :, None]
        )
        S = N_SUB + 1
        pts_world = edge_pts.reshape(-1, 3)
        T_world_cam = T_wr @ cam.T_camera_rig.inverse()
        pts_cam = T_world_cam.inverse().transform(pts_world)
        proj_cam = cam
        if self._rgb_vrs_w > 0 and self._rgb_vrs_h > 0:
            cam_w = cam.size[..., 0].item()
            cam_h = cam.size[..., 1].item()
            if abs(cam_w - self._rgb_vrs_w) > 1 or abs(cam_h - self._rgb_vrs_h) > 1:
                proj_cam = cam.scale_to_size((self._rgb_vrs_w, self._rgb_vrs_h))
        pts_2d, valid = proj_cam.project(
            pts_cam.unsqueeze(0),
            fov_deg=140.0 if self._vrs_is_nebula else 120.0,
        )
        pts_2d = pts_2d.squeeze(0).cpu().numpy()
        valid = valid.squeeze(0).cpu().numpy()

        if not self._vrs_is_nebula:
            old_x = pts_2d[:, 0].copy()
            old_y = pts_2d[:, 1].copy()
            pts_2d[:, 0] = self._rgb_vrs_h - 1 - old_y
            pts_2d[:, 1] = old_x

        N = len(obbs)
        pts_2d = pts_2d.reshape(N, 12, S, 2)
        valid = valid.reshape(N, 12, S)

        lines: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        out_labels: list[tuple[float, float, str, np.ndarray]] = []
        for i in range(N):
            lines.append((pts_2d[i], valid[i], colors[i]))
            if labels is not None and i < len(labels):
                v = valid[i]
                if v.any():
                    all_valid_pts = pts_2d[i][v]
                    out_labels.append(
                        (
                            float(all_valid_pts[:, 0].mean()),
                            float(all_valid_pts[:, 1].mean()),
                            labels[i],
                            colors[i],
                        )
                    )
        return lines, out_labels

    def _draw_projected_labels(
        self,
        draw_list,
        labels: list[tuple],
        img_min,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Draw text labels on an RGB panel."""
        text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        label_scale = max(0.5, float(self.rgb_text_scale))
        imgui.set_window_font_scale(label_scale)
        for label_item in labels:
            lx, ly, label = label_item[0], label_item[1], label_item[2]
            if len(label_item) >= 4:
                c = label_item[3]
                bg_col = imgui.get_color_u32_rgba(
                    float(c[0]) * 0.4, float(c[1]) * 0.4, float(c[2]) * 0.4, 0.6
                )
            else:
                br, bg_, bb, ba = self.overlay_text_bg_rgba
                bg_col = imgui.get_color_u32_rgba(br, bg_, bb, ba)
            sx = img_min.x + lx * scale_x
            sy = img_min.y + ly * scale_y - (14.0 * label_scale)
            tw, th = imgui.calc_text_size(label)
            draw_list.add_rect_filled(sx - 2, sy - 1, sx + tw + 2, sy + th + 1, bg_col)
            draw_list.add_text(sx, sy, text_col, label)
        imgui.set_window_font_scale(1.0)

    def _rebuild_rgb_projections(self) -> None:
        """Recompute projected 3DBB overlays for current frame."""
        self._rgb_projected_raw_lines = []
        self._rgb_projected_tracked_all_lines = []
        self._rgb_projected_tracked_visible_lines = []
        self._rgb_projected_labels = []
        self._rgb_projected_tracked_all_labels = []
        self._rgb_projected_tracked_visible_labels = []

        if self.total_frames == 0 or self.current_frame_idx >= self.total_frames:
            return
        if getattr(self, "_rgb_images", None) is None:
            return
        ts = self.sorted_timestamps[self.current_frame_idx]
        nav_ts = self._get_navigation_timestamp(self.current_frame_idx, ts)
        cam, T_wr = self._get_cam_and_pose(nav_ts)
        if cam is None or T_wr is None:
            return

        raw_obbs = self.timed_obbs.get(ts)
        if raw_obbs is not None and len(raw_obbs) > 0:
            raw_colors = np.tile(
                np.array([[0.6, 0.6, 0.6]], dtype=np.float32), (len(raw_obbs), 1)
            )
            raw_lines, _ = self._project_obbs_for_rgb(raw_obbs, cam, T_wr, raw_colors)
            self._rgb_projected_raw_lines = raw_lines

        if self.tracked_all_instances:
            tracked_all_obbs = torch.stack(
                [inst.obb for inst in self.tracked_all_instances]
            )
            reuse_cached_colors = (
                self._rgb_tracked_all_color_cache is not None
                and self._rgb_tracked_all_color_mode_cache == self.color_mode
                and self._rgb_tracked_all_cache_epoch_used
                == self._rgb_tracked_all_cache_epoch
                and len(self._rgb_tracked_all_color_cache) == len(tracked_all_obbs)
            )
            if reuse_cached_colors:
                tracked_all_colors = self._rgb_tracked_all_color_cache
            else:
                if self.color_mode == COLOR_MODE_PROBABILITY:
                    probs = tracked_all_obbs.prob.squeeze(-1).cpu().numpy()
                    tracked_all_colors = cm.jet(probs)[:, :3].astype(np.float32)
                elif self.color_mode == COLOR_MODE_RANDOM:
                    tracked_all_colors = (
                        self._obbs_random_colors(tracked_all_obbs)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                elif self.color_mode == COLOR_MODE_PCA:
                    tracked_all_colors = (
                        self._create_pca_colors_from_embeddings(
                            self._get_semantic_embeddings(tracked_all_obbs),
                            use_cached_pca=True,
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                else:
                    tracked_all_colors = (
                        self._obbs_random_colors(tracked_all_obbs)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                self._rgb_tracked_all_color_cache = tracked_all_colors
                self._rgb_tracked_all_color_mode_cache = self.color_mode
                self._rgb_tracked_all_cache_epoch_used = (
                    self._rgb_tracked_all_cache_epoch
                )
            tracked_all_labels = (
                self.tracked_all_text_labels
                if self.tracked_all_text_labels
                else [f"tracked_{i}" for i in range(len(tracked_all_obbs))]
            )
            tracked_all_lines, tracked_all_labels_xy = self._project_obbs_for_rgb(
                tracked_all_obbs,
                cam,
                T_wr,
                tracked_all_colors,
                labels=tracked_all_labels,
            )
            self._rgb_projected_tracked_all_lines = tracked_all_lines
            self._rgb_projected_tracked_all_labels = tracked_all_labels_xy
            tracked_visible_obbs = self.frame_obb_sets.tracked_visible
            if tracked_visible_obbs is not None and len(tracked_visible_obbs) > 0:
                vis_color = np.tile(
                    np.array([[1.0, 0.6, 0.0]], dtype=np.float32),
                    (len(tracked_visible_obbs), 1),
                )
                vis_labels = tracked_visible_obbs.text_string()
                if isinstance(vis_labels, str):
                    vis_labels = [vis_labels]
                vis_lines, vis_labels_xy = self._project_obbs_for_rgb(
                    tracked_visible_obbs, cam, T_wr, vis_color, labels=vis_labels
                )
                self._rgb_projected_tracked_visible_lines = vis_lines
                self._rgb_projected_tracked_visible_labels = vis_labels_xy

    def render_3d(self, time: float, frame_time: float) -> None:
        """Render 3D boxes + trajectory + frustum overlays."""
        super().render_3d(time, frame_time)

        # Render navigation overlays (trajectory + frustum) in the 3D viewport
        full_w, _ = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)
        self.ctx.disable(self.ctx.DEPTH_TEST)

        mvp = self._last_render_mvp
        if mvp is None:
            return
        mvp_bytes = mvp.astype("f4").tobytes()
        viewport_bytes = np.array([w, h], dtype="f4").tobytes()
        prob_zero_bytes = np.array(0.0, dtype="f4").tobytes()

        if self.show_trajectory and self.traj_instance_vao is not None:
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["alpha"].write(
                np.array(self.traj_alpha, dtype="f4").tobytes()
            )
            self.line_prog["prob_threshold"].write(prob_zero_bytes)
            self.line_prog["line_width"].value = 2.0
            self.line_prog["viewport_size"].write(viewport_bytes)
            self.traj_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.traj_instance_count
            )
        if self.show_frustum and self.frustum_instance_vao is not None:
            self.line_prog["mvp"].write(mvp_bytes)
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(prob_zero_bytes)
            self.line_prog["line_width"].value = 2.0
            self.line_prog["viewport_size"].write(viewport_bytes)
            self.frustum_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.frustum_instance_count
            )

        # Restore full viewport
        self.ctx.viewport = (0, 0, full_w, h)
        self.ctx.scissor = None

    def render_ui(self) -> None:
        """Render UI panel + RGB panel."""
        super().render_ui()

        # Right RGB panel
        if self._rgb_texture is not None and self.show_rgb:
            tex_w, tex_h = self._rgb_tex_size
            win_w, win_h = self.wnd.size
            panel_h = win_h
            panel_w = self._compute_rgb_panel_width(win_w, win_h)
            panel_x = self.ui_panel_width

            imgui.set_next_window_position(panel_x, 0, imgui.ALWAYS)
            imgui.set_next_window_size(panel_w, panel_h, imgui.ALWAYS)
            expanded, _ = imgui.begin(
                "RGB View",
                flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
            )
            if expanded:
                avail_w, avail_h = imgui.get_content_region_available()
                img_scale = min(avail_w / tex_w, avail_h / tex_h)
                draw_w = tex_w * img_scale
                draw_h = tex_h * img_scale
                imgui.image(self._rgb_texture.glo, draw_w, draw_h)
                img_min = imgui.get_item_rect_min()
                scale_x = draw_w / tex_w * self._rgb_img_scale
                scale_y = draw_h / tex_h * self._rgb_img_scale
                draw_list = imgui.get_window_draw_list()

                if self.show_rgb_obbs and self.show_rgb_raw:
                    for edge_pts, edge_valid, color in self._rgb_projected_raw_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                    y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                    x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                    y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                    draw_list.add_line(
                                        x0, y0, x1, y1, col, self.rgb_obb_thickness
                                    )

                if self.show_rgb_obbs and self.show_rgb_tracked_all:
                    for (
                        edge_pts,
                        edge_valid,
                        color,
                    ) in self._rgb_projected_tracked_all_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                    y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                    x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                    y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                    draw_list.add_line(
                                        x0, y0, x1, y1, col, self.rgb_obb_thickness
                                    )
                if self.show_rgb_obbs and self.show_rgb_tracked_visible:
                    for (
                        edge_pts,
                        edge_valid,
                        color,
                    ) in self._rgb_projected_tracked_visible_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                    y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                    x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                    y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                    draw_list.add_line(
                                        x0,
                                        y0,
                                        x1,
                                        y1,
                                        col,
                                        self.rgb_obb_thickness + 1.0,
                                    )

                if self.show_rgb_tracked_visible:
                    self._rgb_projected_labels = list(
                        self._rgb_projected_tracked_visible_labels
                    )
                elif self.show_rgb_tracked_all:
                    self._rgb_projected_labels = list(
                        self._rgb_projected_tracked_all_labels
                    )
                else:
                    self._rgb_projected_labels = []

                if self.show_rgb_labels and self._rgb_projected_labels:
                    self._draw_projected_labels(
                        draw_list,
                        self._rgb_projected_labels,
                        img_min,
                        scale_x,
                        scale_y,
                    )
            imgui.end()

    def _render_common_visual_controls(
        self,
        *,
        tracked_all_checkbox_label: str = "Show Tracked All",
        tracked_all_line_label: str = "Tracked All Line Width",
        show_visible_line_width: bool = False,
    ) -> None:
        """Render visual controls including trajectory/frustum/RGB toggles."""
        super()._render_common_visual_controls(
            tracked_all_checkbox_label=tracked_all_checkbox_label,
            tracked_all_line_label=tracked_all_line_label,
            show_visible_line_width=show_visible_line_width,
        )
        _changed, self.show_trajectory = imgui.checkbox(
            "Show Trajectory", self.show_trajectory
        )
        _changed, self.show_frustum = imgui.checkbox(
            "Show Frustum", self.show_frustum
        )
        _changed, self.show_rgb = imgui.checkbox("Show RGB Panel", self.show_rgb)
        if self.show_rgb:
            imgui.push_item_width(200)
            _changed, self.rgb_panel_max_frac = imgui.slider_float(
                "RGB Panel Width", self.rgb_panel_max_frac, 0.25, 0.75
            )
            imgui.pop_item_width()
            _changed, self.show_rgb_obbs = imgui.checkbox(
                "Show RGB OBBs", self.show_rgb_obbs
            )
            _changed, self.show_rgb_raw = imgui.checkbox(
                "Show RGB Raw", self.show_rgb_raw
            )
            _changed, self.show_rgb_tracked_visible = imgui.checkbox(
                "Show RGB Tracked Visible", self.show_rgb_tracked_visible
            )
            _changed, self.show_rgb_tracked_all = imgui.checkbox(
                "Show RGB Tracked All", self.show_rgb_tracked_all
            )
            _changed, self.show_rgb_labels = imgui.checkbox(
                "Show RGB Labels", self.show_rgb_labels
            )
            imgui.push_item_width(200)
            _changed, self.rgb_obb_thickness = imgui.slider_float(
                "RGB 3DBB Width", self.rgb_obb_thickness, 1.0, 10.0
            )
            _changed, self.rgb_text_scale = imgui.slider_float(
                "RGB Text Scale", self.rgb_text_scale, 0.5, 5.0
            )
            imgui.pop_item_width()
        imgui.push_item_width(200)
        _changed, self.traj_alpha = imgui.slider_float(
            "Traj Alpha", self.traj_alpha, 0.0, 1.0
        )
        changed, self.traj_tail_secs = imgui.slider_float(
            "Traj Tail (s)", self.traj_tail_secs, 0.5, 30.0
        )
        if changed and self.total_frames > 0:
            ts = self.sorted_timestamps[self.current_frame_idx]
            self._update_trajectory_tail(ts)
        changed, self.frustum_scale = imgui.slider_float(
            "Frustum Scale", self.frustum_scale, 0.001, 0.5
        )
        if changed and self.total_frames > 0:
            ts = self.sorted_timestamps[self.current_frame_idx]
            cam, T_wr = self._get_cam_and_pose(ts)
            if cam is not None and T_wr is not None:
                self._build_frustum_geometry(cam, T_wr)
        imgui.pop_item_width()


class TrackerViewer(SequenceOBBViewer):
    """Online 3D bounding box tracker visualization.

    Subclasses SequenceOBBViewer to reuse all rendering infrastructure (shaders, GPU
    geometry building, 3D rendering pipeline) while adding frame-by-frame
    playback and tracker parameter controls.
    """

    title = "OBB Tracker Viewer"
    window_size = (2250 * scale_factor, 1100 * scale_factor)

    def __init__(
        self,
        timed_obbs: dict[int, ObbTW],
        root_path: str,
        loader_max_frames: int = 0,
        loader_start_frame: int = 0,
        init_rgb_text_scale: Optional[float] = None,
        init_color_mode: Optional[str] = None,
        init_follow: bool = False,
        init_follow_behind: Optional[float] = None,
        init_follow_above: Optional[float] = None,
        init_follow_look_ahead: Optional[float] = None,
        force_cpu: bool = False,
        bb2d_csv_path: str = "",
        autorecord: bool = False,
        record_fps: float = 10.0,
        teaser: bool = False,
        already_tracked: bool = False,
        init_show_obs: bool = False,
        init_image_panel_width: Optional[float] = None,
        verbose: bool = False,
        scannet_scene: str | None = None,
        scannet_annotation_path: str = os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"),
        seq_ctx: dict | None = None,
        freeze_tracker: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize tracker viewer.

        Args:
            timed_obbs: Dict mapping timestamp -> ObbTW tensor of detections per frame
            root_path: Root path for the sequence
            force_cpu: Force IoU computation on CPU
            seq_ctx: Pre-built sequence context dict. When provided, skips
                _load_sequence_context_auto() and uses this context directly.
            freeze_tracker: If True, don't run tracker updates (fuse mode).
        """
        self._prebuilt_seq_ctx = seq_ctx
        t_init0 = time_module.perf_counter()
        _startup_log("TrackerViewer init start")
        self.timed_obbs = timed_obbs
        self.sorted_timestamps = sorted(timed_obbs.keys())
        self.total_frames = len(self.sorted_timestamps)
        if self.total_frames >= 2:
            deltas_ns = [
                self.sorted_timestamps[i + 1] - self.sorted_timestamps[i]
                for i in range(self.total_frames - 1)
            ]
            self._median_frame_delta_ns = int(np.median(np.asarray(deltas_ns)))
        else:
            self._median_frame_delta_ns = int(1e9 / 30.0)

        # Playback state
        self.current_frame_idx = 0
        self.is_playing = False
        self.freeze_tracker = freeze_tracker
        self.playback_fps = _infer_fps_from_timestamps_ns(
            np.array(self.sorted_timestamps, dtype=np.int64),
            fallback=10.0,
            source=("scannet" if scannet_scene is not None else None),
        )
        self._last_step_time = 0.0
        # Confidence thresholds (tracker mode)
        self.raw_conf_threshold = 0.55  # incoming per-frame detections

        # Recording state
        self._recording = False
        self._record_dir: Optional[str] = None
        self._record_frame_idx = 0
        self._record_last_playback_frame: int = -1
        self._autorecord = autorecord
        # 0 => auto-detect from RGB timeline after loader context is available.
        self._record_fps = float(record_fps) if record_fps > 0 else 0.0

        # Tracker parameters (adjustable in UI)
        self.tracker_iou_threshold = 0.25
        self.tracker_min_hits = 8
        self.tracker_conf_threshold = 0.55
        self.tracker_force_cpu = force_cpu
        self.tracker_merge_iou = 0.5
        self.tracker_merge_sem = 0.7
        self.tracker_merge_iou_2d = 0.7
        self.tracker_merge_interval = 5
        self.tracker_min_conf_mass = 4.0
        self.tracker_min_obs_points = 4
        self.tracker_verbose = verbose

        # Create tracker
        self.tracker = BoundingBox3DTracker(
            iou_threshold=self.tracker_iou_threshold,
            min_hits=self.tracker_min_hits,
            conf_threshold=self.tracker_conf_threshold,
            force_cpu=self.tracker_force_cpu,
            merge_iou_threshold=self.tracker_merge_iou,
            merge_semantic_threshold=self.tracker_merge_sem,
            merge_iou_2d_threshold=self.tracker_merge_iou_2d,
            merge_interval=self.tracker_merge_interval,
            min_confidence_mass=self.tracker_min_conf_mass,
            min_obs_points=self.tracker_min_obs_points,
            verbose=self.tracker_verbose,
        )

        # Load trajectory/calibration/RGB from AriaLoader
        seq_name = os.path.basename(root_path)
        self._seq_name = seq_name
        self.scannet_scene = scannet_scene
        self.scannet_annotation_path = scannet_annotation_path
        self._loader_max_frames = int(loader_max_frames)
        self._loader_start_frame = int(loader_start_frame)
        boxy_data_dir = os.path.join(os.path.expanduser("~"), "boxy_data", seq_name)
        self._boxy_data_dir = boxy_data_dir

        # RGB image display state
        self._rgb_texture = None
        self._rgb_tex_size = (0, 0)
        self._rgb_panel_rect: Optional[Tuple[float, float, float, float]] = None
        self.show_rgb = True
        self.rgb_panel_max_frac = 0.45
        self._init_image_panel_width = init_image_panel_width
        if init_image_panel_width is not None:
            self.rgb_panel_max_frac = float(init_image_panel_width)
        self.show_rgb_obbs = True
        self.rgb_obb_thickness = 3.0
        self.rgb_bb2_thickness = 4.0
        self.rgb_text_scale = 1.4
        if init_rgb_text_scale is not None:
            self.rgb_text_scale = float(init_rgb_text_scale)

        self.show_rgb_labels = True
        self.show_rgb_raw = True
        self.show_rgb_tracked_all = False
        self.show_rgb_tracked_visible = True
        self.show_rgb_visible_only = False
        self.rgb_max_projected_boxes = 512
        self.debug_visibility = False
        self._rgb_projected_lines: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []  # (pts_2d, valid, color) per visible track
        self._rgb_img_scale: float = 1.0  # VRS original → texture resize scale
        self._rgb_vrs_h: int = 0  # Original VRS image height (before rotation)
        self._rgb_vrs_w: int = 0  # Original VRS image width (before rotation)
        try:
            t_ctx0 = time_module.perf_counter()
            if self._prebuilt_seq_ctx is not None:
                seq_ctx = self._prebuilt_seq_ctx
            else:
                seq_ctx = _load_sequence_context_auto(
                    seq_name=seq_name,
                    scannet_scene=self.scannet_scene,
                    scannet_annotation_path=self.scannet_annotation_path,
                    with_sdp=True,
                    start_frame=max(0, self._loader_start_frame),
                    max_frames=(
                        self._loader_max_frames
                        if self._loader_max_frames > 0
                        else self.total_frames
                    ),
                )
            self._data_source = seq_ctx.get("source", "aria")
            source_name = {
                "ca1m": "CALoader",
                "scannet": "ScanNetLoader",
            }.get(self._data_source, "AriaLoader")
            print("Data source: " + source_name)
            self._loader = seq_ctx.get("loader", None)
            self._scannet_scene_dir = seq_ctx.get("scannet_scene_dir", None)
            self._scannet_frame_ids = seq_ctx.get("scannet_frame_ids", None)
            self._rgb_num_frames = seq_ctx["rgb_num_frames"]
            self._rgb_timestamps = seq_ctx["rgb_timestamps"]
            self._rgb_images = seq_ctx.get("rgb_images", None)
            self._vrs_is_nebula = seq_ctx["is_nebula"]
            self.traj = seq_ctx["traj"]
            self.pose_ts = seq_ctx["pose_ts"]
            self.calibs = seq_ctx["calibs"]
            self.calib_ts = seq_ctx["calib_ts"]
            if self._data_source == "aria" and self._loader is not None:
                print(
                    f"Loaded VRS with {self._rgb_num_frames} RGB frames from {self._loader.vrs_path}"
                )
            elif self._data_source == "scannet":
                print(
                    f"Loaded ScanNet scene with {self._rgb_num_frames} RGB frames from {self.scannet_scene}"
                )
            else:
                print(
                    f"Loaded CA1M with {self._rgb_num_frames} RGB frames from {seq_name}"
                )
            print(
                f"Loaded trajectory with {len(self.traj) if self.traj is not None else 0} poses via AriaLoader"
            )
            print(
                f"Loaded online calibration with {len(self.calibs) if self.calibs is not None else 0} entries via AriaLoader"
            )
            # Optional semidense maps from AriaLoader
            self._aria_uid_to_p3 = seq_ctx.get("uid_to_p3", None)
            self._aria_time_to_uids_slaml = seq_ctx.get("time_to_uids_slaml", None)
            self._aria_time_to_uids_slamr = seq_ctx.get("time_to_uids_slamr", None)
            if self._record_fps <= 0.0:
                self._record_fps = _infer_fps_from_timestamps_ns(
                    self._rgb_timestamps,
                    fallback=10.0,
                    source=self._data_source,
                )
                print(f"[REC] Auto-detected RGB FPS: {self._record_fps:.1f}")
            else:
                print(f"[REC] Using user-specified record FPS: {self._record_fps:.1f}")
            _startup_log(
                f"Tracker sequence context applied in {(time_module.perf_counter() - t_ctx0):.2f}s"
            )
        except Exception as e:
            self._vrs_is_nebula = False
            self._rgb_images = None
            self._loader = None
            self._scannet_frame_ids = None
            self.traj = None
            self.pose_ts = np.array([])
            self.calibs = None
            self.calib_ts = np.array([])
            self._aria_uid_to_p3 = None
            self._aria_time_to_uids_slaml = None
            self._aria_time_to_uids_slamr = None
            if self._record_fps <= 0.0:
                self._record_fps = _infer_fps_from_timestamps_ns(
                    np.array(self.sorted_timestamps, dtype=np.int64),
                    fallback=10.0,
                    source=self._data_source,
                )
                print(f"[REC] Auto-detected CSV FPS: {self._record_fps:.1f}")
            else:
                print(f"[REC] Using user-specified record FPS: {self._record_fps:.1f}")
            print(f"Failed to initialize AriaLoader for {seq_name}: {e}")

        # Load 2D bounding box CSV for BB2 comparison panel
        self._bb2d_data: Optional[Dict[int, dict]] = None
        self._bb2d_timestamps: Optional[np.ndarray] = None
        self._bb2d_current_boxes: list[tuple[float, float, float, float, str, int]] = []
        self.show_bb2_panel = True  # toggle for the top BB2 panel (per-frame 2D detections)
        self.show_bb2_csv = True  # toggle CSV BB2s in top panel
        if bb2d_csv_path and os.path.exists(bb2d_csv_path):
            self._bb2d_data = load_bb2d_csv(bb2d_csv_path)
            self._bb2d_timestamps = np.array(sorted(self._bb2d_data.keys()))
            print(
                f"Loaded {len(self._bb2d_data)} frames of 2D BBs from {bb2d_csv_path}"
            )
        else:
            if bb2d_csv_path:
                print(f"No 2D BB CSV found at {bb2d_csv_path}")

        if self.traj is None or self.calibs is None:
            print("Warning: tracker initialized without SST trajectory/calibration")

        # Semidense point cloud state (loaded after GL context init)
        self.show_global_points = False
        self.show_obs_points = False
        self.point_size = 2.0
        self.point_alpha = 0.2
        self._point_positions = None  # (P, 3) numpy array
        self.point_vbo = None
        self.point_vao = None
        self.point_count = 0
        # Observed points per frame (subset rendered larger)
        self._uid_to_idx: dict[int, int] = {}
        self._time_to_uids_slaml: dict[int, list[int]] = {}
        self._time_to_uids_slamr: dict[int, list[int]] = {}
        self._all_obs_timestamps = np.array([])
        self.obs_point_vbo = None
        self.obs_point_vao = None
        self.obs_point_count = 0
        self.obs_point_size = 2.0
        self.obs_point_alpha = 0.7
        self.obs_trail_secs = 0.0
        self.sdp_color_options = [
            (1.0, 1.0, 1.0),  # White
            (0.8, 0.8, 0.8),  # Light Grey
            (0.5, 0.5, 0.5),  # Grey
            (0.25, 0.25, 0.25),  # Dark Grey
            (0.0, 0.0, 0.0),  # Black
        ]
        # Visibility uses N-frame observation tail. 1 => current frame only.
        self.visibility_obs_trail_frames = 1
        self._scannet_debug_points = True

        self.show_trajectory = True
        self.traj_alpha = 1.0
        self.traj_tail_secs = 3.0
        self.show_frustum = True
        self.show_debug_text = False
        self.follow_view = bool(init_follow)
        self.follow_behind = 5.0
        self.follow_above = 1.0
        self.follow_look_ahead = 1.0
        if init_follow_behind is not None:
            self.follow_behind = float(init_follow_behind)
        if init_follow_above is not None:
            self.follow_above = float(init_follow_above)
        if init_follow_look_ahead is not None:
            self.follow_look_ahead = float(init_follow_look_ahead)
        self.camera_damping = 0.95  # 0 = instant snap, 1 = no movement
        self._smooth_eye: Optional[np.ndarray] = None
        self._smooth_target: Optional[np.ndarray] = None
        self._smooth_up: Optional[np.ndarray] = None
        self.frustum_scale = 0.2 if seq_name.startswith("ca1m") else 0.1

        # Current frame stats
        self._current_detection_count = 0
        self._active_track_count = 0
        self._total_track_count = 0
        self._iou_matrix_m = 0
        self._iou_matrix_n = 0

        # Track color mode: 0=Confidence (jet colormap), 1=Boxy (semantic class colors)
        # Keep Health as default to avoid expensive text-model startup on launch.
        self.track_color_mode = 0
        if init_color_mode is not None:
            parsed_track_mode = _track_color_mode_from_name(init_color_mode)
            parsed_fuse_mode = _fuse_color_mode_from_name(init_color_mode)
            if parsed_track_mode is not None:
                self.track_color_mode = parsed_track_mode
            elif parsed_fuse_mode is not None:
                # If a fuse color mode name is provided, keep tracker defaults unless
                # it's explicitly "random", which maps cleanly to tracker mode too.
                if parsed_fuse_mode == COLOR_MODE_RANDOM:
                    self.track_color_mode = 4
            else:
                print(
                    f"Warning: unknown --init_color_mode='{init_color_mode}' "
                    "(track valid: confidence, boxy, boxy_alt, text_pca, random)"
                )
        self._boxy_color_cache: dict[str, np.ndarray] = {}
        self._boxy_ref_data: Optional[dict] = None

        # Text PCA color cache — computed lazily when Text PCA mode is selected.
        self._pca_color_cache: Optional[dict[str, np.ndarray]] = None

        # FPS measurement
        self._fps_last_time = time_module.time()
        self._fps_frame_count = 0
        self._fps_display = 0.0

        # Track parameter snapshot for staleness detection
        self._cached_params_snapshot = self._get_params_snapshot()
        self._params_dirty_time: Optional[float] = None
        self._seek_dirty_time: Optional[float] = None
        self._seek_target_frame: int = 0

        # Get initial OBBs: all frames when frozen (fuse mode), first frame otherwise
        self._all_frames_obbs = None  # cached stacked OBBs for fuse mode
        if self.total_frames > 0:
            if self.freeze_tracker:
                all_obbs_list = []
                for ts in self.sorted_timestamps:
                    all_obbs_list.extend(self.timed_obbs[ts])
                self._all_frames_obbs = torch.stack(all_obbs_list) if all_obbs_list else ObbTW(torch.zeros(0, 165))
                all_obbs_for_init = self._all_frames_obbs
            else:
                all_obbs_for_init = self.timed_obbs[self.sorted_timestamps[0]]
        else:
            all_obbs_for_init = ObbTW(torch.zeros(0, 165))

        # Initialize parent — skip embedding precomputation and initial geometry build
        # since the tracker builds its own geometry per frame in _rebuild_current_view()
        super().__init__(
            all_obbs=all_obbs_for_init,
            root_path=root_path,
            skip_precompute=True,
            seq_name=self._seq_name,
            init_image_panel_width=self._init_image_panel_width,
            seq_ctx=self._prebuilt_seq_ctx,
            **kwargs,
        )

        _startup_log(
            f"TrackerViewer base init complete in {(time_module.perf_counter() - t_init0):.2f}s"
        )

        # Override defaults: semi-transparent detections, thick/opaque tracks
        self.alpha = 0.5
        self.raw_line_width = 2
        self.tracked_all_line_width = 2
        self.visible_line_width = 6
        self.bg_color_index = 0  # White background
        self.sdp_color_index = 3  # Dark Grey points
        self.visual_theme_mode = 0  # Light
        self._apply_visual_theme()

        # Build trajectory line geometry (needs GL context from super().__init__)
        self.traj_instance_vbo = None
        self.traj_instance_vao = None
        self.traj_instance_count = 0
        self._traj_all_segments = None
        self._traj_seg_ts = None
        self.frustum_instance_vbo = None
        self.frustum_instance_vao = None
        self.frustum_instance_count = 0
        self.outline_instance_vbo = None
        self.outline_instance_vao = None
        self.outline_instance_count = 0
        self._build_trajectory_geometry()
        self._load_semidense_points()
        if init_show_obs:
            self.show_obs_points = True

        # Teaser mode: cleaner visuals
        self._teaser = teaser
        if teaser:
            self.show_text_labels = False
            self.show_obs_points = False
            self.raw_line_width = 1
            self.tracked_all_line_width = 10
            self.visible_line_width = 1
            self.point_alpha = 0.5
            self.track_color_mode = 2  # BoxyAlt Color
            self.rgb_obb_thickness = 5.0
            self.rgb_text_scale = 1.4
            self.tracker_min_obs_points = 5
            self.follow_view = False

        # Process first frame through tracker
        if self.total_frames > 0:
            self._step_to_frame(0)

        # Already-tracked mode: load pre-tracked OBBs and freeze tracker
        if already_tracked and self.total_frames > 0:
            tracked_csv = os.path.join(root_path, "tracked_obbs.csv")
            print(f"\n=== Loading pre-tracked OBBs from {tracked_csv} ===")
            tracked_timed = read_obb_csv(tracked_csv)
            # Flatten all tracked OBBs into TrackedInstance objects
            all_tracked = []
            for ts in tracked_timed:
                all_tracked.extend(tracked_timed[ts])
            from utils.track_3d_boxes import TrackedInstance, TrackState

            self.tracker.tracks = []
            for i, obb in enumerate(all_tracked):
                text = obb.reshape(1, -1).text_string()[0]
                track = TrackedInstance(
                    obb=obb,
                    track_id=i,
                    support_count=20,
                    last_seen_frame=self.total_frames - 1,
                    first_seen_frame=0,
                    state=TrackState.ACTIVE,
                    accumulated_weight=20.0,
                    missed_count=0,
                    last_visible=True,
                    cached_text=text,
                )
                self.tracker.tracks.append(track)
            print(f"=== Loaded {len(self.tracker.tracks)} pre-tracked instances ===\n")
            self.freeze_tracker = True
            self._cached_params_snapshot = self._get_params_snapshot()
            self._rebuild_current_view()
        _startup_log(
            f"TrackerViewer init complete in {(time_module.perf_counter() - t_init0):.2f}s"
        )

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        """Return (width, height) of the 3D viewport (right of controls + RGB panels)."""
        w, h = self.wnd.size
        ui_w = self.ui_panel_width  # controls panel width
        if self._rgb_texture is not None and self.show_rgb:
            tex_w, tex_h = self._rgb_tex_size
            tex_aspect = tex_w / tex_h if tex_h > 0 else 1.0
            show_top = self.show_bb2_panel
            is_portrait = tex_aspect <= 1.0
            side_by_side = show_top and is_portrait
            if side_by_side:
                single_pw = self._compute_rgb_panel_width(w, h)
                total_pw = min(2 * single_pw, int(w * self.rgb_panel_max_frac))
                return max(1, w - ui_w - total_pw), h
            elif show_top:
                panel_h = min(h // 2, h - h // 2)
                panel_w = self._compute_rgb_panel_width(w, panel_h)
                return max(1, w - ui_w - panel_w), h
            else:
                panel_w = self._compute_rgb_panel_width(w, h)
                return max(1, w - ui_w - panel_w), h
        return w, h

    def get_camera_matrices(self):
        """Override to support follow-view mode that tracks the Aria camera."""
        if self.total_frames == 0:
            # No detections loaded: fall back to static orbit camera.
            return super().get_camera_matrices()

        if not self.follow_view:
            # Reset smoothing state when leaving follow mode
            self._smooth_eye = None
            self._smooth_target = None
            self._smooth_up = None
            # Use 3D viewport aspect ratio instead of full window
            vw, vh = self._get_3d_viewport_size()
            aspect_ratio = vw / vh
            projection = Matrix44.perspective_projection(45.0, aspect_ratio, 0.1, 100.0)

            azimuth_rad = np.radians(self.camera_azimuth)
            elevation_rad = np.radians(self.camera_elevation)
            camera_x = (
                self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            )
            camera_y = (
                self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            )
            camera_z = self.camera_distance * np.sin(elevation_rad)
            camera_pos = self.camera_target + np.array([camera_x, camera_y, camera_z])
            view = Matrix44.look_at(
                tuple(camera_pos),
                tuple(self.camera_target),
                (0.0, 0.0, 1.0),
            )
            # Match OBBViewer camera convention exactly to avoid split-view offsets.
            mvp = projection * view * Matrix44.identity()
            return projection, view, mvp

        ts = self.sorted_timestamps[self.current_frame_idx]
        cam, T_wr = self._get_cam_and_pose(ts)

        # T_world_camera = T_world_rig @ T_rig_camera
        T_wc = T_wr @ cam.T_camera_rig.inverse()
        cam_pos = T_wc.t.reshape(3).cpu().float().numpy()
        R_wc = T_wc.R.reshape(3, 3).cpu().float().numpy()

        # Place eye behind and above the camera in its local frame
        # "behind" is along the camera's negative-forward axis, "above" is world-Z
        offset_local = np.array([0.0, 0.0, -self.follow_behind])
        offset = R_wc @ offset_local + np.array([0.0, 0.0, self.follow_above])
        eye = cam_pos + offset

        # Look at a point in front of the camera, constrained to camera height.
        # This keeps follow view leveled while still steering with heading.
        forward_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward_xy = np.array(
            [forward_world[0], forward_world[1], 0.0], dtype=np.float32
        )
        forward_xy_norm = np.linalg.norm(forward_xy)
        if forward_xy_norm > 1e-6:
            forward_xy /= forward_xy_norm
        else:
            forward_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        target = cam_pos + forward_xy * float(self.follow_look_ahead)
        target[2] = cam_pos[2]

        up = np.array([0.0, 0.0, 1.0])

        # Smooth eye, target, and up for fluid follow-cam motion
        alpha = 1.0 - self.camera_damping
        if self._smooth_eye is None:
            self._smooth_eye = eye.copy()
            self._smooth_target = target.copy()
            self._smooth_up = up.copy()
        else:
            self._smooth_eye = alpha * eye + (1.0 - alpha) * self._smooth_eye
            self._smooth_target = alpha * target + (1.0 - alpha) * self._smooth_target
            self._smooth_up = alpha * up + (1.0 - alpha) * self._smooth_up
            # Re-normalize up to avoid drift
            up_norm = np.linalg.norm(self._smooth_up)
            if up_norm > 1e-6:
                self._smooth_up /= up_norm

        vw, vh = self._get_3d_viewport_size()
        aspect = vw / vh
        projection = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        view = Matrix44.look_at(
            tuple(self._smooth_eye),
            tuple(self._smooth_target),
            tuple(self._smooth_up),
        )
        mvp = projection * view * Matrix44.identity()
        return projection, view, mvp

    def _snap_orbit_from_follow(self):
        """Snap orbit camera params to match the current follow-view position."""
        if self._smooth_eye is None or self._smooth_target is None:
            return

        eye = self._smooth_eye
        target = self._smooth_target

        # Convert eye/target to orbit camera params
        self.camera_target = target.copy()
        diff = eye - target
        self.camera_distance = float(np.linalg.norm(diff))
        if self.camera_distance > 1e-6:
            self.camera_elevation = float(
                np.degrees(np.arcsin(diff[2] / self.camera_distance))
            )
            self.camera_azimuth = float(
                np.degrees(np.arctan2(diff[1], diff[0]))
            )

    def _get_params_snapshot(self) -> tuple:
        """Return a tuple of current tracker params for staleness comparison."""
        return (
            self.tracker_iou_threshold,
            self.tracker_min_hits,
            self.raw_conf_threshold,
            self.tracker_merge_iou,
            self.tracker_merge_sem,
            self.tracker_merge_iou_2d,
            self.tracker_merge_interval,
            self.tracker_min_conf_mass,
            self.tracker_min_obs_points,
        )

    def _filter_frame_obbs(self, obbs: ObbTW) -> ObbTW:
        """Apply per-frame 3DBB confidence threshold."""
        if len(obbs) == 0:
            return obbs
        mask = (obbs.prob >= float(self.raw_conf_threshold)).reshape(-1)
        return obbs[mask]

    def _load_semidense_points(self) -> None:
        """Load semidense points and upload as a static white point cloud VBO."""
        if getattr(self, "_data_source", None) == "scannet":
            # ScanNet path: we currently render per-frame observed depth points only
            # (fast startup), not a pre-accumulated global cloud.
            self.show_global_points = False
            self.show_obs_points = False
            self.point_count = 0
            self._point_positions = None
            print("ScanNet: using per-frame observed depth points (no global cloud)")
            return

        if getattr(self, "_data_source", None) == "ca1m":
            ca_loader = getattr(self, "_ca_loader", None)
            positions = (
                _extract_ca_sdp_positions(ca_loader) if ca_loader is not None else None
            )
            if positions is None or len(positions) == 0:
                print("No semidense points found in CALoader.sdp_ws")
                return

            self._point_positions = positions
            self.point_count = len(positions)
            self.show_global_points = True
            self.show_obs_points = False

            colors = np.full((self.point_count, 3), 0.25, dtype=np.float32)
            vertex_data = np.hstack([positions, colors]).astype(np.float32)
            self.point_vbo = self.ctx.buffer(vertex_data.tobytes())
            self.point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self.point_vbo, "3f 3f", "in_position", "in_color")],
            )
            print(f"Loaded {self.point_count} semidense points from CALoader.sdp_ws")
            return

        # Prefer semidense from AriaLoader context only.
        time_to_uids_slaml = getattr(self, "_aria_time_to_uids_slaml", None)
        time_to_uids_slamr = getattr(self, "_aria_time_to_uids_slamr", None)
        uid_to_p3 = getattr(self, "_aria_uid_to_p3", None)

        if (
            time_to_uids_slaml is None
            or time_to_uids_slamr is None
            or uid_to_p3 is None
        ):
            print("No semidense points in loader context; skipping global point cloud")
            return

        if not uid_to_p3:
            print("No semidense points found")
            return

        # Store observation maps for per-frame highlighting
        self._time_to_uids_slaml = dict(time_to_uids_slaml)
        self._time_to_uids_slamr = dict(time_to_uids_slamr)
        self._all_obs_timestamps = np.array(
            sorted(
                set(self._time_to_uids_slaml.keys())
                | set(self._time_to_uids_slamr.keys())
            )
        )

        # Extract (P, 3) positions and build uid -> index map
        uids = list(uid_to_p3.keys())
        P = len(uids)
        positions = np.empty((P, 3), dtype=np.float32)

        for i, uid in enumerate(uids):
            px, py, pz, _inv_dist_std, _dist_std = uid_to_p3[uid]
            positions[i] = (px, py, pz)
            self._uid_to_idx[uid] = i

        # Grey color for semi-dense points (default: dark grey, index 3)
        colors = np.full((P, 3), 0.25, dtype=np.float32)

        # Interleave position and color: [x, y, z, r, g, b] per point
        vertex_data = np.hstack([positions, colors]).astype(np.float32)

        self._point_positions = positions
        self.point_count = P
        self.show_global_points = True
        self.show_obs_points = False

        # Upload to GPU
        self.point_vbo = self.ctx.buffer(vertex_data.tobytes())
        self.point_vao = self.ctx.vertex_array(
            self.point_prog,
            [(self.point_vbo, "3f 3f", "in_position", "in_color")],
        )

        print(f"Loaded {P} semidense points as point cloud")

    def _rebuild_point_vbo_color(self) -> None:
        """Rebuild the global point cloud VBO with the current sdp_color_index."""
        if self._point_positions is None:
            return
        color_val = self.sdp_color_options[self.sdp_color_index]
        P = len(self._point_positions)
        colors = np.full((P, 3), 0, dtype=np.float32)
        colors[:, 0] = color_val[0]
        colors[:, 1] = color_val[1]
        colors[:, 2] = color_val[2]
        vertex_data = np.hstack([self._point_positions, colors]).astype(np.float32)
        self.point_vbo.write(vertex_data.tobytes())

        # Precompute sorted observation timestamps for nearest-neighbor fallback
        self._all_obs_timestamps = np.array(
            sorted(
                set(self._time_to_uids_slaml.keys())
                | set(self._time_to_uids_slamr.keys())
            )
        )

    def _get_observed_points(self, ts_ns: int) -> Optional[torch.Tensor]:
        """Return (K, 3) world-space positions of currently observed semidense points."""
        if getattr(self, "_data_source", None) == "scannet":
            loader = getattr(self, "_scannet_loader", None)
            if loader is None or len(getattr(loader, "frame_ids", [])) == 0:
                if getattr(self, "_scannet_debug_points", True):
                    print("[ScanNet SDP] loader missing or empty frame_ids")
                return None
            idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
            idx = max(0, min(idx, len(loader.frame_ids) - 1))
            prev_idx = int(getattr(loader, "index", 0))
            try:
                loader.index = idx
                datum = next(loader)
                sdp_w = datum.get("sdp_w", None)
                if sdp_w is None or sdp_w.numel() == 0:
                    if getattr(self, "_scannet_debug_points", True):
                        print(
                            f"[ScanNet SDP] ts={ts_ns} idx={idx} frame_id={loader.frame_ids[idx]}: sdp_w empty"
                        )
                    return None
                pts = sdp_w.reshape(-1, 3).float()
                valid = torch.isfinite(pts).all(dim=1)
                nonzero = pts.abs().sum(dim=1) > 1e-6
                pts = pts[valid & nonzero]
                if getattr(self, "_scannet_debug_points", True):
                    print(
                        f"[ScanNet SDP] ts={ts_ns} idx={idx} frame_id={loader.frame_ids[idx]}: "
                        f"raw={sdp_w.reshape(-1, 3).shape[0]} valid={pts.shape[0]}"
                    )
                return pts if len(pts) > 0 else None
            except Exception as e:
                if getattr(self, "_scannet_debug_points", True):
                    print(
                        f"[ScanNet SDP] ts={ts_ns} idx={idx}: exception loading sdp_w: {e}"
                    )
                return None
            finally:
                loader.index = prev_idx

        if self._point_positions is None:
            return None

        # Gather UIDs observed at this timestamp from both SLAM cameras
        observed_uids: set[int] = set()
        if ts_ns in self._time_to_uids_slaml:
            observed_uids.update(self._time_to_uids_slaml[ts_ns])
        if ts_ns in self._time_to_uids_slamr:
            observed_uids.update(self._time_to_uids_slamr[ts_ns])

        # Also check nearby timestamps (observations may not align exactly)
        # Use the nearest observation timestamp within 1ms
        if not observed_uids and len(self._all_obs_timestamps) > 0:
            nearest_idx = find_nearest2(self._all_obs_timestamps, ts_ns)
            nearest_ts = int(self._all_obs_timestamps[nearest_idx])
            if abs(nearest_ts - ts_ns) < 1_000_000:  # within 1ms
                if nearest_ts in self._time_to_uids_slaml:
                    observed_uids.update(self._time_to_uids_slaml[nearest_ts])
                if nearest_ts in self._time_to_uids_slamr:
                    observed_uids.update(self._time_to_uids_slamr[nearest_ts])

        if not observed_uids:
            return None

        indices = [
            self._uid_to_idx[uid] for uid in observed_uids if uid in self._uid_to_idx
        ]
        if not indices:
            return None

        return torch.from_numpy(self._point_positions[indices])  # (K, 3)

    def _get_observed_points_trail(
        self, ts_ns: int, trail_duration_ns: int = 200_000_000
    ) -> Optional[torch.Tensor]:
        """Return (K, 3) world-space positions observed within a trailing time window."""
        if getattr(self, "_data_source", None) == "scannet":
            # ScanNet path currently serves per-frame sampled depth points.
            return self._get_observed_points(ts_ns)

        if self._point_positions is None or len(self._all_obs_timestamps) == 0:
            return None

        # Find observation timestamps within [ts_ns - trail_duration_ns, ts_ns]
        t_start = ts_ns - trail_duration_ns
        idx_lo = bisect_left(self._all_obs_timestamps, t_start)
        idx_hi = bisect_right(self._all_obs_timestamps, ts_ns)

        if idx_lo >= idx_hi:
            # No observations in window — fall back to nearest single timestamp
            return self._get_observed_points(ts_ns)

        observed_uids: set[int] = set()
        for i in range(idx_lo, idx_hi):
            obs_ts = int(self._all_obs_timestamps[i])
            if obs_ts in self._time_to_uids_slaml:
                observed_uids.update(self._time_to_uids_slaml[obs_ts])
            if obs_ts in self._time_to_uids_slamr:
                observed_uids.update(self._time_to_uids_slamr[obs_ts])

        if not observed_uids:
            return None

        indices = [
            self._uid_to_idx[uid] for uid in observed_uids if uid in self._uid_to_idx
        ]
        if not indices:
            return None

        return torch.from_numpy(self._point_positions[indices])

    def _update_observed_points(self, ts_ns: int) -> None:
        """Rebuild the observed-points VBO for the current frame's timestamp."""
        obs_pts = self._get_observed_points_trail(
            ts_ns, trail_duration_ns=int(self.obs_trail_secs * 1e9)
        )
        if obs_pts is None:
            self.obs_point_count = 0
            if getattr(self, "_data_source", None) == "scannet" and getattr(
                self, "_scannet_debug_points", True
            ):
                print(f"[ScanNet SDP] ts={ts_ns}: no observed points after trail query")
            return

        obs_positions = obs_pts.numpy()
        K = len(obs_positions)
        if getattr(self, "_data_source", None) == "scannet" and getattr(
            self, "_scannet_debug_points", True
        ):
            print(f"[ScanNet SDP] ts={ts_ns}: uploading observed points K={K}")
        # Use same color as global points
        if hasattr(self, "sdp_color_options") and hasattr(self, "sdp_color_index"):
            color_val = self.sdp_color_options[self.sdp_color_index]
        else:
            # Early init fallback (before tracker UI/color state is fully initialized)
            color_val = (0.25, 0.25, 0.25)
        obs_colors = np.tile(np.array(color_val, dtype=np.float32), (K, 1))
        obs_data = np.hstack([obs_positions, obs_colors]).astype(np.float32)

        self.obs_point_count = K

        # Reuse or create buffer
        data_bytes = obs_data.tobytes()
        if self.obs_point_vbo is not None and self.obs_point_vbo.size >= len(
            data_bytes
        ):
            self.obs_point_vbo.orphan(self.obs_point_vbo.size)
            self.obs_point_vbo.write(data_bytes)
        else:
            if self.obs_point_vbo is not None:
                self.obs_point_vbo.release()
            if self.obs_point_vao is not None:
                self.obs_point_vao.release()
            self.obs_point_vbo = self.ctx.buffer(data_bytes)
            self.obs_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self.obs_point_vbo, "3f 3f", "in_position", "in_color")],
            )

    def _step_to_frame(self, target_idx: int) -> None:
        """Advance tracker to the target frame index.

        For sequential forward steps (target == current + 1), runs one tracker
        update. For jumps (forward or backward), resets the tracker and starts
        fresh from the target frame — no intermediate frames are computed.
        """
        if self.total_frames == 0:
            self.current_frame_idx = 0
            self.is_playing = False
            return

        if target_idx < 0:
            target_idx = 0
        if target_idx >= self.total_frames:
            target_idx = self.total_frames - 1

        if self.freeze_tracker:
            # Skip tracker updates — just move the frame pointer
            pass
        elif target_idx == self.current_frame_idx + 1:
            ts = self.sorted_timestamps[target_idx]
            detections = self._filter_frame_obbs(
                self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
            )
            cam, T_wr = self._get_cam_and_pose(ts)
            obs_pts = self._get_observed_points_trail(
                ts, trail_duration_ns=int(self.obs_trail_secs * 1e9)
            )
            self.tracker.update(
                detections,
                target_idx,
                cam=cam,
                T_world_rig=T_wr,
                observed_points=obs_pts,
            )
        elif target_idx != self.current_frame_idx:
            # Jump: reset tracker and start fresh from target frame
            self._reset_tracker()
            ts = self.sorted_timestamps[target_idx]
            detections = self._filter_frame_obbs(
                self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
            )
            cam, T_wr = self._get_cam_and_pose(ts)
            obs_pts = self._get_observed_points_trail(
                ts, trail_duration_ns=int(self.obs_trail_secs * 1e9)
            )
            self.tracker.update(
                detections,
                target_idx,
                cam=cam,
                T_world_rig=T_wr,
                observed_points=obs_pts,
            )

        self.current_frame_idx = target_idx
        self._rebuild_current_view()

        # Load and upload RGB image for current timestamp
        ts = self.sorted_timestamps[target_idx]
        rgb = self._load_rgb_for_timestamp(ts)
        if rgb is not None:
            self._upload_rgb_texture(rgb)

    def _reset_tracker(self) -> None:
        """Recreate tracker with current parameters."""
        if self.freeze_tracker:
            return  # Don't reset when using pre-loaded tracks
        self.tracker = BoundingBox3DTracker(
            iou_threshold=self.tracker_iou_threshold,
            min_hits=self.tracker_min_hits,
            conf_threshold=self.tracker_conf_threshold,
            force_cpu=self.tracker_force_cpu,
            merge_iou_threshold=self.tracker_merge_iou,
            merge_semantic_threshold=self.tracker_merge_sem,
            merge_iou_2d_threshold=self.tracker_merge_iou_2d,
            merge_interval=self.tracker_merge_interval,
            min_confidence_mass=self.tracker_min_conf_mass,
            min_obs_points=self.tracker_min_obs_points,
            verbose=self.tracker_verbose,
        )
        self.current_frame_idx = 0
        self._cached_params_snapshot = self._get_params_snapshot()

    @staticmethod
    def _init_boxy_ref_data() -> dict:
        """Load text embedder and precompute reference embeddings."""
        from owl.clip_tokenizer import TextEmbedder
        from utils.taxonomy import (
            BOXY_SEM2NAME,
            SSI_COLORS_ALT,
            TEXT2COLORS,
        )

        model = TextEmbedder()
        sem_id_list = [sid for sid in BOXY_SEM2NAME if sid >= 0]
        name_list = [BOXY_SEM2NAME[sid] for sid in sem_id_list]
        boxy_embs = model.forward(name_list)

        text_color_names = list(TEXT2COLORS.keys())
        text_color_values = list(TEXT2COLORS.values())
        text_color_embs = model.forward(text_color_names)

        return {
            "model": model,
            "sem_id_list": sem_id_list,
            "boxy_embs": boxy_embs,
            "text_color_names": text_color_names,
            "text_color_values": text_color_values,
            "text_color_embs": text_color_embs,
            "boxy_sem2name": BOXY_SEM2NAME,
            "ssi_colors_alt": SSI_COLORS_ALT,
        }

    def _ensure_boxy_ref_data(self) -> dict:
        """Lazily initialize heavy text-model reference data."""
        if self._boxy_ref_data is None:
            self._boxy_ref_data = self._init_boxy_ref_data()
        return self._boxy_ref_data

    _LABEL_OVERRIDES: dict[str, str] = {
        "wall poster": "Wall Art",
        "curtain": "Window",
    }

    def _remap_label(self, label: str) -> str:
        """Apply display-name overrides for known mislabeled classes."""
        return self._LABEL_OVERRIDES.get(label.lower(), label)

    def _get_boxy_color(self, text: str, use_alt: bool = False) -> np.ndarray:
        """Get Boxy semantic color for a text label (cached).

        Matches text to the closest BOXY_SEM2NAME class via cosine similarity
        of sentence embeddings, then looks up the color via TEXT2COLORS,
        or SSI_COLORS_ALT.
        """
        cache_key = ("alt:" + text) if use_alt else text
        if cache_key in self._boxy_color_cache:
            return self._boxy_color_cache[cache_key]

        # Hard overrides: map known labels directly to a Boxy class (sem_id)
        # before falling back to embedding-based matching.
        _TEXT_TO_SEM_ID = {
            "desk": 9,  # Table
            "table": 9,  # Table
            "coffee table": 9,  # Table
        }

        ref = self._ensure_boxy_ref_data()

        # Check hard override first
        override_sem_id = _TEXT_TO_SEM_ID.get(text.lower())
        if override_sem_id is not None:
            matched_sem_id = override_sem_id
        else:
            # Embedding-based matching
            text_emb = ref["model"].forward([text])  # (1, D)
            dist = torch.mm(ref["boxy_embs"], text_emb.t()).squeeze()
            closest = int(dist.argmax().item())
            closest_dist = float(dist[closest].item())

            if closest_dist > 0.5:
                matched_sem_id = ref["sem_id_list"][closest]
            else:
                matched_sem_id = 32  # "Anything"

        # Map matched class to color
        matched_name = ref["boxy_sem2name"][matched_sem_id]
        if use_alt:
            alt_colors = ref["ssi_colors_alt"]
            color = np.array(
                alt_colors.get(matched_name, alt_colors["Anything"]), dtype=np.float32
            )
        else:
            name_emb = ref["model"].forward([matched_name])
            color_dist = torch.mm(ref["text_color_embs"], name_emb.t()).squeeze()
            color_closest = int(color_dist.argmax().item())
            color = np.array(ref["text_color_values"][color_closest], dtype=np.float32)

        self._boxy_color_cache[cache_key] = color
        return color

    def _precompute_pca_colors(self) -> dict[str, np.ndarray]:
        """Precompute PCA-based colors from all text labels across all frames.

        Returns dict mapping label string -> (3,) float32 RGB array.
        """
        import colorsys

        # Collect all unique labels from all frames
        all_labels = set()
        for obbs in self.timed_obbs.values():
            texts = obbs.text_string()
            if isinstance(texts, str):
                texts = [texts]
            all_labels.update(texts)
        unique_labels = sorted(all_labels)
        print(f"Precomputing PCA colors for {len(unique_labels)} unique labels")

        cache: dict[str, np.ndarray] = {}
        if len(unique_labels) <= 1:
            for lbl in unique_labels:
                cache[lbl] = np.array(
                    colorsys.hsv_to_rgb(0.55, 0.75, 0.65), dtype=np.float32
                )
            return cache

        model = self._boxy_ref_data["model"]
        embs = model.forward(unique_labels)  # (N, 384)
        U, S, V = torch.pca_lowrank(embs, q=3)
        proj = torch.mm(embs - embs.mean(dim=0, keepdim=True), V)  # (N, 3)
        p_min = proj.min(dim=0).values
        p_max = proj.max(dim=0).values
        p_range = (p_max - p_min).clamp(min=1e-8)
        proj_norm = (proj - p_min) / p_range  # (N, 3) in [0, 1]

        for i, lbl in enumerate(unique_labels):
            h = float(proj_norm[i, 0])
            s = 0.6 + 0.3 * float(proj_norm[i, 1])
            v = 0.5 + 0.3 * float(proj_norm[i, 2])
            cache[lbl] = np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)
        return cache

    def _get_pca_colors(self, active_tracks: list) -> np.ndarray:
        """Look up precomputed PCA colors for active tracks.

        Returns (M, 3) float32 RGB array.
        """
        if self._pca_color_cache is None:
            self._pca_color_cache = self._precompute_pca_colors()
        fallback = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        colors = np.array(
            [
                self._pca_color_cache.get(track.cached_text, fallback)
                for track in active_tracks
            ],
            dtype=np.float32,
        )
        return colors

    def _rebuild_current_view(self):
        """Rebuild GPU geometry for current frame detections + tracked instances.

        Uses a fast path that builds line geometry directly from OBB corners
        without running semantic embeddings or color maps, so this can run
        at interactive rates during playback. Caches results for instant replay.
        """
        t_start = time_module.perf_counter()

        # Get current frame detections
        ts = self.sorted_timestamps[self.current_frame_idx]
        nav_ts = self._get_navigation_timestamp(self.current_frame_idx, ts)
        current_detections = self._filter_frame_obbs(
            self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
        )
        tracked_all_obbs = self._empty_obbs_like(current_detections)
        tracked_visible_obbs = self._empty_obbs_like(current_detections)

        # Look up CSV 2D bounding boxes for current timestamp
        self._bb2d_current_boxes = []
        self._bb2d_img_wh: tuple[int, int] = (0, 0)
        if self._bb2d_data is not None and self._bb2d_timestamps is not None:
            idx = int(find_nearest2(self._bb2d_timestamps, ts))
            nearest_ts = int(self._bb2d_timestamps[idx])
            # Only use if within 50ms tolerance
            if abs(nearest_ts - ts) < 50_000_000:
                entry = self._bb2d_data[nearest_ts]
                bb2d = entry["bb2d"]  # (N, 4) x1,y1,x2,y2
                labels = entry["labels"]
                sem_ids = entry.get("sem_ids", [-1] * len(labels))
                self._bb2d_img_wh = (int(entry["img_width"]), int(entry["img_height"]))
                orig_h = self._bb2d_img_wh[1]
                # Gen1 VRS image is rotated 90° CW for display —
                # swap displayed dims to match rotated image.
                if not self._vrs_is_nebula:
                    self._bb2d_img_wh = (
                        int(entry["img_height"]),
                        int(entry["img_width"]),
                    )
                for j in range(len(bb2d)):
                    x1 = float(bb2d[j, 0])
                    y1 = float(bb2d[j, 1])
                    x2 = float(bb2d[j, 2])
                    y2 = float(bb2d[j, 3])
                    # Apply same 90° CW rotation to BB2 coordinates.
                    if not self._vrs_is_nebula:
                        rx1 = orig_h - 1 - y2
                        ry1 = x1
                        rx2 = orig_h - 1 - y1
                        ry2 = x2
                        x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                    self._bb2d_current_boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            self._remap_label(labels[j])[:12],
                            int(sem_ids[j]),
                        )
                    )

        # Get camera and pose for frustum rendering
        cam, T_wr = self._get_cam_and_pose(nav_ts)

        # Rebuild RGB projection caches for this frame (track mode path).
        self._rgb_projected_lines = []
        self._rgb_projected_raw_lines = []
        self._rgb_projected_tracked_all_lines = []
        self._rgb_projected_tracked_visible_lines = []
        self._rgb_projected_labels = []
        self._rgb_projected_tracked_all_labels = []
        self._rgb_projected_tracked_visible_labels = []

        # Raw detections projected into RGB (gray), independent of tracked state.
        if len(current_detections) > 0:
            raw_colors = np.tile(
                np.array([[0.6, 0.6, 0.6]], dtype=np.float32),
                (len(current_detections), 1),
            )
            raw_lines, _ = self._project_obbs_for_rgb(
                current_detections,
                cam,
                T_wr,
                raw_colors,
            )
            self._rgb_projected_raw_lines = raw_lines

        # Update stats
        active_tracks = self.tracker._get_active_tracks()
        conf_thresh = float(self.tracker_conf_threshold)
        shown_tracks = []
        shown_track_indices = []
        for idx, t in enumerate(active_tracks):
            avg_conf = t.accumulated_weight / max(t.support_count, 1)
            if avg_conf >= conf_thresh:
                shown_tracks.append(t)
                shown_track_indices.append(idx)

        # Recompute visibility for all active tracks using current camera/pose.
        # This ensures visibility is fresh even when the tracker is frozen.
        if active_tracks:
            stacked_obbs = torch.stack([t.obb for t in active_tracks])
            tracked_all_obbs = stacked_obbs
            rgb_point_mask = torch.ones(len(stacked_obbs), dtype=torch.bool)
            _, bb2_valid = stacked_obbs.get_pseudo_bb2(
                cam.unsqueeze(0),
                T_wr.unsqueeze(0),
                num_samples_per_edge=10,
                valid_ratio=0.16667,
            )
            is_visible = bb2_valid.squeeze(0)  # (K,) bool
            tracked_visible_obbs = self._subset_obbs(stacked_obbs, is_visible)

            # Semidense points visibility check.
            # N=1 means strict current-frame only; N>1 uses a short trailing window.
            vis_frames = max(1, int(getattr(self, "visibility_obs_trail_frames", 1)))
            has_obs_maps = (
                len(self._time_to_uids_slaml) > 0 or len(self._time_to_uids_slamr) > 0
            )
            if vis_frames <= 1:
                obs_pts = self._get_observed_points(ts)
            else:
                trail_ns = int((vis_frames - 1) * self._median_frame_delta_ns)
                obs_pts = self._get_observed_points_trail(
                    ts, trail_duration_ns=trail_ns
                )
            if obs_pts is not None:
                K = len(stacked_obbs)
                N_pts = obs_pts.shape[0]
                pts_expanded = obs_pts.unsqueeze(0).expand(K, N_pts, 3)
                inside = stacked_obbs.batch_points_inside_bb3(pts_expanded)
                points_inside_count = inside.sum(dim=1)
                has_enough_points = points_inside_count >= self.tracker_min_obs_points
                rgb_point_mask = has_enough_points
            else:
                # If no semidense maps are available (e.g. unsupported dataset),
                # fall back to projection-only visibility.
                if not has_obs_maps:
                    has_enough_points = torch.ones(len(stacked_obbs), dtype=torch.bool)
                    rgb_point_mask = has_enough_points
                else:
                    # With semidense available, no observed points means not visible.
                    has_enough_points = torch.zeros(len(stacked_obbs), dtype=torch.bool)
                    rgb_point_mask = has_enough_points

            for k, track in enumerate(active_tracks):
                projected_visible = is_visible[k].item()
                contains_points = has_enough_points[k].item()
                track.last_visible = projected_visible and contains_points
                if self.debug_visibility:
                    pts_count = (
                        int(points_inside_count[k].item()) if obs_pts is not None else 0
                    )
                    print(
                        f"  [VIS] track={track.track_id} "
                        f"proj={projected_visible} pts={pts_count} "
                        f"visible={track.last_visible} "
                        f"label={track.cached_text[:15]}"
                    )

        self._current_detection_count = len(current_detections)
        self._active_track_count = len(active_tracks)
        self._total_track_count = len(self.tracker.get_all_tracks())
        self._iou_matrix_m, self._iou_matrix_n = self.tracker.last_iou_matrix_size
        self._set_frame_obb_sets(
            raw=current_detections,
            tracked_all=(
                torch.stack([t.obb for t in shown_tracks])
                if shown_tracks
                else self._empty_obbs_like(current_detections)
            ),
            tracked_visible=(
                self._subset_obbs(
                    torch.stack([t.obb for t in shown_tracks]),
                    torch.tensor(
                        [t.last_visible for t in shown_tracks], dtype=torch.bool
                    ),
                )
                if shown_tracks
                else self._empty_obbs_like(current_detections)
            ),
        )

        edge_indices = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long)  # (12, 2)

        # --- Build raw detection geometry (fast path) ---
        # In fuse mode (freeze_tracker), show ALL raw detections across all frames
        # instead of just the current frame's detections.
        if self.freeze_tracker and self._all_frames_obbs is not None:
            raw_obbs_for_render = self._all_frames_obbs
        else:
            raw_obbs_for_render = current_detections

        t0 = time_module.perf_counter()
        det_data = None
        if len(raw_obbs_for_render) > 0:
            N = len(raw_obbs_for_render)
            corners = raw_obbs_for_render.bb3corners_world  # (N, 8, 3)
            probs = raw_obbs_for_render.prob.squeeze(-1)  # (N,)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            # Color raw detections by semantic class
            colors = self._obbs_random_colors(raw_obbs_for_render).cpu()

            batch_idx = torch.arange(N)[:, None].expand(N, 12)
            start_pts = corners[batch_idx, edge_indices[:, 0][None, :].expand(N, 12)]
            end_pts = corners[batch_idx, edge_indices[:, 1][None, :].expand(N, 12)]
            colors_exp = colors[:, None, :].expand(N, 12, 3)
            probs_exp = probs[:, None].expand(N, 12)

            det_data = (
                torch.cat(
                    [
                        start_pts,
                        end_pts,
                        colors_exp,
                        probs_exp.unsqueeze(-1),
                    ],
                    dim=2,
                )
                .reshape(-1, 10)
                .numpy()
                .astype("f4")
            )

        # --- Build tracked instance geometry (fast path) ---
        t_det = time_module.perf_counter()
        track_data = None
        text_labels = []
        label_positions = []
        label_colors = []
        track_snapshots = []  # (support_count, missed_count, is_visible) for label rendering

        if shown_tracks:
            track_obbs = torch.stack([t.obb for t in shown_tracks])
            M = len(track_obbs)
            t_corners = track_obbs.bb3corners_world  # (M, 8, 3)

            # Use cached visibility and text from tracker (avoids redundant
            # get_pseudo_bb2 + batch_points_inside_bb3 + text_string calls)
            t_vis0 = time_module.perf_counter()

            # Batch box centers: single .cpu().numpy() transfer
            all_centers = t_corners.mean(dim=1).cpu().numpy()  # (M, 3)

            # Collect per-track metadata from cached tracker state
            health_scores = []
            for idx, track in enumerate(shown_tracks):
                label = self._remap_label(track.cached_text)
                text_labels.append(label[:20])
                label_positions.append(all_centers[idx])

                track_snapshots.append(
                    (
                        track.support_count,
                        track.missed_count,
                        track.last_visible,
                        track.accumulated_weight,
                    )
                )

                # Health-based score: new→0.5 (green), well-tracked→1.0 (red), dying→0.0 (blue)
                score = max(
                    0.0,
                    min(
                        1.0,
                        0.5
                        + 0.5 * track.support_count / 20.0
                        - 0.5 * track.missed_count / 90.0,
                    ),
                )
                health_scores.append(score)

            # Compute track colors based on selected color mode
            t_meta = time_module.perf_counter()
            if self.track_color_mode in (1, 2):  # Boxy Color or BoxyAlt
                use_alt = self.track_color_mode == 2
                boxy_colors = np.array(
                    [
                        self._get_boxy_color(
                            self._remap_label(track.cached_text),
                            use_alt=use_alt,
                        )
                        for track in shown_tracks
                    ],
                    dtype=np.float32,
                )
                t_colors = torch.from_numpy(boxy_colors)
            elif self.track_color_mode == 3:  # Text PCA
                t_colors = torch.from_numpy(
                    self._get_pca_colors(shown_tracks).astype(np.float32)
                )
            elif self.track_color_mode == 4:  # Random (theme-aware, class-consistent)
                shown_track_obbs = torch.stack([t.obb for t in shown_tracks])
                t_colors = self._obbs_random_colors(shown_track_obbs).cpu()
            else:  # Health (default)
                health_colors_rgba = cm.jet(np.array(health_scores))[
                    :, :3
                ]  # (M, 3)
                t_colors = torch.from_numpy(health_colors_rgba.astype(np.float32))

            label_colors = [t_colors[i].numpy() for i in range(M)]

            batch_idx = torch.arange(M)[:, None].expand(M, 12)
            start_pts = t_corners[batch_idx, edge_indices[:, 0][None, :].expand(M, 12)]
            end_pts = t_corners[batch_idx, edge_indices[:, 1][None, :].expand(M, 12)]
            colors_exp = t_colors[:, None, :].expand(M, 12, 3)
            avg_probs = torch.tensor(
                [t.accumulated_weight / max(t.support_count, 1) for t in shown_tracks],
                dtype=torch.float32,
            ).clamp(0.3, 1.0)
            track_probs = avg_probs[:, None, None].expand(M, 12, 1)

            track_data = (
                torch.cat(
                    [
                        start_pts,
                        end_pts,
                        colors_exp,
                        track_probs,
                    ],
                    dim=2,
                )
                .reshape(-1, 10)
                .numpy()
                .astype("f4")
            )

            # Build thicker colored outline for visible tracks only
            visible_mask = torch.tensor(
                [s[2] for s in track_snapshots], dtype=torch.bool
            )
            if shown_track_indices:
                shown_track_idx_t = torch.tensor(shown_track_indices, dtype=torch.long)
                rgb_point_mask_shown = rgb_point_mask[shown_track_idx_t]
            else:
                rgb_point_mask_shown = torch.zeros(M, dtype=torch.bool)
            if visible_mask.any():
                vis_start = start_pts[visible_mask]
                vis_end = end_pts[visible_mask]
                V = vis_start.shape[0]
                vis_colors = t_colors[visible_mask][:, None, :].expand(V, 12, 3)
                vis_ones = torch.ones(V, 12, 1)
                outline_data = (
                    torch.cat([vis_start, vis_end, vis_colors, vis_ones], dim=2)
                    .reshape(-1, 10)
                    .numpy()
                    .astype("f4")
                )
            else:
                outline_data = None

            # Cache projected 2D sub-segments for RGB overlay.
            # Subdivide each of the 12 edges into N_SUB segments so that
            # partially-visible edges (clipped by image boundary) still render.
            N_SUB = 10
            MAX_RGB_BOXES = int(max(16, self.rgb_max_projected_boxes))
            # Choose which tracks to project to RGB: all or visible-only
            # Pre-filter by semidense point support before doing expensive projection.
            if self.show_rgb_visible_only:
                rgb_mask = visible_mask & rgb_point_mask_shown
            else:
                rgb_mask = rgb_point_mask_shown
            if rgb_mask.any():
                total_rgb_candidates = int(rgb_mask.sum().item())
                if total_rgb_candidates > MAX_RGB_BOXES:
                    print(
                        f"[RGB] Capping projected tracks: {total_rgb_candidates} -> {MAX_RGB_BOXES}"
                    )
                rgb_corners = t_corners[rgb_mask][:MAX_RGB_BOXES]  # (V, 8, 3)
                rgb_colors_np = t_colors[rgb_mask][:MAX_RGB_BOXES].numpy()
                rgb_label_indices = rgb_mask.nonzero(as_tuple=False).squeeze(1)[
                    :MAX_RGB_BOXES
                ]
                T_world_cam = T_wr @ cam.T_camera_rig.inverse()
                V_rgb = rgb_corners.shape[0]

                # Interpolate N_SUB+1 points along each of the 12 edges
                edge_idx = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long)  # (12,2)
                p0 = rgb_corners[:, edge_idx[:, 0], :]  # (V, 12, 3)
                p1 = rgb_corners[:, edge_idx[:, 1], :]  # (V, 12, 3)
                t_interp = torch.linspace(0, 1, N_SUB + 1)  # (S,)
                # (V, 12, S, 3) = lerp between edge endpoints
                edge_pts = (
                    p0[:, :, None, :] * (1 - t_interp[None, None, :, None])
                    + p1[:, :, None, :] * t_interp[None, None, :, None]
                )
                S = N_SUB + 1
                pts_world = edge_pts.reshape(-1, 3)  # (V*12*S, 3)
                pts_cam = T_world_cam.inverse().transform(pts_world)
                # Scale camera to match VRS resolution (see _project_obbs_for_rgb)
                proj_cam = cam
                if self._rgb_vrs_w > 0 and self._rgb_vrs_h > 0:
                    cam_w = cam.size[..., 0].item()
                    cam_h = cam.size[..., 1].item()
                    if (
                        abs(cam_w - self._rgb_vrs_w) > 1
                        or abs(cam_h - self._rgb_vrs_h) > 1
                    ):
                        proj_cam = cam.scale_to_size((self._rgb_vrs_w, self._rgb_vrs_h))
                pts_2d, valid = proj_cam.project(
                    pts_cam.unsqueeze(0),
                    fov_deg=140.0 if self._vrs_is_nebula else 120.0,
                )
                pts_2d = pts_2d.squeeze(0).cpu().numpy()  # (V*12*S, 2)
                valid = valid.squeeze(0).cpu().numpy()  # (V*12*S,)
                # Un-rotate for Gen1
                if not self._vrs_is_nebula:
                    old_x = pts_2d[:, 0].copy()
                    old_y = pts_2d[:, 1].copy()
                    pts_2d[:, 0] = self._rgb_vrs_h - 1 - old_y
                    pts_2d[:, 1] = old_x
                pts_2d = pts_2d.reshape(V_rgb, 12, S, 2)
                valid = valid.reshape(V_rgb, 12, S)
                for i in range(V_rgb):
                    line_item = (pts_2d[i], valid[i], rgb_colors_np[i])
                    self._rgb_projected_lines.append(line_item)
                    self._rgb_projected_tracked_all_lines.append(line_item)
                    is_vis_i = bool(visible_mask[rgb_label_indices[i]].item())
                    if self.show_rgb_visible_only or bool(is_vis_i):
                        self._rgb_projected_tracked_visible_lines.append(line_item)
                    # Label position from all valid sub-points
                    v = valid[i]  # (12, S)
                    if v.any():
                        all_valid_pts = pts_2d[i][v]  # (K, 2)
                        label_x = float(all_valid_pts[:, 0].mean())
                        label_y = float(all_valid_pts[:, 1].mean())
                        label_idx = int(rgb_label_indices[i].item())
                        label_item = (
                            label_x,
                            label_y,
                            text_labels[label_idx],
                            rgb_colors_np[i],
                        )
                        self._rgb_projected_tracked_all_labels.append(label_item)
                        if is_vis_i:
                            self._rgb_projected_tracked_visible_labels.append(
                                label_item
                            )
        else:
            outline_data = None

        # Upload to GPU
        t_geom = time_module.perf_counter()
        self._upload_frame_data(
            det_data,
            track_data,
            outline_data,
            text_labels,
            label_positions,
            track_snapshots,
            label_colors=label_colors,
        )
        t_upload = time_module.perf_counter()

        # Build frustum geometry for current frame
        self._build_frustum_geometry(cam, T_wr)

        # Update trajectory tail for this frame
        if self._traj_all_segments is not None:
            self._update_trajectory_tail(nav_ts)

        # Update observed semidense points for this frame.
        # ScanNet currently uses per-frame observed points only (no global cloud),
        # so point_count can remain 0 while observed points are still available.
        if self.point_count > 0 or getattr(self, "_data_source", None) == "scannet":
            self._update_observed_points(nav_ts)

        t_end = time_module.perf_counter()

        M = len(shown_tracks) if shown_tracks else 0
        if shown_tracks:
            print(
                f"[BENCH] rebuild_view (frame {self.current_frame_idx}, "
                f"{len(current_detections)} dets, {M} tracks): "
                f"det_geom={(t_det - t0) * 1000:.1f}ms  "
                f"metadata={(t_meta - t_vis0) * 1000:.1f}ms  "
                f"track_geom={(t_geom - t_meta) * 1000:.1f}ms  "
                f"gpu_upload={(t_upload - t_geom) * 1000:.1f}ms  "
                f"frustum={(t_end - t_upload) * 1000:.1f}ms  "
                f"TOTAL={(t_end - t_start) * 1000:.1f}ms"
            )
        else:
            print(
                f"[BENCH] rebuild_view (frame {self.current_frame_idx}, "
                f"{len(current_detections)} dets, 0 tracks): "
                f"det_geom={(t_det - t0) * 1000:.1f}ms  "
                f"gpu_upload={(t_upload - t_geom) * 1000:.1f}ms  "
                f"TOTAL={(t_end - t_start) * 1000:.1f}ms"
            )

    def _update_or_create_buffer(
        self,
        vbo: Any,
        vao: Any,
        data_bytes: bytes,
    ) -> tuple[Any, Any]:
        """Reuse GPU buffer if data fits, otherwise allocate a new one.

        Uses vbo.orphan() + vbo.write() to avoid GPU object churn when the
        new data fits in the existing buffer.
        """
        if vbo is not None and vbo.size >= len(data_bytes):
            vbo.orphan(vbo.size)
            vbo.write(data_bytes)
            return vbo, vao
        # Release old and create new
        if vbo is not None:
            vbo.release()
        if vao is not None:
            vao.release()
        vbo = self.ctx.buffer(data_bytes)
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
        return vbo, vao

    def _upload_frame_data(
        self,
        det_data: Optional[np.ndarray],
        track_data: Optional[np.ndarray],
        outline_data: Optional[np.ndarray],
        text_labels: list[str],
        label_positions: list[np.ndarray],
        track_snapshots: list[tuple[int, int, bool, float]],
        label_colors: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Upload precomputed geometry data to GPU buffers."""
        # --- Raw detections ---
        if det_data is not None:
            self.cached_instance_count = len(det_data)
            data_bytes = det_data.tobytes()
            self.cached_instance_vbo, self.cached_instance_vao = (
                self._update_or_create_buffer(
                    self.cached_instance_vbo,
                    self.cached_instance_vao,
                    data_bytes,
                )
            )
        else:
            self.cached_instance_count = 0

        # --- Tracked instances ---
        self.tracked_all_text_labels = text_labels
        self.tracked_all_label_positions = label_positions
        self.tracked_all_label_colors = label_colors if label_colors is not None else []
        self._track_snapshots = track_snapshots

        if track_data is not None:
            self.tracked_all_instance_count = len(track_data)
            data_bytes = track_data.tobytes()
            self.tracked_all_instance_vbo, self.tracked_all_instance_vao = (
                self._update_or_create_buffer(
                    self.tracked_all_instance_vbo,
                    self.tracked_all_instance_vao,
                    data_bytes,
                )
            )
            # Keep a lightweight placeholder list for len() checks in shared code.
            self.tracked_all_instances = [True] * len(track_snapshots)
        else:
            self.tracked_all_instance_count = 0
            self.tracked_all_instances = []

        # --- Outline for visible tracks ---
        if outline_data is not None:
            self.outline_instance_count = len(outline_data)
            data_bytes = outline_data.tobytes()
            self.outline_instance_vbo, self.outline_instance_vao = (
                self._update_or_create_buffer(
                    self.outline_instance_vbo,
                    self.outline_instance_vao,
                    data_bytes,
                )
            )
        else:
            self.outline_instance_count = 0

    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events. Space = play/pause, arrow keys = step."""
        # Intercept ESC to pause instead of quitting
        if key == self.wnd.keys.ESCAPE:
            if action == self.wnd.keys.ACTION_PRESS:
                self.is_playing = False
                if self.follow_view:
                    self._snap_orbit_from_follow()
                    self.follow_view = False
            return

        # Only act on key press (not release)
        if action == self.wnd.keys.ACTION_PRESS and key in (
            self.wnd.keys.SPACE,
            self.wnd.keys.RIGHT,
            self.wnd.keys.LEFT,
        ):
            if self.total_frames == 0:
                return

            if key == self.wnd.keys.SPACE:
                if self.current_frame_idx >= self.total_frames - 1:
                    # At end of playback — reset to beginning and play
                    self._reset_tracker()
                    self._step_to_frame(0)
                    self.is_playing = True
                else:
                    self.is_playing = not self.is_playing
                # Sync follow view with play state
                if self.is_playing:
                    self._smooth_eye = None
                    self._smooth_target = None
                    self._smooth_up = None
                    self.follow_view = True
                else:
                    self._snap_orbit_from_follow()
                    self.follow_view = False
                self._last_step_time = time_module.time()
            elif key == self.wnd.keys.RIGHT:
                self.is_playing = False
                if self.follow_view:
                    self._snap_orbit_from_follow()
                    self.follow_view = False
                self._step_forward()
            elif key == self.wnd.keys.LEFT:
                self.is_playing = False
                if self.follow_view:
                    self._snap_orbit_from_follow()
                    self.follow_view = False
                if self.current_frame_idx > 0:
                    self._step_to_frame(self.current_frame_idx - 1)
            return

        # Delegate all non-playback keys to parent handling
        super().on_key_event(key, action, modifiers)

    def render_3d(self, time: float, frame_time: float) -> None:
        """Render with auto-advance when playing."""
        # Nothing to render frame-wise when there are no detections.
        if self.total_frames == 0:
            self.is_playing = False
            return super().render_3d(time, frame_time)

        # Autorecord: start recording and playing on first render
        if self._autorecord and not self._recording and self.current_frame_idx == 0:
            self._start_recording()
            self.is_playing = True
            self._last_step_time = time_module.time()
            print("[REC] Autorecord armed: will stop + write video at sequence end")
        # Update FPS counter
        self._fps_frame_count += 1
        now = time_module.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_time = now

        # Deferred auto-apply: wait 0.3s after last param change before replaying
        if self._params_dirty_time is not None:
            if now - self._params_dirty_time >= 0.3:
                self._params_dirty_time = None
                was_playing = self.is_playing
                self.is_playing = False
                target = self.current_frame_idx
                self._reset_tracker()
                self._step_to_frame(target)
                self.is_playing = was_playing

        # Deferred frame seek: wait 0.3s after last slider drag before stepping
        if self._seek_dirty_time is not None:
            if now - self._seek_dirty_time >= 0.3:
                target = self._seek_target_frame
                self._seek_dirty_time = None
                self._step_to_frame(target)

        # Auto-advance when playing
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            current_time = time_module.time()
            if current_time - self._last_step_time >= 1.0 / self.playback_fps:
                self._last_step_time = current_time
                self._step_forward()

        # Auto-stop at end
        if self.is_playing and self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            if self.follow_view:
                self._snap_orbit_from_follow()
                self.follow_view = False
            if self._autorecord and self._recording:
                print(
                    "[REC] Reached final frame; stopping recording and encoding video..."
                )
                self._stop_recording()
                self.wnd.close()

        # Delegate to parent for main render pass (raw + tracked + RGB split)
        super().render_3d(time, frame_time)

        # Reuse parent MVP/viewport for additional tracker overlays
        mvp = self._last_render_mvp
        if mvp is None:
            _projection, _view, mvp = self.get_camera_matrices()
            mvp = np.asarray(mvp, dtype=np.float32)
        self._frame_mvp = mvp
        full_w, _ = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)
        mvp_bytes = mvp.astype("f4").tobytes()

        # Render semidense point cloud (all points white)
        if self.show_global_points and self.point_vao is not None:
            self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
            self.point_prog["mvp"].write(mvp_bytes)
            self.point_prog["point_size"].write(
                np.array(self.point_size, dtype="f4").tobytes()
            )
            self.point_prog["alpha"].write(
                np.array(self.point_alpha, dtype="f4").tobytes()
            )
            self.point_vao.render(mode=self.ctx.POINTS, vertices=self.point_count)

        # Render currently observed points larger (cyan)
        if (
            self.show_obs_points
            and self.obs_point_count > 0
            and self.obs_point_vao is not None
        ):
            self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
            self.point_prog["mvp"].write(mvp_bytes)
            self.point_prog["point_size"].write(
                np.array(self.obs_point_size, dtype="f4").tobytes()
            )
            self.point_prog["alpha"].write(
                np.array(self.obs_point_alpha, dtype="f4").tobytes()
            )
            self.obs_point_vao.render(
                mode=self.ctx.POINTS, vertices=self.obs_point_count
            )

        # Restore full viewport for imgui rendering
        full_w, full_h = self.wnd.size
        self.ctx.viewport = (0, 0, full_w, full_h)
        self.ctx.scissor = None

    def _render_text_labels(self) -> None:
        """Override to show track ID and support count on labels."""
        if not self.show_text_labels or not self.tracked_all_label_positions:
            return

        # Reuse cached MVP from render_3d if available, otherwise compute
        mvp = getattr(self, "_frame_mvp", None)
        if mvp is None:
            _projection, _view, mvp = self.get_camera_matrices()
        full_w, _ = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w  # viewport x offset (panels on left)
        mvp_T = np.asarray(mvp, dtype=np.float32).T

        # Use cached track snapshots (works for both live and cached frames)
        track_snapshots = getattr(self, "_track_snapshots", [])

        # Batch-project all label positions to screen coords
        M = len(self.tracked_all_label_positions)
        positions = np.ones((M, 4), dtype=np.float32)
        for i, pos_3d in enumerate(self.tracked_all_label_positions):
            positions[i, :3] = pos_3d
        clip = (mvp_T @ positions.T).T  # (M, 4)
        behind = (clip[:, 3] <= 0) | (clip[:, 2] < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc = clip[:, :3] / clip[:, 3:4]
        outside = (np.abs(ndc[:, 0]) > 1) | (np.abs(ndc[:, 1]) > 1)
        screen_x = (ndc[:, 0] * 0.5 + 0.5) * w + vp_x
        screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
        visible = ~behind & ~outside

        text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        draw_list = imgui.get_foreground_draw_list()

        for i in range(M):
            if not visible[i]:
                continue

            sx = screen_x[i]
            if sx < vp_x + self.ui_panel_width:
                continue

            # Skip labels that overlap the RGB panel
            sy = screen_y[i]
            if self._rgb_panel_rect is not None:
                rx, ry, rw, rh = self._rgb_panel_rect
                if rx <= sx <= rx + rw and ry <= sy <= ry + rh:
                    continue

            text = (
                self.tracked_all_text_labels[i]
                if i < len(self.tracked_all_text_labels)
                else "?"
            )

            # Add track info if debug text is enabled
            if self.show_debug_text and i < len(track_snapshots):
                support_count, missed_count, _is_visible, acc_weight = track_snapshots[
                    i
                ]
                avg_p = acc_weight / max(support_count, 1)
                display_text = (
                    f"{text} {avg_p:.1f} [n={support_count}, m={missed_count}]"
                )
            else:
                display_text = text

            text_x = sx + 10
            text_y = screen_y[i] - 10

            # Background color: darkened OBB color (matches 2D BB style)
            if i < len(self.tracked_all_label_colors):
                c = self.tracked_all_label_colors[i]
                bg_col = imgui.get_color_u32_rgba(
                    float(c[0]) * 0.4, float(c[1]) * 0.4, float(c[2]) * 0.4, 0.6
                )
            else:
                br, bg_, bb, ba = self.overlay_text_bg_rgba
                bg_col = imgui.get_color_u32_rgba(br, bg_, bb, ba)

            tw, th = imgui.calc_text_size(display_text)
            draw_list.add_rect_filled(
                text_x - 2, text_y - 1, text_x + tw + 2, text_y + th + 1, bg_col
            )
            draw_list.add_text(text_x, text_y, text_col, display_text)

    def on_render(self, time: float, frame_time: float) -> None:
        """Override to capture frames for recording after compositing."""
        super().on_render(time, frame_time)
        if self._recording and self._record_dir is not None:
            if self.current_frame_idx == self._record_last_playback_frame:
                return
            self._record_last_playback_frame = self.current_frame_idx
            t0 = time_module.time()
            w, h = self.wnd.size
            x0 = 450  # skip UI panel
            capture_w = w - x0
            if capture_w > 0:
                data = self.ctx.screen.read(
                    viewport=(x0, 0, capture_w, h), components=3
                )
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, capture_w, 3)
                img = np.flipud(img)  # OpenGL origin is bottom-left
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (capture_w // 2, h // 2))
                path = os.path.join(
                    self._record_dir, f"image_{self._record_frame_idx:05d}.png"
                )
                cv2.imwrite(path, img)
                self._record_frame_idx += 1
            # Exclude recording time from FPS measurement
            self._fps_last_time += time_module.time() - t0

    def _start_recording(self) -> None:
        self._record_dir = tempfile.mkdtemp(prefix="viz_tracker_rec_")
        self._record_frame_idx = 0
        self._record_last_playback_frame = -1
        self._recording = True
        print(f"[REC] Recording started — frames -> {self._record_dir}")

    def _stop_recording(self) -> None:
        self._recording = False
        if self._record_dir is None:
            return
        n = self._record_frame_idx
        print(f"[REC] Recording stopped — {n} frames captured")
        if n > 0:
            output_dir = os.path.expanduser("~/Desktop")
            base_name = f"viz_tracker_{self._seq_name}"
            output_name = f"{base_name}.mp4"
            out_path = os.path.join(output_dir, output_name)
            suffix = 1
            while os.path.exists(out_path):
                output_name = f"{base_name}_{suffix}.mp4"
                out_path = os.path.join(output_dir, output_name)
                suffix += 1
            mp4_path = make_mp4(
                self._record_dir,
                framerate=int(self._record_fps),
                output_dir=output_dir,
                output_name=output_name,
                crf=14,
                preset="slow",
            )
            print(f"[REC] Video saved to {mp4_path}")
        self._record_dir = None
        self._record_frame_idx = 0

    def render_ui(self) -> None:
        """Render tracker-specific UI panel."""
        # Render text labels for tracked boxes
        if self.show_tracked_all_set:
            self._render_text_labels()

        w, h = self.wnd.size
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(self.ui_panel_width, h, imgui.ALWAYS)

        imgui.begin(
            "Tracker Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        )

        # === Playback Controls (kept identical to fusion) ===
        self._section_header("Playback")
        if imgui.button(
            "Play" if not self.is_playing else "Pause", width=90, height=28
        ):
            self.is_playing = not self.is_playing
            if self.is_playing:
                self._smooth_eye = None
                self._smooth_target = None
                self._smooth_up = None
                self.follow_view = True
            else:
                self._snap_orbit_from_follow()
                self.follow_view = False
            self._last_step_time = time_module.time()
        imgui.same_line()
        if imgui.button("<", width=30, height=28):
            self.is_playing = False
            if self.follow_view:
                self._snap_orbit_from_follow()
                self.follow_view = False
            if self.current_frame_idx > 0:
                self._step_to_frame(self.current_frame_idx - 1)
        imgui.same_line()
        if imgui.button(">", width=30, height=28):
            self.is_playing = False
            if self.follow_view:
                self._snap_orbit_from_follow()
                self.follow_view = False
            self._step_forward()
        imgui.same_line()
        if imgui.button("Reset", width=90, height=28):
            self.is_playing = False
            if self.follow_view:
                self._snap_orbit_from_follow()
                self.follow_view = False
            self._seek_dirty_time = None
            self._params_dirty_time = None
            self._reset_tracker()
            self._step_to_frame(0)

        imgui.push_item_width(300)
        changed, new_frame = imgui.slider_int(
            "Frame",
            self._seek_target_frame
            if self._seek_dirty_time is not None
            else self.current_frame_idx,
            0,
            max(0, self.total_frames - 1),
        )
        if changed:
            self.is_playing = False
            if self.follow_view:
                self._snap_orbit_from_follow()
                self.follow_view = False
            self._seek_target_frame = new_frame
            self._seek_dirty_time = time_module.time()
        imgui.pop_item_width()

        imgui.push_item_width(200)
        _changed, self.playback_fps = imgui.slider_float(
            "Playback FPS", self.playback_fps, 0.5, 60.0
        )
        imgui.pop_item_width()
        if not self._recording:
            if imgui.button("Record", width=90, height=28):
                self._start_recording()
        else:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.1, 0.1, 1.0)
            if imgui.button("Stop Rec", width=90, height=28):
                self._stop_recording()
            imgui.pop_style_color()
        imgui.same_line()
        if imgui.button("Focus on Scene"):
            self._focus_on_scene()

        _changed, self.freeze_tracker = imgui.checkbox(
            "Freeze Tracker", self.freeze_tracker
        )

        # === Tracker Parameters ===
        self._section_header("Tracker")

        imgui.push_item_width(200)

        _changed, self.tracker_iou_threshold = imgui.slider_float(
            "IoU Threshold", self.tracker_iou_threshold, 0.0, 1.0
        )
        _changed, self.tracker_min_hits = imgui.slider_int(
            "Min Hits", self.tracker_min_hits, 1, 10
        )
        tracked_show_changed, self.tracker_conf_threshold = imgui.slider_float(
            "Tracked 3DBB Conf", self.tracker_conf_threshold, 0.0, 1.0
        )
        raw_det_changed, self.raw_conf_threshold = imgui.slider_float(
            "Per-Frame 3DBB Conf", self.raw_conf_threshold, 0.0, 1.0
        )
        if tracked_show_changed:
            self._rebuild_current_view()
        if raw_det_changed:
            self.prob_threshold = self.raw_conf_threshold
            self._params_dirty_time = time_module.time()

        if imgui.tree_node("Advanced Tracker"):
            _changed, self.tracker_merge_iou = imgui.slider_float(
                "Merge IoU", self.tracker_merge_iou, 0.0, 1.0
            )
            _changed, self.tracker_merge_sem = imgui.slider_float(
                "Merge Sem", self.tracker_merge_sem, 0.0, 1.0
            )
            _changed, self.tracker_merge_iou_2d = imgui.slider_float(
                "Merge IoU 2D", self.tracker_merge_iou_2d, 0.0, 1.0
            )
            _changed, self.tracker_merge_interval = imgui.slider_int(
                "Merge Interval", self.tracker_merge_interval, 1, 50
            )
            _changed, self.tracker_min_conf_mass = imgui.slider_float(
                "Min Conf Mass", self.tracker_min_conf_mass, 0.5, 20.0
            )
            _changed, self.tracker_min_obs_points = imgui.slider_int(
                "Min Obs Points", self.tracker_min_obs_points, 1, 20
            )
            imgui.tree_pop()

        imgui.pop_item_width()

        # Mark params dirty when changed (deferred apply in render_3d)
        if self._get_params_snapshot() != self._cached_params_snapshot:
            self._params_dirty_time = time_module.time()

        # === Visualization Controls ===
        self._section_header("Visualization")
        self._render_common_visual_controls(
            tracked_all_checkbox_label="Tracked All",
            tracked_all_line_label="Track Line Width",
            show_visible_line_width=True,
        )

        # Track color mode dropdown
        imgui.push_item_width(200)
        color_mode_names = [
            "Confidence",
            "Boxy Color",
            "BoxyAlt Color",
            "Text PCA",
            "Random",
        ]
        changed, self.track_color_mode = imgui.combo(
            "Track Color", self.track_color_mode, color_mode_names
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text("Confidence mode uses jet colormap:")
            imgui.text("low confidence = purple/blue")
            imgui.text("high confidence = green/yellow")
            imgui.end_tooltip()
        if changed:
            if self.track_color_mode in (1, 2):
                self._ensure_boxy_ref_data()
            elif self.track_color_mode == 3:
                if self._pca_color_cache is None:
                    self._pca_color_cache = self._precompute_pca_colors()
            self._rebuild_current_view()
        imgui.pop_item_width()

        _changed, self.show_trajectory = imgui.checkbox(
            "Show Trajectory", self.show_trajectory
        )
        _changed, self.show_frustum = imgui.checkbox("Show Frustum", self.show_frustum)
        _changed, self.show_bb2_panel = imgui.checkbox(
            "Show 2DBB Panel", self.show_bb2_panel
        )
        if getattr(self, "_rgb_images", None) is not None:
            _changed, self.show_rgb_labels = imgui.checkbox(
                "Show RGB Labels", self.show_rgb_labels
            )

        if imgui.tree_node("Advanced Visuals"):
            imgui.push_item_width(200)
            _changed, self.traj_alpha = imgui.slider_float(
                "Traj Alpha", self.traj_alpha, 0.0, 1.0
            )
            changed, self.traj_tail_secs = imgui.slider_float(
                "Traj Tail (s)", self.traj_tail_secs, 0.5, 30.0
            )
            if changed and self._traj_all_segments is not None and self.total_frames > 0:
                ts = self.sorted_timestamps[self.current_frame_idx]
                self._update_trajectory_tail(ts)
            changed, self.frustum_scale = imgui.slider_float(
                "Frustum Scale", self.frustum_scale, 0.001, 0.5
            )
            if changed and self.total_frames > 0:
                ts = self.sorted_timestamps[self.current_frame_idx]
                cam, T_wr = self._get_cam_and_pose(ts)
                self._build_frustum_geometry(cam, T_wr)
            if getattr(self, "_rgb_images", None) is not None:
                _changed, self.rgb_obb_thickness = imgui.slider_float(
                    "RGB 3DBB Width", self.rgb_obb_thickness, 1.0, 10.0
                )
                _changed, self.rgb_text_scale = imgui.slider_float(
                    "RGB Text Scale", self.rgb_text_scale, 0.5, 5.0
                )
            imgui.pop_item_width()
            imgui.tree_pop()

        # === Points ===
        if self.point_count > 0:
            self._section_header("Points")
            _changed, self.show_global_points = imgui.checkbox(
                "Show Global Points", self.show_global_points
            )
            if self.show_global_points:
                imgui.push_item_width(200)
                _changed, self.point_size = imgui.slider_float(
                    "Point Size", self.point_size, 1.0, 10.0
                )
                _changed, self.point_alpha = imgui.slider_float(
                    "Point Alpha", self.point_alpha, 0.0, 1.0
                )
                imgui.pop_item_width()
            _changed, self.show_obs_points = imgui.checkbox(
                "Show Observed Points", self.show_obs_points
            )
            if self.show_obs_points:
                imgui.push_item_width(200)
                _changed, self.obs_point_size = imgui.slider_float(
                    "Obs Point Size", self.obs_point_size, 1.0, 20.0
                )
                _changed, self.obs_point_alpha = imgui.slider_float(
                    "Obs Point Alpha", self.obs_point_alpha, 0.0, 1.0
                )
                _changed, self.obs_trail_secs = imgui.slider_float(
                    "Obs Tail", self.obs_trail_secs, 0.0, 5.0
                )
                imgui.pop_item_width()

        self._section_header("Camera")
        imgui.text("Follow: ON" if self.follow_view else "Follow: OFF")
        imgui.push_item_width(200)
        _changed, self.camera_damping = imgui.slider_float(
            "Camera Damping", self.camera_damping, 0.0, 0.99
        )
        _changed, self.follow_behind = imgui.slider_float(
            "Behind Distance", self.follow_behind, 0.0, 10.0
        )
        _changed, self.follow_above = imgui.slider_float(
            "Above Distance", self.follow_above, 0.0, 30.0
        )
        _changed, self.follow_look_ahead = imgui.slider_float(
            "Look Ahead (m)", self.follow_look_ahead, 0.0, 10.0
        )
        imgui.pop_item_width()

        imgui.end()

        # === Statistics overlay (bottom-left of 3D viewport) ===
        if not self._teaser and not self._recording:
            stats_x = 455
            stats_y = h - 130
            imgui.set_next_window_position(stats_x, stats_y, imgui.ONCE)
            imgui.set_next_window_bg_alpha(0.4)
            imgui.begin(
                "Stats",
                flags=imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_ALWAYS_AUTO_RESIZE
                | imgui.WINDOW_NO_FOCUS_ON_APPEARING
                | imgui.WINDOW_NO_NAV,
            )
            imgui.text(f"FPS: {self._fps_display:.1f}")
            imgui.text(f"Frame: {self.current_frame_idx + 1} / {self.total_frames}")
            imgui.text(
                f"Detections: {self._current_detection_count}  Active: {self._active_track_count}  Total: {self._total_track_count}"
            )
            imgui.text(
                f"IoU: {self._iou_matrix_m}x{self._iou_matrix_n}"
                f" ({self._iou_matrix_m * self._iou_matrix_n} pairs)"
            )
            imgui.end()

        # RGB image panel (right side of screen)
        if self.show_rgb and self._rgb_texture is not None:
            win_w, win_h = self.wnd.size
            tex_w, tex_h = self._rgb_tex_size
            if tex_w <= 0 or tex_h <= 0:
                self._rgb_panel_rect = None
                return
            tex_aspect = tex_w / tex_h if tex_h > 0 else 1.0

            # Determine panel layout: stacked or side-by-side for portrait images
            show_top = self.show_bb2_panel
            is_portrait = tex_aspect <= 1.0
            side_by_side = show_top and is_portrait

            ui_w = self.ui_panel_width  # controls panel width
            if side_by_side:
                # Portrait/square: two panels side by side, each full height
                single_pw = self._compute_rgb_panel_width(win_w, win_h)
                total_panel_w = min(2 * single_pw, int(win_w * self.rgb_panel_max_frac))
                left_panel_w = total_panel_w // 2
                right_panel_w = total_panel_w - left_panel_w
                panel_w = total_panel_w
                panel_x = ui_w
                bottom_y = 0
                bottom_h = win_h
            elif show_top:
                top_h = win_h // 2
                bottom_h = win_h - top_h
                bottom_y = top_h
                panel_w = self._compute_rgb_panel_width(win_w, min(top_h, bottom_h))
                panel_x = ui_w
            else:
                top_h = 0
                bottom_h = win_h
                bottom_y = 0
                panel_w = self._compute_rgb_panel_width(win_w, bottom_h)
                panel_x = ui_w

            # --- Top panel: BB2 comparison (collapsible) ---
            if show_top:
                if side_by_side:
                    imgui.set_next_window_position(panel_x, 0, imgui.ALWAYS)
                    imgui.set_next_window_size(left_panel_w, win_h, imgui.ALWAYS)
                else:
                    imgui.set_next_window_position(panel_x, 0, imgui.ALWAYS)
                    imgui.set_next_window_size(panel_w, top_h, imgui.ALWAYS)
                imgui.set_next_window_collapsed(False, imgui.ONCE)
                expanded, _ = imgui.begin(
                    "Per-Frame Detections",
                    flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
                )
                if expanded:
                    avail_w, avail_h = imgui.get_content_region_available()
                    img_scale = min(avail_w / tex_w, avail_h / tex_h)
                    draw_w = tex_w * img_scale
                    draw_h = tex_h * img_scale
                    imgui.image(self._rgb_texture.glo, draw_w, draw_h)
                    img_min = imgui.get_item_rect_min()
                    scale_x = draw_w / tex_w * self._rgb_img_scale
                    scale_y = draw_h / tex_h * self._rgb_img_scale
                    draw_list = imgui.get_window_draw_list()

                    # Draw raw per-frame 3DBB projections
                    if self.show_rgb_obbs and self._rgb_projected_raw_lines:
                        for edge_pts, edge_valid, color in self._rgb_projected_raw_lines:
                            col = imgui.get_color_u32_rgba(
                                float(color[0]), float(color[1]), float(color[2]), 1.0
                            )
                            for e in range(edge_pts.shape[0]):
                                for s in range(edge_pts.shape[1] - 1):
                                    if edge_valid[e, s] and edge_valid[e, s + 1]:
                                        x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                        y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                        x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                        y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                        draw_list.add_line(
                                            x0, y0, x1, y1, col, self.rgb_obb_thickness
                                        )

                    # Draw CSV BB2s (green, or per-class random color)
                    if self.show_bb2_csv and self._bb2d_current_boxes:
                        # CSV BB2 coords are in detector image space — scale
                        # directly from CSV resolution to widget size
                        csv_w, csv_h = self._bb2d_img_wh
                        if csv_w > 0 and csv_h > 0:
                            csv_sx = draw_w / csv_w
                            csv_sy = draw_h / csv_h
                        else:
                            csv_sx = scale_x
                            csv_sy = scale_y
                        use_random_col = self.track_color_mode == 4
                        default_col = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0)
                        text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
                        default_bg = imgui.get_color_u32_rgba(0.0, 0.4, 0.0, 0.6)
                        label_scale = max(0.5, float(self.rgb_text_scale))
                        imgui.set_window_font_scale(label_scale)
                        for x1, y1, x2, y2, label, sem_id in self._bb2d_current_boxes:
                            if use_random_col:
                                if sem_id >= 0 and sem_id in BOXY_SEM2NAME:
                                    key = f"sem:{sem_id}:{BOXY_SEM2NAME[sem_id]}"
                                else:
                                    key = f"lbl:{label.strip().lower()}"
                                r, g, b = self._label_to_theme_random_color(key)
                                csv_col = imgui.get_color_u32_rgba(r, g, b, 1.0)
                                bg_col = imgui.get_color_u32_rgba(
                                    r * 0.4, g * 0.4, b * 0.4, 0.6
                                )
                            else:
                                csv_col = default_col
                                bg_col = default_bg
                            rx0 = img_min.x + x1 * csv_sx
                            ry0 = img_min.y + y1 * csv_sy
                            rx1 = img_min.x + x2 * csv_sx
                            ry1 = img_min.y + y2 * csv_sy
                            draw_list.add_rect(
                                rx0,
                                ry0,
                                rx1,
                                ry1,
                                csv_col,
                                0.0,
                                0,
                                self.rgb_bb2_thickness,
                            )
                            # Label above box
                            tw, th = imgui.calc_text_size(label)
                            draw_list.add_rect_filled(
                                rx0 - 1, ry0 - th - 2, rx0 + tw + 2, ry0, bg_col
                            )
                            draw_list.add_text(rx0, ry0 - th - 1, text_col, label)
                        imgui.set_window_font_scale(1.0)

                # Get actual window height (title bar only when collapsed)
                _, top_actual_h = imgui.get_window_size()
                imgui.end()
                if not side_by_side:
                    bottom_y = int(top_actual_h)
                    bottom_h = win_h - bottom_y

            # --- Bottom panel: Tracked 3DBBs ---
            if side_by_side:
                imgui.set_next_window_position(panel_x + left_panel_w, 0, imgui.ALWAYS)
                imgui.set_next_window_size(right_panel_w, win_h, imgui.ALWAYS)
            else:
                imgui.set_next_window_position(panel_x, bottom_y, imgui.ALWAYS)
                imgui.set_next_window_size(panel_w, bottom_h, imgui.ALWAYS)
            imgui.begin(
                "Tracked 3DBBs",
                flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
            )
            # Draw image filling the panel (preserve aspect ratio)
            avail_w, avail_h = imgui.get_content_region_available()
            img_scale = min(avail_w / tex_w, avail_h / tex_h)
            draw_w = tex_w * img_scale
            draw_h = tex_h * img_scale
            imgui.image(self._rgb_texture.glo, draw_w, draw_h)
            # Compute image widget position and VRS-to-widget scale for overlays
            img_min = imgui.get_item_rect_min()
            scale_x = draw_w / tex_w * self._rgb_img_scale
            scale_y = draw_h / tex_h * self._rgb_img_scale
            # Draw projected tracked OBB wireframes on top of the RGB image
            if (
                self.show_rgb_obbs
                and self.show_rgb_tracked_all
                and self._rgb_projected_tracked_all_lines
            ):
                draw_list = imgui.get_window_draw_list()
                for (
                    edge_pts,
                    edge_valid,
                    color,
                ) in self._rgb_projected_tracked_all_lines:
                    # edge_pts: (12, S, 2), edge_valid: (12, S)
                    col = imgui.get_color_u32_rgba(
                        float(color[0]), float(color[1]), float(color[2]), 1.0
                    )
                    for e in range(edge_pts.shape[0]):
                        for s in range(edge_pts.shape[1] - 1):
                            if edge_valid[e, s] and edge_valid[e, s + 1]:
                                x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                draw_list.add_line(
                                    x0, y0, x1, y1, col, self.rgb_obb_thickness
                                )
            if (
                self.show_rgb_obbs
                and self.show_rgb_tracked_visible
                and self._rgb_projected_tracked_visible_lines
            ):
                draw_list = imgui.get_window_draw_list()
                for (
                    edge_pts,
                    edge_valid,
                    color,
                ) in self._rgb_projected_tracked_visible_lines:
                    col = imgui.get_color_u32_rgba(
                        float(color[0]), float(color[1]), float(color[2]), 1.0
                    )
                    for e in range(edge_pts.shape[0]):
                        for s in range(edge_pts.shape[1] - 1):
                            if edge_valid[e, s] and edge_valid[e, s + 1]:
                                x0 = img_min.x + edge_pts[e, s, 0] * scale_x
                                y0 = img_min.y + edge_pts[e, s, 1] * scale_y
                                x1 = img_min.x + edge_pts[e, s + 1, 0] * scale_x
                                y1 = img_min.y + edge_pts[e, s + 1, 1] * scale_y
                                draw_list.add_line(
                                    x0, y0, x1, y1, col, self.rgb_obb_thickness + 1.0
                                )
            if self.show_rgb_tracked_visible:
                self._rgb_projected_labels = list(
                    self._rgb_projected_tracked_visible_labels
                )
            elif self.show_rgb_tracked_all:
                self._rgb_projected_labels = list(
                    self._rgb_projected_tracked_all_labels
                )
            else:
                self._rgb_projected_labels = []
            # Draw text labels above projected OBBs
            if self.show_rgb_labels and self._rgb_projected_labels:
                draw_list = imgui.get_window_draw_list()
                self._draw_projected_labels(
                    draw_list,
                    self._rgb_projected_labels,
                    img_min,
                    scale_x,
                    scale_y,
                )
            # Store panel rect so text labels can avoid overlapping
            px, py = imgui.get_window_position()
            pw, ph = imgui.get_window_size()
            self._rgb_panel_rect = (px, py, pw, ph)
            imgui.end()
        else:
            self._rgb_panel_rect = None
