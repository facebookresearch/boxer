#! /usr/bin/env python3

# pyre-unsafe
"""Shared utilities for view_fusion.py and view_tracker.py."""

import argparse
import os
import re
import sys

import numpy as np
import torch
from utils.demo_utils import EVAL_PATH
from utils.demo_utils import DEFAULT_SEQ, expand_seq_shorthand, handle_input
from utils.file_io import read_obb_csv
from loaders.omni_loader import OMNI3D_DATASETS


def build_seq_ctx(input_path, dataset_type):
    """Build viewer context from input path (creates loader for traj/calib/RGB)."""
    if dataset_type == "aria":
        from loaders.aria_loader import AriaLoader

        loader = AriaLoader(
            input_path,
            camera="rgb",
            with_traj=True,
            with_sdp=True,
            with_img=True,
            with_obb=False,
            restrict_range=False,
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
        from loaders.ca_loader import CALoader

        loader = CALoader(
            input_path,
            start_frame=0,
            skip_frames=1,
            max_frames=1_000_000,
            pinhole=True,
            use_canny=False,
        )
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
        import cv2
        from loaders.scannet_loader import ScanNetLoader
        from tw.camera import CameraTW
        from tw.pose import PoseTW

        loader = ScanNetLoader(
            scene_dir=input_path,
            annotation_path=os.path.expanduser("~/data/scannet/full_annotations.json"),
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
            "uid_to_p3": None,
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
        input_path = handle_input(expand_seq_shorthand(input_str))
        if not os.path.isabs(input_path) and not os.path.exists(input_path):
            resolved = os.path.expanduser(os.path.join("~/boxy_data", input_path))
            if os.path.exists(resolved):
                input_path = resolved
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
    """Resolve 2D BB CSV path. Returns path string (empty if not found)."""
    if bb2d_csv_arg:
        return os.path.join(log_dir, bb2d_csv_arg)
    default_bb2d = os.path.join(log_dir, f"{write_name}_2dbbs.csv")
    return default_bb2d if os.path.exists(default_bb2d) else ""


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
    parser.add_argument("--scannet_annotation_path", type=str, default="~/data/scannet/full_annotations.json")
    # fmt: on


def load_common(args):
    """Shared loading logic. Returns (timed_obbs, input_path, dataset_type, seq_name, log_dir, view_path, load_view_data)."""
    input_path, dataset_type, seq_name = resolve_input(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    view_path, load_view_data = load_view_file(log_dir, args.load_view)
    return input_path, dataset_type, seq_name, log_dir, view_path, load_view_data


def launch_viewer(ViewerClass):
    """Run moderngl viewer, protecting sys.argv."""
    import moderngl_window as mglw

    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    try:
        mglw.run_window_config(ViewerClass)
    finally:
        sys.argv = saved_argv
