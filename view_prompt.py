#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Interactive 2D bounding box prompting with BoxerNet 3D visualization.

Draw a 2D bounding box on the RGB image panel to prompt BoxerNet,
then view the predicted 3D OBB alongside the camera trajectory and
semi-dense points.

Usage:
    python view_prompt.py --input hohen
    python view_prompt.py --input scene0084_02
"""

import argparse
import os
import time

import cv2
import utils.imgui_compat as imgui
import numpy as np
import torch

from boxernet.boxernet import BoxerNet, sdp_to_patches
from owl.owl_wrapper import OwlWrapper
from utils.demo_utils import CKPT_PATH
from utils.image import render_depth_patches
from utils.tw.camera import CameraTW
from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.tw.tensor_utils import find_nearest2
from utils.viewer_3d import (
    SequenceOBBViewer,
    _look_at,
    _perspective_projection,
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    scale_factor,
)

# Saturated colors visible on both light and dark backgrounds
_BOX_COLORS = [
    (0.12, 0.47, 0.71),  # blue
    (1.00, 0.50, 0.05),  # orange
    (0.17, 0.63, 0.17),  # green
    (0.84, 0.15, 0.16),  # red
    (0.58, 0.40, 0.74),  # purple
    (0.55, 0.34, 0.29),  # brown
    (0.89, 0.47, 0.76),  # pink
    (0.74, 0.74, 0.13),  # olive
    (0.09, 0.75, 0.81),  # cyan
    (0.00, 0.50, 0.00),  # dark green
    (0.80, 0.20, 0.60),  # magenta
    (0.20, 0.60, 0.80),  # sky blue
    (0.90, 0.60, 0.00),  # amber
    (0.40, 0.20, 0.60),  # indigo
    (0.60, 0.80, 0.20),  # lime
    (0.80, 0.40, 0.20),  # rust
]


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Interactive 2D BB prompting with BoxerNet")
    add_common_args(parser)
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_PATH, "boxernet_hw960in4x6d768-wssxpf9p.ckpt"), help="BoxerNet checkpoint")
    parser.add_argument("--force_precision", type=str, default=None, choices=["float32", "bfloat16"])
    parser.add_argument("--force_cpu", action="store_true")
    # fmt: on
    args = parser.parse_args()

    input_path, dataset_type, seq_name, log_dir, view_path, load_view_data = (
        load_common(args)
    )
    seq_ctx = build_seq_ctx(input_path, dataset_type)

    # Load BoxerNet
    if torch.backends.mps.is_available() and not args.force_cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
    else:
        device = "cpu"
    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    if args.force_precision is not None:
        precision_dtype = (
            torch.bfloat16 if args.force_precision == "bfloat16" else torch.float32
        )
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        precision_dtype = torch.bfloat16
    else:
        precision_dtype = torch.float32

    # Load OWLv2 open-vocabulary detector
    owl = OwlWrapper(
        device,
        text_prompts=["object"],
        min_confidence=0.2,
        precision=args.force_precision,
    )

    # Build one timed_obbs entry per actual RGB frame (not per pose timestamp).
    # For Aria, seq_ctx["rgb_timestamps"] is pose_ts (~200Hz); we need the real
    # per-frame capture timestamps (~10Hz) so each step shows a new image.
    empty_obb = ObbTW(torch.zeros(0, 165))
    loader = seq_ctx.get("loader", None)
    if dataset_type == "aria" and loader is not None:
        stream_id = loader.stream_id[0]
        n_frames = loader.provider.get_num_data(stream_id)
        frame_ts = []
        for i in range(n_frames):
            _, record = loader.provider.get_image_data_by_index(stream_id, i)
            frame_ts.append(record.capture_timestamp_ns)

        # Restrict to intersection of traj and sdp time ranges so every
        # sampled frame has valid pose and semi-dense points.
        range_start = -float("inf")
        range_end = float("inf")
        pose_ts = getattr(loader, "pose_ts", None)
        if pose_ts is not None and len(pose_ts) > 0:
            range_start = max(range_start, float(min(pose_ts)))
            range_end = min(range_end, float(max(pose_ts)))
        sdp_ts = getattr(loader, "sdp_times_combined", None)
        if sdp_ts is not None and len(sdp_ts) > 0:
            range_start = max(range_start, float(min(sdp_ts)))
            range_end = min(range_end, float(max(sdp_ts)))
        frame_ts = [ts for ts in frame_ts if range_start <= ts <= range_end]

        empty_timed_obbs = {int(ts): empty_obb for ts in frame_ts}
    else:
        empty_timed_obbs = {int(ts): empty_obb for ts in seq_ctx["rgb_timestamps"]}

    default_w, default_h = 2250 * scale_factor, 1100 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class PromptViewer(SequenceOBBViewer):
        title = "BoxerNet Prompt Viewer"
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            self._prompted_obbs: list[ObbTW] = []
            self._prompted_labels: list[str] = []
            self._prompted_colors: list[tuple[float, float, float]] = []
            self._owl_text = "chair"
            self._owl_stage = 0  # 0=idle, 1=showing 2D BBs, 2=showing 3D BBs
            self._owl_stage_time = 0.0
            self._owl_2d_boxes = []  # (N, 4) raw image coords (x1, x2, y1, y2)
            self._owl_2d_scores = []  # (N,) float
            self._owl_2d_colors = []  # (N,) pre-assigned colors
            self._owl_3d_confs = []  # (N,) float, set after lift
            self._owl_cached_datum = None
            self._owl_dt_owl = 0.0
            self._owl_dt_bxr = 0.0
            self._drawing = False
            self._draw_start: tuple[float, float] | None = None
            self._draw_end: tuple[float, float] | None = None
            self._draw_start_screen: tuple[float, float] | None = None
            self._draw_end_screen: tuple[float, float] | None = None
            self._prompt_dirty = False
            self.conf_threshold = 0.6
            self._flash_text = ""
            self._flash_time = 0.0  # time remaining for flash
            self._flash_color = (0.0, 1.0, 0.0)  # green by default
            self._flash_screen_pos = None  # (x, y) screen coords for flash

            # 3D line geometry for prompted OBBs
            self._prompt_line_vbo = None
            self._prompt_line_vao = None
            self._prompt_line_count = 0

            # SDP point cloud
            self._sdp_loader = seq_ctx.get("loader", None)
            self._sdp_positions = None
            self._sdp_point_vbo = None
            self._sdp_point_vao = None
            self._sdp_point_count = 0
            # Separate VBO/VAO for points inside OBBs (rendered larger)
            self._sdp_inside_vbo = None
            self._sdp_inside_vao = None
            self._sdp_inside_count = 0
            self.show_sdp = True
            self.sdp_point_size = 3.0
            self.sdp_point_alpha = 0.3

            # Projected OBB lines for RGB overlay
            self._prompted_rgb_lines = []
            self._prompted_rgb_labels = []

            # SDP overlays
            self.show_sdp_overlay = False  # raw point projection
            self.show_sdp_patches = False  # 16x16 patch median depth

            # Always show 3DBBs in both RGB and 3D
            self.show_rgb_obbs = True
            self.show_rgb_tracked_all = True
            self.show_rgb_labels = True
            self.show_tracked_all_set = True

            # Follow-view camera state
            self.follow_view = False
            self.follow_above = 3.0
            self.follow_behind = 6.0
            self.follow_look_ahead = 2.0
            self.camera_damping = 0.92
            self._smooth_eye = None
            self._smooth_target = None
            self._smooth_up = None

            super().__init__(
                all_obbs=ObbTW(torch.zeros(0, 165)),
                root_path=log_dir,
                timed_obbs=empty_timed_obbs,
                seq_ctx=seq_ctx,
                init_color_mode=args.init_color_mode,
                init_image_panel_width=args.init_image_panel_width,
                load_view_data=load_view_data,
                view_save_path=view_path,
                seq_name=seq_name,
                skip_precompute=True,
                **kw,
            )

            # Scale up all ImGui elements
            imgui.get_style().font_scale_main *= 1.4
            self.ui_panel_width = 550

            # Load SDP after GL context is ready
            self._load_sdp_point_cloud()

            # Auto-focus camera above current frame
            self._focus_on_current_frame()

        # ── SDP loading ──────────────────────────────────────────────

        def _load_sdp_point_cloud(self):
            """Load semi-dense points into a GPU point cloud VBO."""
            # Try sdp_global first (ca1m, scannet)
            sdp_global = seq_ctx.get("sdp_global", None)
            if sdp_global is not None and len(sdp_global) > 0:
                positions = (
                    sdp_global.astype(np.float32)
                    if isinstance(sdp_global, np.ndarray)
                    else sdp_global.numpy().astype(np.float32)
                )
                P = len(positions)
                self._sdp_positions = positions
                colors = np.full((P, 3), 0.25, dtype=np.float32)
                vertex_data = np.hstack([positions, colors]).astype(np.float32)
                self._sdp_point_vbo = self.ctx.buffer(vertex_data.tobytes())
                self._sdp_point_vao = self.ctx.vertex_array(
                    self.point_prog,
                    [(self._sdp_point_vbo, "3f 3f", "in_position", "in_color")],
                )
                self._sdp_point_count = P
                print(f"Loaded {P} global semidense points as point cloud")
                return

            uid_to_p3 = seq_ctx.get("uid_to_p3", None)
            if uid_to_p3 is None or not uid_to_p3:
                # For omni3d: load per-frame SDP for the first frame
                if dataset_type == "omni3d" and self.total_frames > 0:
                    self._upload_sdp_for_frame(0)
                return

            uids = list(uid_to_p3.keys())
            P = len(uids)
            positions = np.empty((P, 3), dtype=np.float32)
            for i, uid in enumerate(uids):
                px, py, pz = uid_to_p3[uid][:3]
                positions[i] = (px, py, pz)

            self._sdp_positions = positions  # keep for recoloring
            colors = np.full((P, 3), 0.25, dtype=np.float32)
            vertex_data = np.hstack([positions, colors]).astype(np.float32)

            self._sdp_point_vbo = self.ctx.buffer(vertex_data.tobytes())
            self._sdp_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self._sdp_point_vbo, "3f 3f", "in_position", "in_color")],
            )
            self._sdp_point_count = P
            print(f"Loaded {P} semidense points as point cloud")

        def _upload_sdp_for_frame(self, idx):
            """Load per-frame SDP from the loader and upload to GPU."""
            loader = self._sdp_loader
            if loader is None or not hasattr(loader, "dataset_name"):
                return
            datum = loader.load(idx)
            sdp_w = datum.get("sdp_w", None)
            if sdp_w is None or len(sdp_w) == 0:
                self._sdp_point_count = 0
                return
            valid = ~torch.isnan(sdp_w[:, 0])
            if not valid.any():
                self._sdp_point_count = 0
                return
            positions = sdp_w[valid].numpy().astype(np.float32)
            self._sdp_positions = positions
            colors = np.full((len(positions), 3), 0.25, dtype=np.float32)
            vertex_data = np.hstack([positions, colors]).astype(np.float32)

            if self._sdp_point_vbo is not None:
                self._sdp_point_vbo.release()
            if self._sdp_point_vao is not None:
                self._sdp_point_vao.release()
            self._sdp_point_vbo = self.ctx.buffer(vertex_data.tobytes())
            self._sdp_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self._sdp_point_vbo, "3f 3f", "in_position", "in_color")],
            )
            self._sdp_point_count = len(positions)

        # ── Follow-view camera ────────────────────────────────────────

        def get_camera_matrices(self):
            """Override: position camera above and behind current T_wr."""
            if not self.follow_view or self.total_frames == 0:
                return super().get_camera_matrices()

            ts = self.sorted_timestamps[self.current_frame_idx]
            cam, T_wr = self._get_cam_and_pose(ts)
            if cam is None or T_wr is None:
                return super().get_camera_matrices()

            T_wc = T_wr @ cam.T_camera_rig.inverse()
            cam_pos = T_wc.t.reshape(3).cpu().float().numpy()
            R_wc = T_wc.R.reshape(3, 3).cpu().float().numpy()

            # Place eye behind and above the camera
            offset_local = np.array([0.0, 0.0, -self.follow_behind])
            offset = R_wc @ offset_local + np.array([0.0, 0.0, self.follow_above])
            eye = cam_pos + offset

            # Look ahead along camera's forward direction (XY plane)
            forward_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            forward_xy = np.array(
                [forward_world[0], forward_world[1], 0.0], dtype=np.float32
            )
            fwd_norm = np.linalg.norm(forward_xy)
            if fwd_norm > 1e-6:
                forward_xy /= fwd_norm
            else:
                forward_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            target = cam_pos + forward_xy * self.follow_look_ahead
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
                self._smooth_target = (
                    alpha * target + (1.0 - alpha) * self._smooth_target
                )
                self._smooth_up = alpha * up + (1.0 - alpha) * self._smooth_up
                norm = np.linalg.norm(self._smooth_up)
                if norm > 1e-6:
                    self._smooth_up /= norm

            vw, vh = self._get_3d_viewport_size()
            aspect = vw / vh
            projection = _perspective_projection(45.0, aspect, 0.1, 100.0)
            view = _look_at(
                tuple(self._smooth_eye),
                tuple(self._smooth_target),
                tuple(self._smooth_up),
            )
            mvp = np.eye(4, dtype="f4") @ view @ projection
            return projection, view, mvp

        def _focus_on_current_frame(self):
            """Snap orbit camera above current frame using follow-view params."""
            if self._smooth_eye is not None and self._smooth_target is not None:
                # Use smoothed state for seamless transition
                eye = self._smooth_eye
                target = self._smooth_target
            elif self.total_frames > 0:
                ts = self.sorted_timestamps[self.current_frame_idx]
                cam, T_wr = self._get_cam_and_pose(ts)
                if cam is None or T_wr is None:
                    return
                T_wc = T_wr @ cam.T_camera_rig.inverse()
                cam_pos = T_wc.t.reshape(3).cpu().float().numpy()
                R_wc = T_wc.R.reshape(3, 3).cpu().float().numpy()

                offset_local = np.array([0.0, 0.0, -self.follow_behind])
                offset = R_wc @ offset_local + np.array([0.0, 0.0, self.follow_above])
                eye = cam_pos + offset

                forward_world = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                forward_xy = np.array(
                    [forward_world[0], forward_world[1], 0.0], dtype=np.float32
                )
                fwd_norm = np.linalg.norm(forward_xy)
                if fwd_norm > 1e-6:
                    forward_xy /= fwd_norm
                else:
                    forward_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                target = cam_pos + forward_xy * self.follow_look_ahead
                target[2] = cam_pos[2]
            else:
                return

            # Convert eye/target to orbit camera params
            self.camera_target = target.copy()
            diff = eye - target
            self.camera_distance = float(np.linalg.norm(diff))
            if self.camera_distance > 1e-6:
                self.camera_elevation = float(
                    np.degrees(np.arcsin(diff[2] / self.camera_distance))
                )
                self.camera_azimuth = float(np.degrees(np.arctan2(diff[1], diff[0])))

            # Reset smoothing so follow-view starts clean if toggled on
            self._smooth_eye = None
            self._smooth_target = None
            self._smooth_up = None

        # ── Mouse interaction ────────────────────────────────────────

        def _screen_to_image_coords(self, sx, sy):
            """Convert screen coords to image pixel coords (in VRS resolution)."""
            rect = getattr(self, "_img_screen_rect", None)
            if rect is None:
                return None
            ix, iy, iw, ih = rect
            if sx < ix or sx > ix + iw or sy < iy or sy > iy + ih:
                return None
            u_norm = (sx - ix) / iw
            v_norm = (sy - iy) / ih
            img_u = u_norm * self._rgb_vrs_w
            img_v = v_norm * self._rgb_vrs_h

            # Aria Gen 1: undo rot90 k=3 applied in _load_rgb_for_timestamp
            if not self._vrs_is_nebula:
                orig_u = img_v
                orig_v = self._rgb_vrs_w - 1 - img_u
                img_u, img_v = orig_u, orig_v

            return img_u, img_v

        def on_mouse_press_event(self, x, y, button):
            self.imgui.mouse_press_event(x, y, button)
            # Let imgui handle interactive widgets (sliders, buttons, text)
            # Use is_any_item_hovered to avoid blocking draws on the RGB panel background
            if imgui.get_io().want_capture_mouse and imgui.is_any_item_hovered():
                return
            # Check image area (drawing 2D BBs)
            if button == 1:
                coords = self._screen_to_image_coords(x, y)
                if coords is not None:
                    self._drawing = True
                    self._draw_start = coords
                    self._draw_end = coords
                    self._draw_start_screen = (x, y)
                    self._draw_end_screen = (x, y)
                    return
            # 3D viewport — camera controls
            super().on_mouse_press_event(x, y, button)

        def on_mouse_drag_event(self, x, y, dx, dy):
            self.imgui.mouse_drag_event(x, y, dx, dy)
            # Image drawing in progress
            if self._drawing:
                coords = self._screen_to_image_coords(x, y)
                if coords is not None:
                    self._draw_end = coords
                self._draw_end_screen = (x, y)
                return
            # Let imgui handle widget drags (sliders, trackbar)
            if imgui.get_io().want_capture_mouse:
                return
            # 3D viewport — camera controls
            super().on_mouse_drag_event(x, y, dx, dy)

        def on_mouse_release_event(self, x, y, button):
            if self._drawing and button == 1:
                self._drawing = False
                coords = self._screen_to_image_coords(x, y)
                if coords is not None:
                    self._draw_end = coords

                if self._draw_start and self._draw_end:
                    u0, v0 = self._draw_start
                    u1, v1 = self._draw_end
                    xmin, xmax = min(u0, u1), max(u0, u1)
                    ymin, ymax = min(v0, v1), max(v0, v1)
                    if (xmax - xmin) > 5 and (ymax - ymin) > 5:
                        # Store 2D BB center in screen coords for flash
                        if self._draw_start_screen and self._draw_end_screen:
                            sx0, sy0 = self._draw_start_screen
                            sx1, sy1 = self._draw_end_screen
                            self._flash_screen_pos = (
                                (sx0 + sx1) / 2,
                                (sy0 + sy1) / 2,
                            )
                        self._run_boxernet_prompt(xmin, xmax, ymin, ymax)

                self._draw_start = None
                self._draw_end = None
                self._draw_start_screen = None
                self._draw_end_screen = None
                return
            super().on_mouse_release_event(x, y, button)

        # ── SDP retrieval for BoxerNet datum ─────────────────────────

        def _get_sdp_for_timestamp(self, ts_ns):
            """Get semi-dense world points for the given timestamp."""
            loader = self._sdp_loader
            # Aria path: per-timestamp SDP via SLAM observations
            if (
                loader is not None
                and hasattr(loader, "time_to_uids_combined")
                and hasattr(loader, "p3_array")
            ):
                sdp_times = loader.sdp_times_combined
                if len(sdp_times) > 0:
                    nearest_idx = find_nearest2(sdp_times, ts_ns)
                    sdp_ns = sdp_times[nearest_idx]
                    uids = loader.time_to_uids_combined[sdp_ns]
                    indices = [loader.uid_to_idx[uid] for uid in uids]
                    p3d = torch.from_numpy(loader.p3_array[indices, :3])
                    return p3d

            # Omni3D path: load per-image SDP from loader
            if loader is not None and hasattr(loader, "dataset_name"):
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                datum = loader.load(idx)
                sdp_w = datum.get("sdp_w", None)
                if sdp_w is not None and len(sdp_w) > 0:
                    valid = ~torch.isnan(sdp_w[:, 0])
                    if valid.any():
                        return sdp_w[valid].float()

            # CA1M path: load per-frame depth points
            if loader is not None and hasattr(loader, "image_tags"):
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                frame = loader._load_frame(loader.image_tags[idx])
                sdp_w = frame["sdp_w"]
                if len(sdp_w) > 0:
                    return sdp_w.float()

            # ScanNet/other: subsample from global point cloud
            sdp_global = seq_ctx.get("sdp_global", None)
            if sdp_global is not None and len(sdp_global) > 0:
                n = min(10000, len(sdp_global))
                idx = np.random.choice(len(sdp_global), n, replace=False)
                return torch.from_numpy(sdp_global[idx]).float()

            return torch.zeros(0, 3, dtype=torch.float32)

        # ── BoxerNet inference ───────────────────────────────────────

        def _run_boxernet_prompt(self, xmin, xmax, ymin, ymax):
            """Run BoxerNet on the drawn 2D BB and add result to prompted_obbs."""
            if self.total_frames == 0:
                return

            ts_ns = self.sorted_timestamps[self.current_frame_idx]
            cam, T_wr = self._get_cam_and_pose(ts_ns)
            if cam is None or T_wr is None:
                print("No camera/pose available for this frame")
                return

            img_np = self._load_raw_image(ts_ns)
            if img_np is None:
                print("Failed to load raw image")
                return

            H, W = img_np.shape[:2]
            bxr_hw = boxernet.hw
            img_resized = cv2.resize(img_np, (bxr_hw, bxr_hw))
            img_torch = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

            scale_x = bxr_hw / W
            scale_y = bxr_hw / H
            bb2d = torch.tensor(
                [[xmin * scale_x, xmax * scale_x, ymin * scale_y, ymax * scale_y]],
                dtype=torch.float32,
            )

            cam_data = cam._data.clone()
            cam_scaled = CameraTW(cam_data)
            cam_scaled._data[0] = bxr_hw
            cam_scaled._data[1] = bxr_hw
            cam_scaled._data[2] *= scale_x
            cam_scaled._data[3] *= scale_y
            cam_scaled._data[4] *= scale_x
            cam_scaled._data[5] *= scale_y

            sdp_w = self._get_sdp_for_timestamp(ts_ns)
            rotated = not self._vrs_is_nebula

            datum = {
                "img0": img_torch[None],
                "cam0": cam_scaled.float(),
                "T_world_rig0": T_wr.float(),
                "rotated0": torch.tensor([rotated]),
                "sdp_w": sdp_w.float(),
                "bb2d": bb2d,
            }

            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to(device)
                elif hasattr(v, "_data"):
                    datum[k] = v.to(device)

            try:
                t0 = time.perf_counter()
                if device == "mps":
                    outputs = boxernet.forward(datum)
                else:
                    with torch.autocast(device_type=device, dtype=precision_dtype):
                        outputs = boxernet.forward(datum)
                obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
                dt_ms = (time.perf_counter() - t0) * 1000
                timing = f"(forward took: {dt_ms:.0f}ms, {device}, {precision_dtype})"

                if len(obb_pr_w) > 0:
                    obb = obb_pr_w[0]
                    conf = float(obb.prob.squeeze())
                    if conf >= self.conf_threshold:
                        print(f"BoxerNet prompt -> conf={conf:.2f} {timing}")
                        self._prompted_obbs.append(obb)
                        self._prompted_labels.append("")
                        self._prompted_colors.append(
                            _BOX_COLORS[len(self._prompted_obbs) % len(_BOX_COLORS)]
                        )
                        self._prompt_dirty = True
                        self._flash_text = f"conf={conf:.2f}"
                        self._flash_color = (0.0, 1.0, 0.3)
                    else:
                        print(f"BoxerNet prompt -> conf={conf:.2f} (rejected) {timing}")
                        self._flash_text = f"conf={conf:.2f} (rejected)"
                        self._flash_color = (1.0, 0.2, 0.2)
                    self._flash_time = 1.5
                else:
                    print(f"BoxerNet prompt -> no prediction {timing}")
                    self._flash_text = "no prediction"
                    self._flash_color = (1.0, 0.2, 0.2)
                    self._flash_time = 1.5
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  -> BoxerNet error: {e}")

        def _image_to_screen_coords(self, img_u, img_v):
            """Convert raw image pixel coords to screen coords (inverse of _screen_to_image_coords)."""
            rect = getattr(self, "_img_screen_rect", None)
            if rect is None:
                return None
            ix, iy, iw, ih = rect
            # Aria Gen 1: apply rot90 k=3
            if not self._vrs_is_nebula:
                disp_u = self._rgb_vrs_w - 1 - img_v
                disp_v = img_u
            else:
                disp_u = img_u
                disp_v = img_v
            u_norm = disp_u / self._rgb_vrs_w
            v_norm = disp_v / self._rgb_vrs_h
            return ix + u_norm * iw, iy + v_norm * ih

        def _run_owl_prompt(self):
            """Run OWL 2D detection and enter stage 1 (showing 2D BBs)."""
            self._owl_stage = 0  # reset so early returns don't re-trigger
            if self.total_frames == 0:
                return

            ts_ns = self.sorted_timestamps[self.current_frame_idx]
            cam, T_wr = self._get_cam_and_pose(ts_ns)
            if cam is None or T_wr is None:
                print("No camera/pose available for this frame")
                return

            img_np = self._load_raw_image(ts_ns)
            if img_np is None:
                print("Failed to load raw image")
                return

            H, W = img_np.shape[:2]
            rotated = not self._vrs_is_nebula
            # Run OWL 2D detection
            owl.set_text_prompts([self._owl_text])
            img_torch_255 = (
                torch.from_numpy(img_np).permute(2, 0, 1).float()[None]
            )  # (1, 3, H, W) in [0, 255]
            t0 = time.perf_counter()
            boxes, scores2d, label_ints, _ = owl.forward(
                img_torch_255, rotated, resize_to_HW=(906, 906)
            )
            self._owl_dt_owl = (time.perf_counter() - t0) * 1000

            n_dets = len(boxes)
            if n_dets == 0:
                print(
                    f"OWL: 0 detections for '{self._owl_text}' ({self._owl_dt_owl:.0f}ms)"
                )
                self._flash_text = f"0 detections for '{self._owl_text}'"
                self._flash_color = (1.0, 0.2, 0.2)
                self._flash_time = 1.5
                rect = getattr(self, "_img_screen_rect", None)
                if rect is not None:
                    self._flash_screen_pos = (
                        rect[0] + rect[2] / 2,
                        rect[1] + rect[3] / 2,
                    )
                return

            # Store 2D results for overlay, pre-assign colors to match 3D
            self._owl_2d_boxes = (
                boxes.numpy()
            )  # (N, 4) in (x1, x2, y1, y2) raw image coords
            self._owl_2d_scores = scores2d.numpy()  # (N,)
            base = len(self._prompted_obbs)
            self._owl_2d_colors = [
                _BOX_COLORS[(base + 1 + i) % len(_BOX_COLORS)] for i in range(n_dets)
            ]

            # Cache BoxerNet datum for stage 2
            bxr_hw = boxernet.hw
            scale_x = bxr_hw / W
            scale_y = bxr_hw / H
            bb2d = boxes.clone()
            bb2d[:, 0] *= scale_x
            bb2d[:, 1] *= scale_x
            bb2d[:, 2] *= scale_y
            bb2d[:, 3] *= scale_y

            img_resized = cv2.resize(img_np, (bxr_hw, bxr_hw))
            img_torch = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

            cam_data = cam._data.clone()
            cam_scaled = CameraTW(cam_data)
            cam_scaled._data[0] = bxr_hw
            cam_scaled._data[1] = bxr_hw
            cam_scaled._data[2] *= scale_x
            cam_scaled._data[3] *= scale_y
            cam_scaled._data[4] *= scale_x
            cam_scaled._data[5] *= scale_y

            sdp_w = self._get_sdp_for_timestamp(ts_ns)

            datum = {
                "img0": img_torch[None],
                "cam0": cam_scaled.float(),
                "T_world_rig0": T_wr.float(),
                "rotated0": torch.tensor([rotated]),
                "sdp_w": sdp_w.float(),
                "bb2d": bb2d,
            }
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to(device)
                elif hasattr(v, "_data"):
                    datum[k] = v.to(device)
            self._owl_cached_datum = datum

            print(
                f"OWL: {n_dets} detections for '{self._owl_text}' ({self._owl_dt_owl:.0f}ms)"
            )
            self._owl_stage = 1
            self._owl_stage_time = 12.0

        def _run_owl_lift(self):
            """Stage 2: lift cached OWL 2D BBs to 3D via BoxerNet."""
            datum = self._owl_cached_datum
            if datum is None:
                self._owl_stage = 0
                return

            try:
                t1 = time.perf_counter()
                if device == "mps":
                    outputs = boxernet.forward(datum)
                else:
                    with torch.autocast(device_type=device, dtype=precision_dtype):
                        outputs = boxernet.forward(datum)
                obb_pr_w = outputs["obbs_pr_w"].cpu()[0]  # (M, ...)
                self._owl_dt_bxr = (time.perf_counter() - t1) * 1000

                self._owl_3d_confs = []
                n_accepted = 0
                for i in range(len(obb_pr_w)):
                    obb = obb_pr_w[i]
                    conf = float(obb.prob.squeeze())
                    self._owl_3d_confs.append(conf)
                    if conf >= self.conf_threshold:
                        n_accepted += 1
                        self._prompted_obbs.append(obb)
                        self._prompted_labels.append(self._owl_text)
                        self._prompted_colors.append(self._owl_2d_colors[i])

                self._prompt_dirty = True
                n_dets = len(obb_pr_w)
                timing = f"owl:{self._owl_dt_owl:.0f}ms bxr:{self._owl_dt_bxr:.0f}ms"
                print(
                    f"BoxerNet lift: {n_accepted}/{n_dets} accepted "
                    f"(>={self.conf_threshold:.2f}) {timing}"
                )

                self._owl_stage = 2
                self._owl_stage_time = 2.0
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  -> BoxerNet lift error: {e}")
                self._owl_stage = 0

            self._owl_cached_datum = None

        def _load_raw_image(self, ts_ns):
            """Load the raw (unscaled, unrotated) RGB image for a timestamp."""
            if (
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
                return img
            elif getattr(self, "_data_source", None) == "scannet":
                scene_dir = getattr(self, "_scannet_scene_dir", None)
                if scene_dir is None:
                    return None
                frame_ids = getattr(self, "_scannet_frame_ids", None)
                if frame_ids is not None:
                    idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                    fid = str(int(frame_ids[idx]))
                else:
                    fid = str(int(ts_ns))
                for ext in [".jpg", ".png"]:
                    path = os.path.join(scene_dir, "frames", "color", f"{fid}{ext}")
                    if os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return None
            elif (
                getattr(self, "_data_source", None) == "omni3d"
                and self._loader is not None
            ):
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                datum = self._loader.load(idx)
                img_t = datum["img0"][0].permute(1, 2, 0).cpu().numpy()
                return np.clip(img_t * 255.0, 0, 255).astype(np.uint8)
            elif (
                getattr(self, "_data_source", None) == "ca1m"
                and self._loader is not None
            ):
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                tag = self._loader.image_tags[idx]
                img_path = os.path.join(
                    self._loader.data_dir, tag + ".wide", "image.png"
                )
                img = cv2.imread(img_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return None
            elif getattr(self, "_rgb_images", None) is not None:
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                img = self._rgb_images[idx]
                if img is not None and img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return img
            return None

        # ── 3D geometry for prompted OBBs ────────────────────────────

        def _rebuild_prompted_geometry(self):
            """Rebuild 3D line geometry and RGB projections for prompted OBBs."""
            # Release old GPU buffers
            if self._prompt_line_vbo is not None:
                self._prompt_line_vbo.release()
                self._prompt_line_vbo = None
            if self._prompt_line_vao is not None:
                self._prompt_line_vao.release()
                self._prompt_line_vao = None
            self._prompt_line_count = 0
            self._prompted_rgb_lines = []
            self._prompted_rgb_labels = []

            if len(self._prompted_obbs) == 0:
                self._recolor_sdp_points()
                return

            stacked = ObbTW(torch.stack([o._data for o in self._prompted_obbs]))
            N = len(stacked)
            corners = stacked.bb3corners_world  # (N, 8, 3)
            probs = stacked.prob.squeeze(-1)  # (N,)

            edge_indices = torch.tensor(
                BB3D_LINE_ORDERS, dtype=torch.long, device=corners.device
            )
            batch_idx = torch.arange(N, device=corners.device)[:, None].expand(N, 12)
            start_idx = edge_indices[:, 0][None, :].expand(N, 12)
            end_idx = edge_indices[:, 1][None, :].expand(N, 12)
            start_pts = corners[batch_idx, start_idx]  # (N, 12, 3)
            end_pts = corners[batch_idx, end_idx]  # (N, 12, 3)

            # Per-box colors
            color_list = [
                torch.tensor(self._prompted_colors[i], dtype=torch.float32)
                for i in range(N)
            ]
            colors = torch.stack(color_list)  # (N, 3)
            colors = colors[:, None, :].expand(N, 12, 3)
            probs_exp = probs[:, None, None].expand(N, 12, 1)

            instance_data = torch.cat(
                [start_pts, end_pts, colors, probs_exp], dim=2
            )  # (N, 12, 10)
            instance_array = instance_data.reshape(-1, 10).cpu().numpy().astype("f4")
            self._prompt_line_count = len(instance_array)

            self._prompt_line_vbo = self.ctx.buffer(instance_array.tobytes())
            self._prompt_line_vao = self.ctx.vertex_array(
                self.line_prog,
                [
                    (self.quad_vbo, "2f", "in_quad_pos"),
                    (
                        self._prompt_line_vbo,
                        "3f 3f 3f 1f /i",
                        "start_pos",
                        "end_pos",
                        "line_color",
                        "line_prob",
                    ),
                ],
            )

            # Recolor SDP points that fall inside prompted OBBs
            self._recolor_sdp_points()

            # Also project onto current RGB frame
            self._update_prompted_rgb_projections()

        def _recolor_sdp_points(self):
            """Color SDP points by their containing OBB instance color.

            Points inside an OBB are moved to a separate buffer so they
            can be rendered at 2x point size.
            """
            if self._sdp_positions is None or len(self._sdp_positions) == 0:
                return

            positions = self._sdp_positions  # (P, 3) numpy
            P = len(positions)
            colors = np.full((P, 3), 0.25, dtype=np.float32)
            any_inside = np.zeros(P, dtype=bool)

            if len(self._prompted_obbs) > 0:
                pts_t = torch.from_numpy(positions).float()
                for obb, color in zip(self._prompted_obbs, self._prompted_colors):
                    inside = obb.points_inside_bb3(pts_t).numpy()
                    colors[inside] = color
                    any_inside |= inside

            # Outside points -> main VBO
            outside = ~any_inside
            out_data = np.hstack([positions[outside], colors[outside]]).astype(
                np.float32
            )
            self._sdp_point_count = int(outside.sum())
            if self._sdp_point_vbo is not None:
                self._sdp_point_vbo.release()
            self._sdp_point_vbo = self.ctx.buffer(out_data.tobytes())
            self._sdp_point_vao = self.ctx.vertex_array(
                self.point_prog,
                [(self._sdp_point_vbo, "3f 3f", "in_position", "in_color")],
            )

            # Inside points -> separate VBO (rendered at 2x size)
            n_inside = int(any_inside.sum())
            if n_inside > 0:
                in_data = np.hstack([positions[any_inside], colors[any_inside]]).astype(
                    np.float32
                )
                if self._sdp_inside_vbo is not None:
                    self._sdp_inside_vbo.release()
                self._sdp_inside_vbo = self.ctx.buffer(in_data.tobytes())
                self._sdp_inside_vao = self.ctx.vertex_array(
                    self.point_prog,
                    [(self._sdp_inside_vbo, "3f 3f", "in_position", "in_color")],
                )
                self._sdp_inside_count = n_inside
            else:
                self._sdp_inside_count = 0

        def _update_prompted_rgb_projections(self):
            """Project prompted OBBs onto the current RGB frame for overlay."""
            self._prompted_rgb_lines = []
            self._prompted_rgb_labels = []
            if len(self._prompted_obbs) == 0:
                return
            if self.total_frames == 0:
                return
            ts = self.sorted_timestamps[self.current_frame_idx]
            nav_ts = self._get_navigation_timestamp(self.current_frame_idx, ts)
            cam, T_wr = self._get_cam_and_pose(nav_ts)
            if cam is None or T_wr is None:
                return

            stacked = ObbTW(torch.stack([o._data for o in self._prompted_obbs]))
            colors = np.array(self._prompted_colors, dtype=np.float32)
            lines, labels = self._project_obbs_for_rgb(
                stacked, cam, T_wr, colors, labels=self._prompted_labels
            )
            self._prompted_rgb_lines = lines
            self._prompted_rgb_labels = labels

        # ── Rendering ────────────────────────────────────────────────

        def render_3d(self, time_val: float, frame_time: float) -> None:
            super().render_3d(time_val, frame_time)

            # Get 3D viewport dimensions
            full_w, _ = self.wnd.size
            w, h = self._get_3d_viewport_size()
            vp_x = full_w - w
            self.ctx.viewport = (vp_x, 0, w, h)
            self.ctx.scissor = (vp_x, 0, w, h)

            _, _, mvp = self.get_camera_matrices()
            mvp_bytes = np.array(mvp, dtype="f4").tobytes()

            # Render SDP point cloud
            if (
                self.show_sdp
                and self._sdp_point_vao is not None
                and self._sdp_point_count > 0
            ):
                self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
                self.point_prog["mvp"].write(mvp_bytes)
                self.point_prog["point_size"].write(
                    np.array(self.sdp_point_size, dtype="f4").tobytes()
                )
                self.point_prog["alpha"].write(
                    np.array(self.sdp_point_alpha, dtype="f4").tobytes()
                )
                self._sdp_point_vao.render(
                    mode=self.ctx.POINTS, vertices=self._sdp_point_count
                )

            # Render SDP points inside OBBs at 2x size
            if (
                self.show_sdp
                and self._sdp_inside_vao is not None
                and self._sdp_inside_count > 0
            ):
                self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
                self.point_prog["mvp"].write(mvp_bytes)
                self.point_prog["point_size"].write(
                    np.array(self.sdp_point_size * 2.0, dtype="f4").tobytes()
                )
                self.point_prog["alpha"].write(
                    np.array(self.sdp_point_alpha, dtype="f4").tobytes()
                )
                self._sdp_inside_vao.render(
                    mode=self.ctx.POINTS, vertices=self._sdp_inside_count
                )

            # Render prompted OBB lines
            if self._prompt_line_vao is not None and self._prompt_line_count > 0:
                self.line_prog["mvp"].write(mvp_bytes)
                self.line_prog["line_width"].write(np.array(3.0, dtype="f4").tobytes())
                self.line_prog["prob_threshold"].write(
                    np.array(0.0, dtype="f4").tobytes()
                )
                viewport = np.array([w, h], dtype="f4")
                self.line_prog["viewport_size"].write(viewport.tobytes())
                self._prompt_line_vao.render(
                    mode=self.ctx.TRIANGLES,
                    instances=self._prompt_line_count,
                )

            # Restore full viewport
            full_w, full_h = self.wnd.size
            self.ctx.viewport = (0, 0, full_w, full_h)
            self.ctx.scissor = None

        def render_ui(self) -> None:
            """Render UI with drawing overlay and prompt controls."""
            if self._prompt_dirty:
                self._rebuild_prompted_geometry()
                self._prompt_dirty = False

            # Render the control panel (replaces OBBViewer.render_ui
            # to use our wider panel width)
            if self.show_tracked_all_set:
                self._render_text_labels()
            w, h = self.wnd.size
            imgui.set_next_window_position(0, 0, imgui.ONCE)
            imgui.set_next_window_size(self.ui_panel_width, h - 95, imgui.ALWAYS)
            imgui.begin(
                "OBB Controls",
                flags=imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
            )
            self._render_main_controls()
            imgui.end()

            # Now render the RGB panel ourselves so we can capture img_min
            # and draw prompted OBB projections
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
                    flags=imgui.WINDOW_NO_RESIZE
                    | imgui.WINDOW_NO_MOVE
                    | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
                )
                if expanded:
                    avail_w, avail_h = imgui.get_content_region_available()
                    img_scale = min(avail_w / tex_w, avail_h / tex_h)
                    draw_w = tex_w * img_scale
                    draw_h = tex_h * img_scale
                    imgui.image(self._rgb_texture.glo, draw_w, draw_h)
                    img_min = imgui.get_item_rect_min()

                    # Store exact image rect for mouse hit-testing
                    self._img_screen_rect = (
                        img_min.x,
                        img_min.y,
                        draw_w,
                        draw_h,
                    )

                    scale_x = draw_w / tex_w * self._rgb_img_scale
                    scale_y = draw_h / tex_h * self._rgb_img_scale
                    draw_list = imgui.get_window_draw_list()

                    # Draw prompted OBB projections on the RGB image
                    for edge_pts, edge_valid, color in self._prompted_rgb_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]),
                            float(color[1]),
                            float(color[2]),
                            1.0,
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
                                        self.rgb_obb_thickness,
                                    )

                imgui.end()

            # Draw rectangle overlay while drawing
            if self._drawing and self._draw_start_screen and self._draw_end_screen:
                draw_list = imgui.get_foreground_draw_list()
                sx0, sy0 = self._draw_start_screen
                sx1, sy1 = self._draw_end_screen
                green = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 0.8)
                draw_list.add_rect(sx0, sy0, sx1, sy1, green, thickness=5.0)

            # Flash confidence text centered on the 2D BB
            if self._flash_time > 0 and self._flash_screen_pos is not None:
                self._flash_time -= 1.0 / 60.0  # approximate frame time
                alpha = min(1.0, self._flash_time / 0.5)  # fade out
                r, g, b = self._flash_color
                draw_list = imgui.get_foreground_draw_list()
                cx, cy = self._flash_screen_pos
                tw = len(self._flash_text) * 7 + 16
                tx = cx - tw * 0.5
                ty = cy - 10
                col = imgui.get_color_u32_rgba(r, g, b, alpha)
                draw_list.add_text(tx + 8, ty, col, self._flash_text)

            # ── OWL staged detection overlay ─────────────────────────
            if self._owl_stage == -1:
                # Frame 1: show "Detecting..." overlay, advance to stage -2
                rect = getattr(self, "_img_screen_rect", None)
                if rect is not None:
                    draw_list = imgui.get_foreground_draw_list()
                    cx = rect[0] + rect[2] / 2
                    cy = rect[1] + rect[3] / 2
                    text = f"Detecting '{self._owl_text}'..."
                    tw = len(text) * 7 + 16
                    col = imgui.get_color_u32_rgba(1.0, 1.0, 0.2, 1.0)
                    draw_list.add_text(cx - tw / 2 + 8, cy - 8, col, text)
                self._owl_stage = -2  # will run OWL next frame

            elif self._owl_stage == -2:
                # Frame 2: run OWL (blocks), keep showing overlay
                rect = getattr(self, "_img_screen_rect", None)
                if rect is not None:
                    draw_list = imgui.get_foreground_draw_list()
                    cx = rect[0] + rect[2] / 2
                    cy = rect[1] + rect[3] / 2
                    text = f"Detecting '{self._owl_text}'..."
                    tw = len(text) * 7 + 16
                    col = imgui.get_color_u32_rgba(1.0, 1.0, 0.2, 1.0)
                    draw_list.add_text(cx - tw / 2 + 8, cy - 8, col, text)
                self._run_owl_prompt()

            if self._owl_stage > 0:
                self._owl_stage_time -= 1.0 / 60.0
                draw_list = imgui.get_foreground_draw_list()

                if self._owl_stage == 1:
                    # Draw 2D BBs with scores on the image
                    for i, (box, score) in enumerate(
                        zip(self._owl_2d_boxes, self._owl_2d_scores)
                    ):
                        x1, x2, y1, y2 = box
                        tl = self._image_to_screen_coords(x1, y1)
                        br = self._image_to_screen_coords(x2, y2)
                        if tl is None or br is None:
                            continue
                        color = self._owl_2d_colors[i]
                        col = imgui.get_color_u32_rgba(
                            color[0], color[1], color[2], 0.9
                        )
                        draw_list.add_rect(
                            tl[0], tl[1], br[0], br[1], col, thickness=6.0
                        )
                        # Score label at top-left of box
                        label = f"{score:.2f}"
                        draw_list.add_text(tl[0] + 4, tl[1] - 16, col, label)

                    # Lift to 3D on the next frame (so 2D BBs render first)
                    if self._owl_stage_time < 12.0:
                        self._run_owl_lift()

                elif self._owl_stage == 2:
                    # Draw 2D BBs colored by 3D acceptance
                    for i, (box, conf3d) in enumerate(
                        zip(self._owl_2d_boxes, self._owl_3d_confs)
                    ):
                        x1, x2, y1, y2 = box
                        tl = self._image_to_screen_coords(x1, y1)
                        br = self._image_to_screen_coords(x2, y2)
                        if tl is None or br is None:
                            continue
                        accepted = conf3d >= self.conf_threshold
                        if accepted:
                            color = self._owl_2d_colors[i]
                            col = imgui.get_color_u32_rgba(
                                color[0], color[1], color[2], 0.9
                            )
                        else:
                            col = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 0.5)
                        draw_list.add_rect(
                            tl[0], tl[1], br[0], br[1], col, thickness=3.0
                        )
                        # 3D conf label
                        tag = "3D" if accepted else "rej"
                        label = f"{tag} {conf3d:.2f}"
                        draw_list.add_text(tl[0] + 4, tl[1] - 16, col, label)

                    if self._owl_stage_time <= 0:
                        self._owl_stage = 0
                        self._owl_2d_boxes = []
                        self._owl_2d_scores = []
                        self._owl_2d_colors = []
                        self._owl_3d_confs = []

            # ── Bottom playback bar ──────────────────────────────────
            self._render_bottom_playback_bar()

        def _render_bottom_playback_bar(self):
            """Render a horizontal playback bar anchored to the bottom."""
            import time as time_module

            win_w, win_h = self.wnd.size
            bar_h = 95
            imgui.set_next_window_position(0, win_h - bar_h, imgui.ALWAYS)
            imgui.set_next_window_size(win_w, bar_h, imgui.ALWAYS)
            imgui.begin(
                "Playback",
                flags=(
                    imgui.WINDOW_NO_RESIZE
                    | imgui.WINDOW_NO_MOVE
                    | imgui.WINDOW_NO_TITLE_BAR
                    | imgui.WINDOW_NO_SCROLLBAR
                ),
            )

            if imgui.button(
                "Play" if not self.is_playing else "Pause", width=120, height=45
            ):
                self.is_playing = not self.is_playing
                if self.is_playing:
                    self._smooth_eye = None
                    self._smooth_target = None
                    self._smooth_up = None
                    self.follow_view = True
                else:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._last_step_time = time_module.time()
            imgui.same_line()
            if imgui.button("<", width=45, height=45):
                self.is_playing = False
                if self.follow_view:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._step_to_frame(self.current_frame_idx - 1)
            imgui.same_line()
            if imgui.button(">", width=45, height=45):
                self.is_playing = False
                if self.follow_view:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._step_forward()

            if self.total_frames > 0:
                imgui.same_line()
                slider_w = max(200, win_w - 600)
                imgui.push_item_width(slider_w)
                changed, new_frame = imgui.slider_int(
                    "##frame",
                    self.current_frame_idx,
                    0,
                    max(0, self.total_frames - 1),
                )
                if changed:
                    self.is_playing = False
                    if self.follow_view:
                        self._focus_on_current_frame()
                        self.follow_view = False
                    self._step_to_frame(new_frame)
                imgui.pop_item_width()

                imgui.same_line()
                imgui.push_item_width(120)
                _changed, self.playback_fps = imgui.slider_float(
                    "FPS", self.playback_fps, 0.5, 60.0
                )
                imgui.pop_item_width()

            imgui.end()

        def _render_main_controls(self):
            """Render prompt + visualization controls (playback is in bottom bar)."""
            # Prompt controls (in sidebar)
            self._section_header("Prompt")
            imgui.text_colored("Option A: Drag a 2DBB", 0.0, 1.0, 0.0)

            # OWL text-prompt detection
            imgui.spacing()
            imgui.text_colored("Option B: Detect Text (OWL)", 0.2, 0.8, 1.0)
            imgui.spacing()
            imgui.push_item_width(300)
            enter_pressed, self._owl_text = imgui.input_text(
                "##owl_prompt",
                self._owl_text,
                256,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
            )
            self._owl_text_active = imgui.is_item_active()
            if enter_pressed:
                imgui.set_keyboard_focus_here(-1)
            imgui.pop_item_width()
            imgui.same_line()
            if (imgui.button("Detect") or enter_pressed) and self._owl_stage == 0:
                self._owl_stage = -1  # pending: renders overlay, then runs OWL
            imgui.spacing()

            if imgui.button("Clear All"):
                self._prompted_obbs.clear()
                self._prompted_labels.clear()
                self._prompted_colors.clear()
                self._prompt_dirty = True
            if len(self._prompted_obbs) > 0:
                imgui.same_line()
                if imgui.button("Undo Last"):
                    self._prompted_obbs.pop()
                    self._prompted_labels.pop()
                    self._prompted_colors.pop()
                    self._prompt_dirty = True

            imgui.push_item_width(200)
            _changed, owl.min_confidence = imgui.slider_float(
                "2DBB Conf Threshold", owl.min_confidence, 0.0, 1.0
            )
            _changed, self.conf_threshold = imgui.slider_float(
                "3DBB Conf Threshold", self.conf_threshold, 0.0, 1.0
            )
            imgui.pop_item_width()
            imgui.text(f"Total 3D boxes: {len(self._prompted_obbs)}")

            self._section_header("Visualization")
            imgui.push_item_width(200)
            _theme_changed, self.visual_theme_mode = imgui.combo(
                "Theme", self.visual_theme_mode, ["Light", "Dark"]
            )
            if _theme_changed:
                self._apply_visual_theme()
            _changed, self.show_trajectory = imgui.checkbox(
                "Show Trajectory", self.show_trajectory
            )
            _changed, self.show_frustum = imgui.checkbox(
                "Show Frustum", self.show_frustum
            )
            _changed, self.show_rgb = imgui.checkbox("Show RGB Panel", self.show_rgb)
            if self.show_rgb:
                _changed, self.rgb_panel_max_frac = imgui.slider_float(
                    "RGB Panel Width", self.rgb_panel_max_frac, 0.25, 0.75
                )
                _changed, self.rgb_obb_thickness = imgui.slider_float(
                    "RGB 3DBB Width", self.rgb_obb_thickness, 1.0, 10.0
                )
                sdp_changed, self.show_sdp_overlay = imgui.checkbox(
                    "Show SDP Depth Overlay", self.show_sdp_overlay
                )
                if sdp_changed and self.show_sdp_overlay:
                    self.show_sdp_patches = False  # mutually exclusive
                patch_changed, self.show_sdp_patches = imgui.checkbox(
                    "Show SDP Patch Overlay", self.show_sdp_patches
                )
                if patch_changed and self.show_sdp_patches:
                    self.show_sdp_overlay = False  # mutually exclusive
                if sdp_changed or patch_changed:
                    if self.show_sdp_patches:
                        self._reupload_rgb_with_sdp_patches()
                    elif self.show_sdp_overlay:
                        self._reupload_rgb_with_sdp()
                    else:
                        ts = self.sorted_timestamps[self.current_frame_idx]
                        rgb = self._load_rgb_for_timestamp(ts)
                        if rgb is not None:
                            self._upload_rgb_texture(rgb)
            imgui.pop_item_width()

            # SDP controls
            self._section_header("Points")
            if self._sdp_point_count > 0:
                imgui.text(f"{self._sdp_point_count} semi-dense points")
                _changed, self.show_sdp = imgui.checkbox("Show Points", self.show_sdp)
                imgui.push_item_width(200)
                _changed, self.sdp_point_size = imgui.slider_float(
                    "Point Size", self.sdp_point_size, 1.0, 10.0
                )
                _changed, self.sdp_point_alpha = imgui.slider_float(
                    "Point Alpha", self.sdp_point_alpha, 0.01, 1.0
                )
                imgui.pop_item_width()
            else:
                imgui.text("No semi-dense points loaded")

            self._section_header("Camera")
            imgui.text("Follow: ON" if self.follow_view else "Follow: OFF")
            imgui.push_item_width(200)
            _changed, self.camera_damping = imgui.slider_float(
                "Damping", self.camera_damping, 0.0, 0.99
            )
            _changed, self.follow_behind = imgui.slider_float(
                "Behind", self.follow_behind, 0.0, 10.0
            )
            _changed, self.follow_above = imgui.slider_float(
                "Above", self.follow_above, 0.0, 10.0
            )
            _changed, self.follow_look_ahead = imgui.slider_float(
                "Look Ahead", self.follow_look_ahead, 0.0, 10.0
            )
            imgui.pop_item_width()
            if imgui.button("Focus on Scene"):
                self._focus_on_current_frame()

            # Help
            self._section_header("Help")
            _help = [
                "[Space] Play / Pause",
                "[Left / Right] Step Frame",
                "[Left click] Draw 2D Box",
                "[Right drag] Orbit Camera",
                "[Middle drag] Pan Camera",
                "[Scroll] Zoom",
                "[Escape] Quit",
            ]
            for line in _help:
                imgui.text(line)

        def _reupload_rgb_with_sdp(self):
            """Re-upload RGB texture with projected SDP depth overlay."""
            if self.total_frames == 0:
                return
            ts = self.sorted_timestamps[self.current_frame_idx]
            rgb = self._load_rgb_for_timestamp(ts)
            if rgb is None:
                return

            cam, T_wr = self._get_cam_and_pose(ts)
            if cam is None or T_wr is None:
                self._upload_rgb_texture(rgb)
                return

            # Get SDP 3D points for current timestamp
            sdp_w = self._get_sdp_for_timestamp(ts)
            if len(sdp_w) == 0:
                self._upload_rgb_texture(rgb)
                return

            # Project 3D points into camera
            T_wc = T_wr @ cam.T_camera_rig.inverse()
            pts_cam = T_wc.inverse().transform(sdp_w.float())

            # Scale camera to VRS resolution for projection
            proj_cam = cam
            if self._rgb_vrs_w > 0 and self._rgb_vrs_h > 0:
                cam_w = cam.size[..., 0].item()
                cam_h = cam.size[..., 1].item()
                if abs(cam_w - self._rgb_vrs_w) > 1 or abs(cam_h - self._rgb_vrs_h) > 1:
                    proj_cam = cam.scale_to_size((self._rgb_vrs_w, self._rgb_vrs_h))
            fov = 140.0 if self._vrs_is_nebula else 120.0
            pts_2d, valid = proj_cam.project(pts_cam.unsqueeze(0), fov_deg=fov)
            pts_2d = pts_2d.squeeze(0).cpu().numpy()
            valid = valid.squeeze(0).cpu().numpy()
            depths = pts_cam[:, 2].cpu().numpy()

            # Filter to valid projections with positive depth
            mask = valid & (depths > 0.1) & (depths < 10.0)
            if not mask.any():
                self._upload_rgb_texture(rgb)
                return

            pts_2d = pts_2d[mask]
            depths = depths[mask]

            # pts_2d is now in VRS pixel coords (_rgb_vrs_w × _rgb_vrs_h).
            # The displayed RGB was rotated (rot90 k=3 for non-nebula) then
            # resized by _rgb_img_scale.  Apply the same transforms.
            if not self._vrs_is_nebula:
                # rot90 k=3: (x, y) -> (H-1-y, x)  where H = _rgb_vrs_h
                old_x = pts_2d[:, 0].copy()
                old_y = pts_2d[:, 1].copy()
                pts_2d[:, 0] = self._rgb_vrs_h - 1 - old_y
                pts_2d[:, 1] = old_x

            # Scale by _rgb_img_scale (same resize applied to the displayed image)
            pts_2d *= self._rgb_img_scale

            # Map to displayed image pixel coords
            HH, WW = rgb.shape[:2]
            depth_img = np.zeros((HH, WW), dtype=np.float32)
            px = np.round(pts_2d[:, 0]).astype(np.int32)
            py = np.round(pts_2d[:, 1]).astype(np.int32)
            in_bounds = (px >= 0) & (px < WW) & (py >= 0) & (py < HH)
            px, py, d = px[in_bounds], py[in_bounds], depths[in_bounds]

            # Use a small radius to make points more visible
            for r in range(-3, 4):
                for c in range(-3, 4):
                    ppx = np.clip(px + c, 0, WW - 1)
                    ppy = np.clip(py + r, 0, HH - 1)
                    depth_img[ppy, ppx] = d

            # Colorize with jet and blend at 40% overlay
            max_depth, min_depth = 5.0, 0.1
            d_norm = np.clip((depth_img - min_depth) / (max_depth - min_depth), 0, 1)
            d_u8 = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            d_color_rgb = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
            has_depth = depth_img > 0.1
            if has_depth.any():
                mask3 = has_depth[:, :, None]
                blended = np.where(
                    mask3,
                    (
                        (
                            d_color_rgb.astype(np.uint16) * 102
                            + rgb.astype(np.uint16) * 154
                        )
                        >> 8
                    ).astype(np.uint8),
                    rgb,
                )
                self._upload_rgb_texture(blended)
            else:
                self._upload_rgb_texture(rgb)

        def _reupload_rgb_with_sdp_patches(self):
            """Re-upload RGB with SDP patch median depth overlay (16x16 patches)."""
            if self.total_frames == 0:
                return
            ts = self.sorted_timestamps[self.current_frame_idx]
            rgb = self._load_rgb_for_timestamp(ts)
            if rgb is None:
                return

            cam, T_wr = self._get_cam_and_pose(ts)
            if cam is None or T_wr is None:
                self._upload_rgb_texture(rgb)
                return

            sdp_w = self._get_sdp_for_timestamp(ts)
            if len(sdp_w) == 0:
                self._upload_rgb_texture(rgb)
                return

            # Scale camera to boxernet resolution
            bxr_hw = boxernet.hw
            patch_size = boxernet.patch_size
            cam_data = cam._data.clone()
            cam_scaled = CameraTW(cam_data)
            vrs_w = cam.size[..., 0].item()
            vrs_h = cam.size[..., 1].item()
            scale_x = bxr_hw / vrs_w
            scale_y = bxr_hw / vrs_h
            cam_scaled._data[0] = bxr_hw
            cam_scaled._data[1] = bxr_hw
            cam_scaled._data[2] *= scale_x
            cam_scaled._data[3] *= scale_y
            cam_scaled._data[4] *= scale_x
            cam_scaled._data[5] *= scale_y

            # Compute patch median depth
            sdp_patch = sdp_to_patches(
                sdp_w.unsqueeze(0).float(),
                cam_scaled.float(),
                T_wr.float(),
                bxr_hw,
                bxr_hw,
                patch_size,
            )  # (1, 1, fH, fW)

            rotated = not self._vrs_is_nebula
            HH, WW = rgb.shape[:2]
            viz_sdp_bgr, sdp_resized = render_depth_patches(
                sdp_patch[0].cpu(), rotated=rotated, HH=HH, WW=WW
            )
            viz_sdp = cv2.cvtColor(np.ascontiguousarray(viz_sdp_bgr), cv2.COLOR_BGR2RGB)
            mask = sdp_resized > 0.1
            if mask.any():
                mask3 = mask[:, :, None]
                # 20% overlay, 80% original (same as run_boxer.py)
                blended = np.where(
                    mask3,
                    (
                        (viz_sdp.astype(np.uint16) * 51 + rgb.astype(np.uint16) * 205)
                        >> 8
                    ).astype(np.uint8),
                    rgb,
                )
                self._upload_rgb_texture(blended)
            else:
                self._upload_rgb_texture(rgb)

        def on_key_event(self, key, action, modifiers):
            """Override to sync follow_view with play/pause."""
            # When imgui text input is focused, forward key to imgui but
            # don't process viewer shortcuts (space, arrows, etc.)
            # Only block viewer shortcuts when the OWL text input is active
            if getattr(self, "_owl_text_active", False):
                self.imgui.key_event(key, action, modifiers)
                return

            if key == self.wnd.keys.ESCAPE:
                if action == self.wnd.keys.ACTION_PRESS:
                    self.is_playing = False
                    if self.follow_view:
                        self._focus_on_current_frame()
                        self.follow_view = False
                return

            if action != self.wnd.keys.ACTION_PRESS:
                super().on_key_event(key, action, modifiers)
                return

            if key == self.wnd.keys.SPACE:
                if self.current_frame_idx >= self.total_frames - 1:
                    self._step_to_frame(0)
                    self.is_playing = True
                else:
                    self.is_playing = not self.is_playing
                if self.is_playing:
                    self._smooth_eye = None
                    self._smooth_target = None
                    self._smooth_up = None
                    self.follow_view = True
                else:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._last_step_time = time.time()
            elif key == self.wnd.keys.RIGHT:
                self.is_playing = False
                if self.follow_view:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._step_forward()
            elif key == self.wnd.keys.LEFT:
                self.is_playing = False
                if self.follow_view:
                    self._focus_on_current_frame()
                    self.follow_view = False
                self._step_to_frame(self.current_frame_idx - 1)
            else:
                super().on_key_event(key, action, modifiers)

        def _step_to_frame(self, target_idx: int) -> None:
            """Override to update prompted OBB projections on frame change."""
            super()._step_to_frame(target_idx)
            # Reload per-frame SDP for omni3d (each image has its own depth)
            # Also clear 3D BBs since each image has its own coordinate frame
            if dataset_type == "omni3d":
                self._upload_sdp_for_frame(self.current_frame_idx)
                self._prompted_obbs.clear()
                self._prompted_labels.clear()
                self._prompted_colors.clear()
                self._prompt_dirty = True
            if self.show_sdp_patches:
                self._reupload_rgb_with_sdp_patches()
            elif self.show_sdp_overlay:
                self._reupload_rgb_with_sdp()
            if self._prompted_obbs:
                self._update_prompted_rgb_projections()

    launch_viewer(PromptViewer)


if __name__ == "__main__":
    main()
