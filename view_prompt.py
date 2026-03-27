#! /usr/bin/env python3

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

import imgui
import numpy as np
import torch

from boxernet.boxernet import BoxerNet
from utils.tw.camera import CameraTW
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import find_nearest2
from utils.demo_utils import CKPT_PATH
from utils.viewer import (
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    SequenceOBBViewer,
)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Interactive 2D BB prompting with BoxerNet")
    add_common_args(parser)
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_PATH, "bxr_alln2nw12bs12hw960in2x6d768ni1_Nov20.ckpt"), help="BoxerNet checkpoint")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"])
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
    precision_dtype = (
        torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    )

    # We need an empty timed_obbs to satisfy SequenceOBBViewer
    empty_timed_obbs = {}
    # Use first timestamp from seq_ctx to create a single entry
    rgb_ts = seq_ctx["rgb_timestamps"]
    if len(rgb_ts) > 0:
        empty_timed_obbs[int(rgb_ts[0])] = ObbTW(torch.zeros(0, 165))

    default_w, default_h = 2250, 1100
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class PromptViewer(SequenceOBBViewer):
        title = "BoxerNet Prompt Viewer"
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            self._prompted_obbs: list[ObbTW] = []
            self._prompted_labels: list[str] = []
            self._drawing = False
            self._draw_start: tuple[float, float] | None = None
            self._draw_end: tuple[float, float] | None = None
            self._draw_start_screen: tuple[float, float] | None = None
            self._draw_end_screen: tuple[float, float] | None = None
            # Image region in screen coords: (x, y, w, h)
            self._img_screen_rect: tuple[float, float, float, float] | None = None
            self._prompt_label = "object"
            self._prompt_dirty = False  # flag to rebuild geometry

            # Store SDP data from seq_ctx
            self._sdp_time_to_uids_slaml = seq_ctx.get("time_to_uids_slaml", None)
            self._sdp_time_to_uids_slamr = seq_ctx.get("time_to_uids_slamr", None)
            self._sdp_uid_to_p3 = seq_ctx.get("uid_to_p3", None)
            self._sdp_loader = seq_ctx.get("loader", None)

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

        def _screen_to_image_coords(self, sx, sy):
            """Convert screen coords to image pixel coords (in VRS resolution)."""
            if self._img_screen_rect is None:
                return None
            ix, iy, iw, ih = self._img_screen_rect
            # Check bounds
            if sx < ix or sx > ix + iw or sy < iy or sy > iy + ih:
                return None
            # Normalize within image rect
            u_norm = (sx - ix) / iw
            v_norm = (sy - iy) / ih
            # Convert to VRS pixel coords (before _rgb_img_scale resize)
            img_u = u_norm * self._rgb_vrs_w
            img_v = v_norm * self._rgb_vrs_h

            # If image was rotated (Aria Gen 1: rot90 k=3 applied in _load_rgb_for_timestamp)
            if not self._vrs_is_nebula:
                # rot90 k=3 maps (x, y) in rotated image to (y, W-1-x) in original
                # Rotated image has swapped dims: displayed (w, h) = (vrs_h, vrs_w)
                orig_u = img_v
                orig_v = self._rgb_vrs_w - 1 - img_u
                img_u, img_v = orig_u, orig_v

            return img_u, img_v

        def on_mouse_press_event(self, x, y, button):
            # Check if click is in the image panel area
            if button == 1:  # left click
                coords = self._screen_to_image_coords(x, y)
                if coords is not None:
                    self._drawing = True
                    self._draw_start = coords
                    self._draw_end = coords
                    self._draw_start_screen = (x, y)
                    self._draw_end_screen = (x, y)
                    return
            super().on_mouse_press_event(x, y, button)

        def on_mouse_drag_event(self, x, y, dx, dy):
            if self._drawing:
                coords = self._screen_to_image_coords(x, y)
                if coords is not None:
                    self._draw_end = coords
                self._draw_end_screen = (x, y)
                return
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
                    # Ensure min/max ordering
                    xmin = min(u0, u1)
                    xmax = max(u0, u1)
                    ymin = min(v0, v1)
                    ymax = max(v0, v1)

                    # Skip tiny boxes (accidental clicks)
                    if (xmax - xmin) > 5 and (ymax - ymin) > 5:
                        self._run_boxernet_prompt(xmin, xmax, ymin, ymax)

                self._draw_start = None
                self._draw_end = None
                self._draw_start_screen = None
                self._draw_end_screen = None
                return
            super().on_mouse_release_event(x, y, button)

        def _get_sdp_for_timestamp(self, ts_ns):
            """Get semi-dense world points for the given timestamp."""
            loader = self._sdp_loader
            if loader is None:
                return torch.zeros(0, 3, dtype=torch.float32)

            # For Aria: use combined time_to_uids maps
            if hasattr(loader, "time_to_uids_combined") and hasattr(loader, "p3_array"):
                sdp_times = loader.sdp_times_combined
                if len(sdp_times) == 0:
                    return torch.zeros(0, 3, dtype=torch.float32)
                nearest_idx = find_nearest2(sdp_times, ts_ns)
                sdp_ns = sdp_times[nearest_idx]
                uids = loader.time_to_uids_combined[sdp_ns]
                indices = [loader.uid_to_idx[uid] for uid in uids]
                p3d = torch.from_numpy(loader.p3_array[indices, :3])
                return p3d

            return torch.zeros(0, 3, dtype=torch.float32)

        def _run_boxernet_prompt(self, xmin, xmax, ymin, ymax):
            """Run BoxerNet on the drawn 2D BB and add result to prompted_obbs."""
            if self.total_frames == 0:
                return

            ts_ns = self.sorted_timestamps[self.current_frame_idx]
            cam, T_wr = self._get_cam_and_pose(ts_ns)
            if cam is None or T_wr is None:
                print("No camera/pose available for this frame")
                return

            # Load image
            rgb = self._load_rgb_for_timestamp(ts_ns)
            if rgb is None:
                print("No RGB image for this frame")
                return

            # The loaded rgb has been scaled by _rgb_img_scale and possibly rotated.
            # We need the original VRS image. Re-load from loader.
            img_np = self._load_raw_image(ts_ns)
            if img_np is None:
                print("Failed to load raw image")
                return

            H, W = img_np.shape[:2]

            # Resize image to BoxerNet resolution
            bnet_hw = boxernet.hw
            import cv2
            img_resized = cv2.resize(img_np, (bnet_hw, bnet_hw))
            img_torch = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

            # Scale bb2d coords from original image to boxernet resolution
            scale_x = bnet_hw / W
            scale_y = bnet_hw / H
            bb2d = torch.tensor(
                [[xmin * scale_x, xmax * scale_x, ymin * scale_y, ymax * scale_y]],
                dtype=torch.float32,
            )

            # Scale camera intrinsics
            cam_data = cam._data.clone()
            cam_scaled = CameraTW(cam_data)
            cam_scaled._data[0] = bnet_hw  # width
            cam_scaled._data[1] = bnet_hw  # height
            cam_scaled._data[2] *= scale_x  # fx
            cam_scaled._data[3] *= scale_y  # fy
            cam_scaled._data[4] *= scale_x  # cx
            cam_scaled._data[5] *= scale_y  # cy

            # Get SDP
            sdp_w = self._get_sdp_for_timestamp(ts_ns)

            # Determine rotation
            rotated = not self._vrs_is_nebula

            # Build datum
            datum = {
                "img0": img_torch[None],
                "cam0": cam_scaled.float(),
                "T_world_rig0": T_wr.float(),
                "rotated0": torch.tensor([rotated]),
                "sdp_w": sdp_w.float(),
                "bb2d": bb2d,
            }

            # Move to device
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to(device)
                elif hasattr(v, "_data"):
                    datum[k] = v.to(device)

            # Run inference
            print(f"Running BoxerNet on bb2d={bb2d[0].tolist()}")
            try:
                if device == "mps":
                    outputs = boxernet.forward(datum)
                else:
                    with torch.autocast(device_type=device, dtype=precision_dtype):
                        outputs = boxernet.forward(datum)
                obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
                if len(obb_pr_w) > 0:
                    prob = obb_pr_w.prob.squeeze(-1)
                    print(f"  -> Predicted {len(obb_pr_w)} 3D box(es), conf={prob.tolist()}")
                    for i in range(len(obb_pr_w)):
                        self._prompted_obbs.append(obb_pr_w[i])
                        self._prompted_labels.append(self._prompt_label)
                    self._prompt_dirty = True
                else:
                    print("  -> No 3D box predicted")
            except Exception as e:
                print(f"  -> BoxerNet error: {e}")

        def _load_raw_image(self, ts_ns):
            """Load the raw (unscaled, unrotated) RGB image for a timestamp."""
            if getattr(self, "_data_source", None) == "aria" and self._loader is not None:
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
                import cv2
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
            elif getattr(self, "_rgb_images", None) is not None:
                idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
                img = self._rgb_images[idx]
                if img is not None and img.ndim == 2:
                    import cv2
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return img
            return None

        def render_ui(self) -> None:
            """Render UI with drawing overlay and prompt controls."""
            # Rebuild prompted OBB geometry if dirty
            if self._prompt_dirty:
                self._rebuild_prompted_geometry()
                self._prompt_dirty = False

            super().render_ui()

            # Draw rectangle overlay on the RGB panel
            if self._rgb_texture is not None and self.show_rgb:
                draw_list = imgui.get_foreground_draw_list()

                # Draw current rectangle being drawn
                if self._drawing and self._draw_start_screen and self._draw_end_screen:
                    sx0, sy0 = self._draw_start_screen
                    sx1, sy1 = self._draw_end_screen
                    green = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 0.8)
                    draw_list.add_rect(sx0, sy0, sx1, sy1, green, thickness=2.0)

            # Prompt controls in a small window
            win_w, win_h = self.wnd.size
            ctrl_w = 250
            ctrl_h = 150
            ctrl_x = self.ui_panel_width
            ctrl_y = win_h - ctrl_h - 10
            imgui.set_next_window_position(ctrl_x, ctrl_y, imgui.ONCE)
            imgui.set_next_window_size(ctrl_w, ctrl_h, imgui.ONCE)
            expanded, _ = imgui.begin("Prompt Controls")
            if expanded:
                imgui.text(f"Prompted boxes: {len(self._prompted_obbs)}")
                imgui.text("Draw a box on the image")
                imgui.text("to prompt BoxerNet")
                imgui.separator()
                changed, self._prompt_label = imgui.input_text(
                    "Label", self._prompt_label, 64
                )
                if imgui.button("Clear All"):
                    self._prompted_obbs.clear()
                    self._prompted_labels.clear()
                    self._prompt_dirty = True
                if len(self._prompted_obbs) > 0:
                    imgui.same_line()
                    if imgui.button("Undo Last"):
                        self._prompted_obbs.pop()
                        self._prompted_labels.pop()
                        self._prompt_dirty = True
            imgui.end()

        def _rebuild_prompted_geometry(self):
            """Rebuild 3D geometry for prompted OBBs."""
            if len(self._prompted_obbs) == 0:
                # Clear the tracked_all set
                self._set_frame_obb_sets(
                    raw=self._empty_obbs_like(),
                    tracked_all=self._empty_obbs_like(),
                    tracked_visible=self._empty_obbs_like(),
                )
                return

            # Stack all prompted OBBs
            stacked = ObbTW(torch.stack([o._data for o in self._prompted_obbs]))
            self._set_frame_obb_sets(
                raw=self._empty_obbs_like(),
                tracked_all=stacked,
                tracked_visible=self._empty_obbs_like(),
            )

        def _render_main_controls(self):
            """Minimal controls for prompt mode."""
            pass

        def _render_common_visual_controls(self, **kw):
            super()._render_common_visual_controls(
                tracked_all_checkbox_label="Show Prompted 3DBBs",
                show_visible_checkbox=False,
                show_sets_header=False,
            )

        def render_3d(self, time_val: float, frame_time: float) -> None:
            """Override to track the image panel rect for mouse hit testing."""
            super().render_3d(time_val, frame_time)

            # After rendering, capture the image panel rect from ImGui
            # The image is rendered in the "RGB View" window by SequenceOBBViewer.render_ui()
            # We need to know its screen-space rect for mouse coordinate mapping
            if self._rgb_texture is not None and self.show_rgb:
                tex_w, tex_h = self._rgb_tex_size
                win_w, win_h = self.wnd.size
                panel_w = self._compute_rgb_panel_width(win_w, win_h)
                panel_x = self.ui_panel_width

                # The image is drawn with imgui.image() inside the panel
                # Account for the ImGui window title bar and padding
                title_bar_h = 20  # approximate
                padding = 8
                avail_w = panel_w - 2 * padding
                avail_h = win_h - title_bar_h - 2 * padding
                img_scale = min(avail_w / tex_w, avail_h / tex_h)
                draw_w = tex_w * img_scale
                draw_h = tex_h * img_scale

                img_x = panel_x + padding
                img_y = title_bar_h + padding
                self._img_screen_rect = (img_x, img_y, draw_w, draw_h)

    launch_viewer(PromptViewer)


if __name__ == "__main__":
    main()
