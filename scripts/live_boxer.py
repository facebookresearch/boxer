"""Live Aria Gen2 streaming + BoxerNet demo with interactive 3D viewer.

moderngl-window viewer with three regions:
  * Left:   ImGui control panel (sliders, toggles).
  * Center: Live RGB frame + OWLv2 2D bounding-box overlays.
  * Right:  Interactive 3D scene (orbit camera) with BoxerNet 3D OBBs and
            a camera frustum marker for the current device pose.

Press 'q' or Esc to quit. Right-drag to orbit, left-drag to pan, scroll to zoom.
"""

import argparse
import os
import platform
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
from owl.owl_wrapper import OwlWrapper
from utils.demo_utils import CKPT_PATH
from utils.image import draw_bb3s, put_text, render_bb2, torch2cv2
from utils.taxonomy import load_text_labels
from utils.tw.camera import CameraTW
from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.tw.pose import PoseTW
from utils.viewer_3d import OrbitViewer, launch_viewer


def ensure_aria_tools_on_path() -> None:
    aria_dir = os.path.dirname(os.path.abspath(aria.__file__))
    tools_dir = os.path.join(aria_dir, "tools")
    if not os.path.exists(os.path.join(tools_dir, "adb")):
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if tools_dir not in path_parts:
        os.environ["PATH"] = tools_dir + os.pathsep + os.environ.get("PATH", "")


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


class StreamState:
    """Thread-safe slots populated by streaming callbacks, read on main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame: Optional[tuple[np.ndarray, int]] = None
        self.T_world_rig: Optional[PoseTW] = None
        self.T_camera_rig: Optional[PoseTW] = None
        self.rgb_image_size: Optional[tuple[int, int]] = None
        self.rgb_intrinsics: Optional[list[float]] = None

    def snapshot(self):
        with self.lock:
            return (
                self.frame,
                self.T_world_rig,
                self.rgb_intrinsics,
                self.T_camera_rig,
                self.rgb_image_size,
            )


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


def build_cam(intrinsics, T_camera_rig, calib_image_size, target_hw):
    W, H = calib_image_size
    valid_radius = float(np.sqrt(W * W + H * H) / 2.0)
    cam = CameraTW.from_surreal(
        width=W,
        height=H,
        type_str="Fisheye624",
        params=torch.tensor(intrinsics, dtype=torch.float32),
        T_camera_rig=T_camera_rig,
        valid_radius=torch.tensor([valid_radius], dtype=torch.float32),
    ).float()
    return cam.scale_to_size((target_hw, target_hw))


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
):
    """Run one OWL+BoxerNet pass on the latest frame.

    Returns dict with keys: viz_2d_bgr, obb_pr_w, T_wr, cam, n_2d, n_3d, ts_ns
    Or None if no frame is available yet.
    """
    frame, T_wr, intr, T_cr, csize = state.snapshot()
    if frame is None or T_wr is None or intr is None or T_cr is None:
        return None

    arr_rgb, ts_ns = frame
    arr_resized = cv2.resize(arr_rgb, (HW, HW), interpolation=cv2.INTER_LINEAR)
    img_torch = torch.from_numpy(arr_resized).permute(2, 0, 1)[None].float() / 255.0
    cam = build_cam(intr, T_cr, csize, HW)

    bb2d, scores2d, label_ints, _ = owl.forward(
        img_torch * 255.0,
        rotated=False,
        resize_to_HW=(detector_hw, detector_hw),
    )
    labels2d = [text_labels[i] for i in label_ints]

    # Center panel: RGB with 2D and projected 3D overlays.
    viz_2d = torch2cv2(img_torch, rotate=False, ensure_rgb=True)
    if bb2d.shape[0] > 0:
        bb2_texts = [f"{l[:10]} {s:.2f}" for s, l in zip(scores2d, labels2d)]
        bb2_colors = jet_colors_bgr(scores2d)
        viz_2d = render_bb2(
            viz_2d,
            bb2d,
            scale=float(bb2_line_width),
            rotated=False,
            texts=bb2_texts,
            clr=bb2_colors,
        )
    put_text(viz_2d, f"OWLv2 {detector_hw}x{detector_hw}", scale=0.6, line=0)
    put_text(viz_2d, f"t={ts_ns / 1e9:.3f}s", scale=0.5, line=2)
    viz_3d = torch2cv2(img_torch, rotate=False, ensure_rgb=True)

    obb_pr_w = ObbTW(torch.zeros(0, 165))
    scores3d = torch.zeros(0)
    labels3d: list = []
    n_2d = bb2d.shape[0]
    n_3d = 0

    if n_2d > 0:
        datum = {
            "img0": img_torch,
            "cam0": cam,
            "T_world_rig0": T_wr,
            "rotated0": torch.tensor([False]),
            "sdp_w": torch.zeros(0, 3),
            "bb2d": bb2d,
        }
        if dev == "mps":
            out = boxernet.forward(datum)
        else:
            with torch.autocast(device_type=dev, dtype=pdtype):
                out = boxernet.forward(datum)
        obb_pr_w = out["obbs_pr_w"].cpu()[0]

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
            bb3_colors = jet_colors_bgr(scores3d.tolist())
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
                rotate_label=False,
                colors=bb3_colors,
                texts=bb3_texts,
                text_sz=0.35,
                thickness=bb3_line_width,
            )

    put_text(viz_3d, "Projected BoxerNet 3DBBs", scale=0.6, line=0)

    return {
        "viz_2d_bgr": viz_2d,
        "viz_3d_bgr": viz_3d,
        "obb_pr_w": obb_pr_w,
        "scores3d": scores3d,
        "labels3d": labels3d,
        "T_wr": T_wr,
        "cam": cam,
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
    window_size = (4000, 2000)

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

    # Layout
    ui_panel_width = 320
    rgb_panel_width = 960
    frustum_scale = 0.2

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

        # Per-frame caches
        self._last_ts = -1
        self._target_inited = False
        self._n_2d = 0
        self._n_3d = 0
        self._frame_count = 0
        self._frame_count_t0 = time.time()
        self._fps = 0.0

        # GL resources
        self._rgb_texture: Optional[moderngl.Texture] = None
        self._rgb_tex_size: Optional[tuple[int, int]] = None
        self._obb_vbo: Optional[moderngl.Buffer] = None
        self._obb_vao: Optional[moderngl.VertexArray] = None
        self._obb_count = 0
        self._frustum_vbo: Optional[moderngl.Buffer] = None
        self._frustum_vao: Optional[moderngl.VertexArray] = None
        self._frustum_count = 0

        # ImGui-controlled state
        self.thresh2d = float(self.owl.min_confidence)
        self.thresh3d = float(self.init_thresh3d)
        self.show_obbs_3d = True
        self.show_frustum = True
        self.bb2_line_width = 2
        self.bb3_image_line_width = 2
        self.line_width = 3.0
        self.frustum_line_width = 2.0

        # Better default viewing pose: look down at origin
        self.camera_distance = 4.0
        self.camera_azimuth = -90.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype="f4")

    # -- viewport / camera --

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        w, h = self.wnd.size
        vw = max(1, int(w - self.ui_panel_width - self.rgb_panel_width))
        return vw, h

    def _clamp_panel_widths(self) -> None:
        win_w, _ = self.wnd.size
        min_3d_width = 260
        self.ui_panel_width = float(np.clip(self.ui_panel_width, 240, 560))
        self.rgb_panel_width = float(np.clip(self.rgb_panel_width, 320, 1100))
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
        # Cheap snapshot to skip duplicate frames before doing real work
        with self.state.lock:
            frame = self.state.frame
        if frame is None:
            return
        if frame[1] == self._last_ts:
            return

        # Update OWL threshold from the slider before running
        self.owl.min_confidence = float(self.thresh2d)

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
        )
        if result is None:
            return

        self._last_ts = result["ts_ns"]
        self._n_2d = result["n_2d"]
        self._n_3d = result["n_3d"]

        separator = np.full((6, result["viz_2d_bgr"].shape[1], 3), 24, dtype=np.uint8)
        viz_bgr = np.vstack([result["viz_2d_bgr"], separator, result["viz_3d_bgr"]])
        rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
        self._upload_rgb_texture(rgb)

        # Right panel: rebuild 3D line geometry
        self._rebuild_obb_lines(result["obb_pr_w"], result["scores3d"])
        self._rebuild_frustum(result["cam"], result["T_wr"])

        # Lock orbit center on first frame
        if not self._target_inited:
            t = result["T_wr"].t.reshape(3).cpu().float().numpy().astype("f4")
            self.camera_target = t
            self._target_inited = True

        # FPS counter
        self._frame_count += 1
        now = time.time()
        if now - self._frame_count_t0 >= 1.0:
            self._fps = self._frame_count / (now - self._frame_count_t0)
            self._frame_count = 0
            self._frame_count_t0 = now

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

        rgb = jet_colors_rgb_float(scores.tolist())
        col = torch.tensor(rgb, dtype=torch.float32) if N > 0 else torch.zeros(0, 3)
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
        color = torch.tensor([0.0, 0.8, 0.8], dtype=torch.float32)
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

    # -- render --

    def on_render(self, time_val: float, frame_time: float):
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

        # Restore full viewport for ImGui
        self.ctx.viewport = (0, 0, full_w, full_h)
        self.ctx.scissor = None

    def render_ui(self) -> None:
        self._clamp_panel_widths()
        win_w, win_h = self.wnd.size

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
        imgui.text(f"2D detections: {self._n_2d}")
        imgui.text(f"3D detections: {self._n_3d}")
        imgui.separator()
        slider_w = max(160, int(self.ui_panel_width) - 28)
        imgui.push_item_width(slider_w)

        def labeled_slider_float(label, value, min_value, max_value, fmt="%.3f"):
            imgui.text(label)
            changed, value = imgui.slider_float(
                f"##{label}", value, min_value, max_value, fmt
            )
            return changed, value

        def labeled_slider_int(label, value, min_value, max_value):
            imgui.text(label)
            changed, value = imgui.slider_int(
                f"##{label}", value, min_value, max_value
            )
            return changed, value

        _, self.ui_panel_width = labeled_slider_float(
            "UI width", self.ui_panel_width, 240, 560, "%.0f"
        )
        _, self.rgb_panel_width = labeled_slider_float(
            "Image width", self.rgb_panel_width, 320, 1100, "%.0f"
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
        imgui.pop_item_width()
        imgui.separator()
        _, self.show_obbs_3d = imgui.checkbox("Show 3D OBBs", self.show_obbs_3d)
        _, self.show_frustum = imgui.checkbox("Show camera frustum", self.show_frustum)
        if imgui.button("Recenter on device"):
            self._target_inited = False
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
                avail_w, avail_h = imgui.get_content_region_available()
                scale = min(avail_w / tex_w, avail_h / tex_h)
                imgui.image(
                    self._rgb_texture.glo, tex_w * scale, tex_h * scale
                )
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
    p.add_argument("--image_hw", type=int, default=None)
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(CKPT_PATH, "boxernet_hw960in4x6d768-wssxpf9p.ckpt"),
    )
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument(
        "--force_precision", type=str, default=None, choices=["float32", "bfloat16"]
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

    if args.ip:
        target = sdk_gen2.DeviceTarget(ip=args.ip)
        target_desc = f"ip={args.ip}"
    elif args.serial:
        target = sdk_gen2.DeviceTarget(serial=args.serial)
        target_desc = f"serial={args.serial}"
    else:
        target = None
        target_desc = "auto-USB"

    if args.wifi and target is None:
        raise SystemExit(
            "--wifi requires --ip (preferred) or --serial; auto-discovery over WiFi is "
            "not available in this SDK build. Find the device IP in the Mobile "
            "Companion App."
        )

    print(f"==> Connecting to device ({target_desc})")
    device = (
        device_client.connect(target) if target is not None else device_client.connect()
    )

    sc = sdk_gen2.HttpStreamingConfig()
    sc.profile_name = args.profile_name
    sc.streaming_interface = (
        sdk_gen2.StreamingInterface.WIFI_STA
        if args.wifi
        else sdk_gen2.StreamingInterface.USB_NCM
    )
    print(f"==> Streaming interface: {sc.streaming_interface.name}")
    device.set_streaming_config(sc)
    device.start_streaming()

    state = StreamState()
    device_calib_cb, vio_cb, rgb_cb = make_callbacks(state)

    srv = sdk_gen2.HttpServerConfig()
    srv.address = "0.0.0.0"
    srv.port = 6768

    rx = receiver.StreamReceiver(enable_image_decoding=True, enable_raw_stream=False)
    rx.set_server_config(srv)
    rx.register_device_calib_callback(device_calib_cb)
    rx.register_vio_callback(vio_cb)
    rx.register_rgb_callback(rgb_cb)
    rx.start_server()

    try:
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
        LiveBoxerViewer.init_thresh3d = args.thresh3d
        LiveBoxerViewer.dev = dev
        LiveBoxerViewer.pdtype = pdtype

        print("==> Launching viewer.")
        launch_viewer(LiveBoxerViewer)
    finally:
        device.stop_streaming()
        rx.stop_server()


if __name__ == "__main__":
    main()
