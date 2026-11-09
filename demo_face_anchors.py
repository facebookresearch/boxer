#! /usr/bin/env python3
"""RQ1 demo: face-anchored annotations + motion arrows on 3D bounding boxes.

Standalone demo built on Boxer's existing OrbitViewer (moderngl) stack.
No real data needed: three moving target OBBs, optional similar-looking
distractors and solid occluders are all generated in code.

Object-local coordinate convention:
    +X = right, +Y = front, +Z = up
So the six anchor faces are:
    top/bottom  -> +/-Z
    front/back  -> +/-Y
    right/left  -> +/-X

RQ1 variables exercised here:
    IV1: anchor face (front/back/left/right/top/bottom)
    IV2: scene context
        - similar-looking distractors (correspondence errors)
        - solid occluders (annotations hidden behind real geometry)
    Task info: speed (arrow length + numeric label), direction (arrow
    heading), trajectory (past trail + predicted future path), and
    distance travelled (numeric label, accumulated along the path).

Annotation consistency rule (important for the experiment):
    The motion arrow is the SAME 3D vector no matter which face it is
    anchored to. Its origin is pushed a fixed offset along the face
    normal, and it points along the true world-space velocity. So the
    only thing that changes across conditions is the anchor position,
    not the encoding.

Run:
    python demo_face_anchors.py
Drag to orbit, scroll to zoom, right-drag to pan.
"""

from collections import deque

import numpy as np

import moderngl_window as mglw

from utils.viewer_3d import OrbitViewer, _perspective_projection, _look_at
import utils.imgui_compat as imgui


# Face name -> (normal axis index, normal sign, tangent axis a, tangent axis b)
FACES = {
    "top":    (2, +1, 0, 1),
    "bottom": (2, -1, 0, 1),
    "front":  (1, +1, 0, 2),
    "back":   (1, -1, 0, 2),
    "right":  (0, +1, 1, 2),
    "left":   (0, -1, 1, 2),
}
FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]

# Moving targets.
#   center  : rest position xyz (z = resting height above ground)
#   size    : box size xyz
#   yaw     : rotation about Z in degrees
#   speed   : scalar speed in m/s (arrow length encodes this)
#   heading : unit direction in OBJECT-LOCAL xy ("forward" is +Y)
#   span    : half travel distance along the heading line (ping-pong)
TARGET_OBJECTS = [
    dict(center=[-2.0, -1.0, 0.40], size=[1.20, 0.80, 0.80], yaw=20.0,
         speed=0.55, heading=(0.30, 0.95), span=0.80),
    dict(center=[1.5, 1.0, 0.60], size=[1.00, 1.00, 1.20], yaw=-30.0,
         speed=0.95, heading=(-0.70, 0.70), span=0.55),
    dict(center=[2.5, -2.0, 0.30], size=[0.60, 0.60, 0.60], yaw=60.0,
         speed=0.30, heading=(1.00, 0.20), span=0.50),
]

# Static distractors: same wireframe look and matching sizes/shapes as the
# targets, so users can only tell which box a label belongs to via the
# anchoring link -- this is what drives correspondence-error trials.
SIMILAR_DISTRACTORS = [
    ([-3.4, 0.9, 0.40], [1.20, 0.80, 0.80], 10.0),
    ([0.1, 2.4, 0.60], [1.00, 1.00, 1.20], -50.0),
    ([3.4, -0.7, 0.30], [0.60, 0.60, 0.60], 25.0),
]

# Solid occluders, placed on the default camera ray (az 45, el 30) between
# camera and each target. Rendered as filled boxes so they actually hide
# annotations anchored on far-side faces.
OCCLUDERS = [
    ([-0.95, 0.05, 0.90], [0.50, 0.50, 1.80], 0.0),
    ([2.35, 1.85, 1.00], [0.70, 0.42, 2.00], 15.0),
    ([3.30, -1.20, 0.75], [0.80, 0.50, 1.50], -10.0),
]

CONTEXT_MODES = ["off", "similar distractors", "occluders", "both"]

ARROW_MIN = 0.20      # arrow shaft length at speed 0
ARROW_PER_SPEED = 0.85  # extra shaft length per (m/s)
HEAD_LEN = 0.12
HEAD_HALF = 0.07
MARK_HALF = 0.10      # half size of the anchor cross
ANCHOR_LIFT = 0.05    # fixed offset of arrow origin along face normal
TRAIL_MAX = 240       # trajectory samples per target
PREDICT_SECONDS = 1.0

BOX_COLOR = (0.55, 0.55, 0.60)          # targets AND similar distractors
BOX_HIGHLIGHT_COLOR = (0.80, 0.72, 0.35)
ANCHOR_COLOR = (1.0, 0.80, 0.15)
ARROW_COLOR = (0.95, 0.25, 0.20)
TRAIL_COLOR = (0.90, 0.50, 0.12)
PREDICT_COLOR = (0.55, 0.48, 0.32)
GRID_COLOR = (0.22, 0.22, 0.25)
OCCLUDER_FACE = (0.30, 0.27, 0.24)
OCCLUDER_EDGE = (0.48, 0.44, 0.38)


def yaw_rotation(yaw_deg):
    """3x3 rotation about Z."""
    t = np.radians(yaw_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def box_corners(center, size, R):
    """Eight corners of an OBB in world coordinates."""
    h = np.asarray(size) / 2.0
    local = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * h
    return (R @ local.T).T + np.asarray(center)


def box_edges(center, size, R):
    """12 line segments of an OBB in world coordinates -> list of (p0, p1)."""
    corners = box_corners(center, size, R)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(corners[a], corners[b]) for a, b in edges]


# Triangle indices, counter-clockwise as seen from OUTSIDE the box
# (base class enables back-face culling).
BOX_TRI_FACES = [
    (4, 5, 6), (4, 6, 7),   # +Z top
    (3, 0, 1), (3, 1, 2),   # -Z bottom
    (7, 3, 2), (7, 2, 6),   # +Y front
    (4, 0, 1), (4, 1, 5),   # -Y back
    (1, 2, 6), (1, 6, 5),   # +X right
    (0, 4, 7), (0, 7, 3),   # -X left
]


def face_anchor(center, size, R, face):
    """World position of the center of the given face, plus its world normal and tangents."""
    n_ax, n_sg, ta_ax, tb_ax = FACES[face]
    p_local = np.zeros(3)
    p_local[n_ax] = n_sg * size[n_ax] / 2.0
    pos = R @ p_local + np.asarray(center)

    normal_w = n_sg * R[:, n_ax]
    tangent_a = R[:, ta_ax]
    tangent_b = R[:, tb_ax]
    return pos, normal_w, tangent_a, tangent_b


class GeomBuilder:
    """Accumulate colored lines and triangles; flush into moderngl buffers."""

    def __init__(self):
        self.segs = []
        self.tris = []

    def segment(self, p0, p1, color):
        self.segs.append((np.asarray(p0, np.float32), np.asarray(p1, np.float32), color))

    def triangle(self, p0, p1, p2, color):
        self.tris.append((np.asarray(p0, np.float32), np.asarray(p1, np.float32),
                          np.asarray(p2, np.float32), color))

    def solid_box(self, center, size, R, face_color, edge_color=None):
        corners = box_corners(center, size, R)
        for a, b, c in BOX_TRI_FACES:
            self.triangle(corners[a], corners[b], corners[c], face_color)
        if edge_color is not None:
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]:
                self.segment(corners[a], corners[b], edge_color)

    def arrow(self, origin, direction, color, length,
              head_len=HEAD_LEN, head_half=HEAD_HALF):
        d = direction / (np.linalg.norm(direction) + 1e-9)
        head_len = min(head_len, length * 0.45)
        tip = origin + d * length
        self.segment(origin, tip, color)
        # Build an orthonormal side vector so the arrow head is 3D, not screen-aligned.
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(d, helper)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        side = np.cross(d, helper)
        side /= np.linalg.norm(side) + 1e-9
        for s in (+1, -1):
            base = tip - d * head_len + side * s * head_half
            self.segment(tip, base, color)

    def cross_mark(self, center, ta, tb, color, half=MARK_HALF):
        self.segment(center - ta * half, center + ta * half, color)
        self.segment(center - tb * half, center + tb * half, color)

    def line_array(self):
        if not self.segs:
            return np.zeros((0, 6), dtype="f4")
        rows = []
        for p0, p1, c in self.segs:
            rows.append([*p0, *c])
            rows.append([*p1, *c])
        return np.asarray(rows, dtype="f4")

    def tri_array(self):
        if not self.tris:
            return np.zeros((0, 9), dtype="f4")
        rows = []
        for p0, p1, p2, c in self.tris:
            rows.append([*p0, *c])
            rows.append([*p1, *c])
            rows.append([*p2, *c])
        return np.asarray(rows, dtype="f4")


class FaceAnchorDemo(OrbitViewer):
    title = "RQ1 demo: face anchors + motion arrows"

    def __init__(self, **kwargs):
        self.active_face = "top"
        self.show_all_faces = False
        self.animate = True
        self.show_trails = True
        self.show_prediction = False
        self.show_values = True
        self.highlight_targets = False
        self.context_mode = 0  # index into CONTEXT_MODES

        # Per-target runtime state.
        self.target_states = []
        for obj in TARGET_OBJECTS:
            R = yaw_rotation(obj["yaw"])
            hx, hy = obj["heading"]
            n = np.hypot(hx, hy)
            heading_local = np.array([hx / n, hy / n, 0.0])
            # Velocity direction in world coords (horizontal motion).
            dir_w = R @ heading_local
            self.target_states.append(dict(
                c0=np.asarray(obj["center"], np.float64),
                size=obj["size"],
                yaw=obj["yaw"],
                R=R,
                dir_w=dir_w,
                speed=obj["speed"],
                span=obj["span"],
                s=0.0,               # signed offset along dir_w
                sign=1.0,            # current travel direction
                center=np.asarray(obj["center"], np.float64),
                trail=deque(maxlen=TRAIL_MAX),
                distance=0.0,        # accumulated path length (m)
            ))

        super().__init__(**kwargs)
        self.camera_distance = 7.0
        self.camera_target = np.array([0.3, -0.3, 0.4], dtype=np.float32)

    def init_scene(self):
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 a_pos;
                in vec3 a_color;
                uniform mat4 u_mvp;
                out vec3 v_color;
                void main() {
                    gl_Position = u_mvp * vec4(a_pos, 1.0);
                    v_color = a_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() { f_color = vec4(v_color, 1.0); }
            """,
        )
        self.line_vbo = self.ctx.buffer(reserve=6 * 4 * 30000)
        self.tri_vbo = self.ctx.buffer(reserve=6 * 4 * 30000)
        self.line_vao = self.ctx.vertex_array(
            self.prog, [(self.line_vbo, "3f 3f", "a_pos", "a_color")]
        )
        self.tri_vao = self.ctx.vertex_array(
            self.prog, [(self.tri_vbo, "3f 3f", "a_pos", "a_color")]
        )

    def get_camera_matrices(self):
        w, h = self.wnd.size
        projection = _perspective_projection(45.0, w / max(h, 1), 0.1, 100.0)
        az, el = np.radians(self.camera_azimuth), np.radians(self.camera_elevation)
        eye = self.camera_target + np.array([
            self.camera_distance * np.cos(el) * np.cos(az),
            self.camera_distance * np.cos(el) * np.sin(az),
            self.camera_distance * np.sin(el),
        ])
        view = _look_at(tuple(eye), tuple(self.camera_target), (0, 0, 1))
        return projection, view, np.eye(4, dtype="f4") @ view @ projection

    def update_motion(self, dt):
        if not self.animate:
            return
        for st in self.target_states:
            ds = st["sign"] * st["speed"] * dt
            st["s"] += ds
            if st["s"] > st["span"]:
                st["s"] = 2 * st["span"] - st["s"]
                st["sign"] = -1.0
            elif st["s"] < -st["span"]:
                st["s"] = -2 * st["span"] - st["s"]
                st["sign"] = 1.0
            st["center"] = st["c0"] + st["dir_w"] * st["s"]
            st["trail"].append(st["center"].copy())
            st["distance"] += abs(ds)

    def world_to_screen(self, p, mvp):
        """Project a world point to window pixels. Returns None if behind camera."""
        clip = np.append(np.asarray(p, np.float64), 1.0) @ mvp
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        w, h = self.wnd.size
        return (
            (ndc[0] * 0.5 + 0.5) * w,
            (1.0 - (ndc[1] * 0.5 + 0.5)) * h,
        )

    def render_3d(self, time, frame_time):
        # clamp dt so a paused window / debugger does not teleport objects
        self.update_motion(min(frame_time, 0.05))

        lb = GeomBuilder()

        # Ground grid (1 m spacing, 8 x 8 m)
        for i in range(-4, 5):
            lb.segment([i, -4, 0], [i, 4, 0], GRID_COLOR)
            lb.segment([-4, i, 0], [4, i, 0], GRID_COLOR)

        mode = CONTEXT_MODES[self.context_mode]
        show_similar = mode in ("similar distractors", "both")
        show_occluders = mode in ("occluders", "both")

        # Similar-looking static distractors (wireframe, same color as targets).
        if show_similar:
            for center, size, yaw in SIMILAR_DISTRACTORS:
                for p0, p1 in box_edges(center, size, yaw_rotation(yaw)):
                    lb.segment(p0, p1, BOX_COLOR)

        # Solid occluders: filled faces so they hide far-side annotations.
        if show_occluders:
            for center, size, yaw in OCCLUDERS:
                lb.solid_box(center, size, yaw_rotation(yaw),
                             OCCLUDER_FACE, OCCLUDER_EDGE)

        faces_to_show = FACE_ORDER if self.show_all_faces else [self.active_face]
        target_color = BOX_HIGHLIGHT_COLOR if self.highlight_targets else BOX_COLOR
        labels = []

        for st in self.target_states:
            center, size, R = st["center"], st["size"], st["R"]

            # Target wireframe.
            for p0, p1 in box_edges(center, size, R):
                lb.segment(p0, p1, target_color)

            # Past trajectory trail.
            if self.show_trails and len(st["trail"]) >= 2:
                trail = list(st["trail"])
                for p0, p1 in zip(trail[:-1], trail[1:]):
                    lb.segment(p0, p1, TRAIL_COLOR)

            # Predicted future path along current velocity.
            if self.show_prediction:
                vel_world = st["dir_w"] * (st["sign"] * st["speed"])
                lb.segment(center, center + vel_world * PREDICT_SECONDS,
                          PREDICT_COLOR)

            # True 3D velocity vector: identical encoding on every face;
            # only the anchor origin changes across conditions.
            vel_world = st["dir_w"] * (st["sign"] * st["speed"])
            for face in faces_to_show:
                anchor, normal_w, ta, tb = face_anchor(center, size, R, face)
                arrow_origin = anchor + normal_w * ANCHOR_LIFT
                lb.cross_mark(anchor, ta, tb, ANCHOR_COLOR)
                arrow_len = ARROW_MIN + st["speed"] * ARROW_PER_SPEED
                lb.arrow(arrow_origin, vel_world, ARROW_COLOR, length=arrow_len)
                # Numeric label floats above the anchor.
                labels.append((anchor + normal_w * 0.28,
                               f"{st['speed']:.2f} m/s | d {st['distance']:.1f} m"))

        _, _, mvp = self.get_camera_matrices()

        # Upload and draw solid faces first, then lines (depth test handles
        # occlusion between them; ordering only matters within same depth).
        tri_data = lb.tri_array()
        self.tri_vbo.write(tri_data.tobytes())
        self.prog["u_mvp"].write(np.ascontiguousarray(mvp))
        if len(tri_data):
            self.tri_vao.render(mglw.constants.TRIANGLES, vertices=len(tri_data))

        line_data = lb.line_array()
        self.line_vbo.write(line_data.tobytes())
        if len(line_data):
            self.line_vao.render(mglw.constants.LINES, vertices=len(line_data))

        # Speed / distance labels in screen space.
        if self.show_values:
            draw = imgui.get_foreground_draw_list()
            for world_p, text in labels:
                sp = self.world_to_screen(world_p, mvp)
                if sp is None:
                    continue
                x, y = sp
                draw.add_text(imgui.ImVec2(x + 8, y - 8),
                              imgui.get_color_u32_rgba(1.0, 0.95, 0.75, 1.0),
                              text)

    def render_ui(self):
        imgui.begin("RQ1 demo controls")
        imgui.text("Annotations live in the OBJECT frame:")
        imgui.text("+X right, +Y front, +Z up")
        imgui.separator()
        current = FACE_ORDER.index(self.active_face)
        changed, current = imgui.combo(
            "Anchor face (IV1)", current, [f.capitalize() for f in FACE_ORDER]
        )
        if changed:
            self.active_face = FACE_ORDER[current]
        _changed, self.show_all_faces = imgui.checkbox(
            "Show all six face anchors (debug)", self.show_all_faces
        )
        imgui.separator()
        _changed, self.context_mode = imgui.combo(
            "Scene context (IV2)", self.context_mode, CONTEXT_MODES
        )
        imgui.separator()
        _changed, self.animate = imgui.checkbox("Play motion", self.animate)
        _changed, self.show_trails = imgui.checkbox("Past trajectory", self.show_trails)
        _changed, self.show_prediction = imgui.checkbox(
            "Predicted path (1 s)", self.show_prediction
        )
        _changed, self.show_values = imgui.checkbox(
            "Speed / distance labels", self.show_values
        )
        _changed, self.highlight_targets = imgui.checkbox(
            "Highlight targets (debug)", self.highlight_targets
        )
        imgui.separator()
        imgui.text("Yellow cross = face anchor point")
        imgui.text("Red arrow = velocity (length = speed)")
        imgui.text("Orange trail = past trajectory")
        imgui.end()


if __name__ == "__main__":
    mglw.run_window_config(FaceAnchorDemo)
