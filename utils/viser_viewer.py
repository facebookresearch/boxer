# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Browser-based Boxer viewers powered by viser.

This module provides a lightweight remote visualization backend that can run
alongside the existing local OpenGL/imgui viewers.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils.tw.obb import BB3D_LINE_ORDERS, ObbTW
from utils.tw.tensor_utils import find_nearest2


SCENE_ROOT = "/boxer/world"


def _require_viser():
    try:
        import viser
        import viser.transforms as tf

        return viser, tf
    except ImportError as exc:
        raise ImportError(
            "viser is required for remote visualization. "
            "Install with: pip install 'viser>=0.2.23'"
        ) from exc


def _empty_segments() -> np.ndarray:
    return np.zeros((0, 2, 3), dtype=np.float32)


def _empty_colors() -> np.ndarray:
    return np.zeros((0, 2, 3), dtype=np.float32)


def _empty_obbs() -> ObbTW:
    return ObbTW(torch.zeros(0, 165))


def _as_2d_obbs(obbs: Optional[ObbTW]) -> ObbTW:
    """Normalize OBB wrapper shape to (N, 165)."""
    if obbs is None or not isinstance(obbs, ObbTW) or obbs._data is None:
        return _empty_obbs()

    data = obbs._data
    if data.ndim == 1:
        if int(data.shape[0]) != 165:
            return _empty_obbs()
        return ObbTW(data.reshape(1, 165))

    if int(data.shape[-1]) != 165:
        return _empty_obbs()

    return ObbTW(data.reshape(-1, 165))


def _stack_obbs(items: List) -> ObbTW:
    """Stack a list of OBB-like items into an ObbTW of shape (N, 165)."""
    if items is None or len(items) == 0:
        return _empty_obbs()

    chunks: List[torch.Tensor] = []
    for item in items:
        if isinstance(item, ObbTW):
            obb = _as_2d_obbs(item)
            if len(obb) > 0 and obb._data is not None:
                chunks.append(obb._data)
            continue

        if isinstance(item, torch.Tensor):
            data = item
            if data.ndim == 1 and int(data.shape[0]) == 165:
                chunks.append(data.reshape(1, 165))
            elif data.ndim >= 2 and int(data.shape[-1]) == 165:
                chunks.append(data.reshape(-1, 165))

    if len(chunks) == 0:
        return _empty_obbs()

    return ObbTW(torch.cat(chunks, dim=0))


def _label_color(label: str) -> np.ndarray:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    rgb = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
    rgb = 70.0 + (rgb / 255.0) * 185.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _prob_color(prob: float) -> np.ndarray:
    # Blue -> Green -> Yellow -> Red
    p = float(np.clip(prob, 0.0, 1.0))
    if p < 0.33:
        t = p / 0.33
        rgb = np.array([0.0, 100.0 + 155.0 * t, 255.0 - 155.0 * t])
    elif p < 0.66:
        t = (p - 0.33) / 0.33
        rgb = np.array([255.0 * t, 255.0, 100.0 * (1.0 - t)])
    else:
        t = (p - 0.66) / 0.34
        rgb = np.array([255.0, 255.0 - 180.0 * t, 0.0])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _obbs_to_segments(
    obbs: ObbTW,
    conf_thresh: float,
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, str]]]:
    if obbs is None or len(obbs) == 0:
        return _empty_segments(), _empty_colors(), []

    probs = obbs.prob.squeeze(-1).detach().cpu().numpy()
    keep = probs >= conf_thresh
    if not np.any(keep):
        return _empty_segments(), _empty_colors(), []

    corners = obbs.bb3corners_world.detach().cpu().numpy()[keep]
    probs = probs[keep]

    sem_ids = obbs.sem_id.squeeze(-1).detach().cpu().numpy()[keep]
    text_labels = obbs.text_string()
    if isinstance(text_labels, str):
        text_labels = [text_labels]
    text_labels = [str(s).strip() for s in text_labels]
    text_labels = [text_labels[i] for i, k in enumerate(keep) if k]

    # Filter invalid or degenerate boxes to avoid NaN segments (which can hide
    # all lines in some viewers while labels may still appear).
    finite_mask = np.isfinite(corners).all(axis=(1, 2))
    spans = corners.max(axis=1) - corners.min(axis=1)
    non_degenerate_mask = np.linalg.norm(spans, axis=1) > 1e-5
    valid_mask = finite_mask & non_degenerate_mask
    if not np.any(valid_mask):
        return _empty_segments(), _empty_colors(), []

    corners = corners[valid_mask]
    probs = probs[valid_mask]
    sem_ids = sem_ids[valid_mask]
    text_labels = [text_labels[i] for i, k in enumerate(valid_mask) if k]

    segments = []
    seg_colors = []
    labels_out: List[Tuple[np.ndarray, str]] = []

    for i in range(corners.shape[0]):
        c = corners[i]
        p = float(probs[i])

        label = text_labels[i] if i < len(text_labels) and text_labels[i] else f"sem_{int(sem_ids[i])}"
        if color_mode == "probability":
            box_color_u8 = _prob_color(p)
        else:
            box_color_u8 = _label_color(label)
        box_color_scene = box_color_u8.astype(np.float32) / 255.0

        for e0, e1 in BB3D_LINE_ORDERS:
            segments.append([c[e0], c[e1]])
            seg_colors.append([box_color_scene, box_color_scene])

        center = c.mean(axis=0)
        labels_out.append((center.astype(np.float32), f"{label} {p:.2f}"))

    return (
        np.asarray(segments, dtype=np.float32),
        np.asarray(seg_colors, dtype=np.float32),
        labels_out,
    )


def _trajectory_segments(seq_ctx: Optional[dict]) -> np.ndarray:
    if seq_ctx is None:
        return _empty_segments()
    traj = seq_ctx.get("traj", None)
    if traj is None or len(traj) < 2:
        return _empty_segments()

    pts = []
    for pose in traj:
        try:
            t = pose.t.reshape(-1).detach().cpu().numpy().astype(np.float32)
            pts.append(t)
        except Exception:
            continue
    if len(pts) < 2:
        return _empty_segments()

    pts_np = np.stack(pts, axis=0)
    segs = np.stack([pts_np[:-1], pts_np[1:]], axis=1)
    return segs.astype(np.float32)


def _get_cam_pose(seq_ctx: Optional[dict], ts_ns: int):
    if seq_ctx is None:
        return None, None

    pose_ts = seq_ctx.get("pose_ts", None)
    traj = seq_ctx.get("traj", None)
    calib_ts = seq_ctx.get("calib_ts", None)
    calibs = seq_ctx.get("calibs", None)
    if pose_ts is None or traj is None or calib_ts is None or calibs is None:
        return None, None

    try:
        pose_idx = int(find_nearest2(pose_ts, ts_ns))
        calib_idx = int(find_nearest2(calib_ts, ts_ns))
        T_wr = traj[pose_idx].float()
        cam = calibs[calib_idx].float()
        return cam, T_wr
    except Exception:
        return None, None


def _load_raw_rgb_image(seq_ctx: Optional[dict], ts_ns: int) -> Optional[np.ndarray]:
    if seq_ctx is None:
        return None

    source = seq_ctx.get("source", "")
    loader = seq_ctx.get("loader", None)
    timestamps = seq_ctx.get("rgb_timestamps", None)
    if loader is None or timestamps is None or len(timestamps) == 0:
        return None

    try:
        if source == "aria":
            stream_id = loader.stream_id[0]
            calibs = loader.calibs[0]
            if hasattr(loader, "_find_frame_by_timestamp"):
                frame_idx = int(loader._find_frame_by_timestamp(int(ts_ns)))
            else:
                idx = int(find_nearest2(timestamps, ts_ns))
                frame_idx = idx

            out = loader._single(frame_idx, stream_id, calibs)
            if out is False or "img" not in out:
                return None
            img_t = out["img"][0].permute(1, 2, 0).cpu().numpy()
            return np.clip(img_t * 255.0, 0, 255).astype(np.uint8)

        idx = int(find_nearest2(timestamps, ts_ns))
        datum = loader.load(idx)
        if datum is False or "img0" not in datum:
            return None
        img_t = datum["img0"][0].permute(1, 2, 0).cpu().numpy()
        return np.clip(img_t * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        return None


def _to_display_rgb(seq_ctx: Optional[dict], raw_img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if raw_img is None:
        return None
    if seq_ctx is None:
        return raw_img
    source = seq_ctx.get("source", "")
    if source == "aria" and not bool(seq_ctx.get("is_nebula", True)):
        return np.rot90(raw_img, k=3).copy()
    return raw_img


def _load_rgb_image(seq_ctx: Optional[dict], ts_ns: int) -> Optional[np.ndarray]:
    return _to_display_rgb(seq_ctx, _load_raw_rgb_image(seq_ctx, ts_ns))


def _transform_boxes_raw_to_display(boxes_xyxy: np.ndarray, raw_w: int) -> np.ndarray:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    out = np.zeros_like(boxes_xyxy, dtype=np.float32)
    for i in range(len(boxes_xyxy)):
        x1, x2, y1, y2 = boxes_xyxy[i]
        corners = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        disp_x = (raw_w - 1.0) - corners[:, 1]
        disp_y = corners[:, 0]
        out[i, 0] = float(np.min(disp_x))
        out[i, 1] = float(np.max(disp_x))
        out[i, 2] = float(np.min(disp_y))
        out[i, 3] = float(np.max(disp_y))
    return out


def _load_sdp(seq_ctx: Optional[dict], ts_ns: int) -> torch.Tensor:
    if seq_ctx is None:
        return torch.zeros(0, 3, dtype=torch.float32)

    loader = seq_ctx.get("loader", None)
    if loader is None:
        return torch.zeros(0, 3, dtype=torch.float32)

    try:
        if (
            hasattr(loader, "time_to_uids_combined")
            and hasattr(loader, "p3_array")
            and hasattr(loader, "sdp_times_combined")
        ):
            sdp_times = loader.sdp_times_combined
            if len(sdp_times) == 0:
                return torch.zeros(0, 3, dtype=torch.float32)
            sdp_idx = int(find_nearest2(sdp_times, ts_ns))
            sdp_ns = int(sdp_times[sdp_idx])
            uids = loader.time_to_uids_combined[sdp_ns]
            if len(uids) == 0:
                return torch.zeros(0, 3, dtype=torch.float32)
            indices = [loader.uid_to_idx[uid] for uid in uids if uid in loader.uid_to_idx]
            if len(indices) == 0:
                return torch.zeros(0, 3, dtype=torch.float32)
            return torch.from_numpy(loader.p3_array[indices, :3]).float()

        timestamps = seq_ctx.get("rgb_timestamps", None)
        if timestamps is None or len(timestamps) == 0:
            return torch.zeros(0, 3, dtype=torch.float32)
        idx = int(find_nearest2(timestamps, ts_ns))
        datum = loader.load(idx)
        if datum is False or "sdp_w" not in datum:
            return torch.zeros(0, 3, dtype=torch.float32)
        sdp = datum["sdp_w"]
        if isinstance(sdp, torch.Tensor):
            if sdp.ndim == 2 and sdp.shape[-1] == 3:
                return sdp.float()
            if sdp.ndim > 2:
                return sdp.reshape(-1, 3).float()
        return torch.zeros(0, 3, dtype=torch.float32)
    except Exception:
        return torch.zeros(0, 3, dtype=torch.float32)


def _subsample_points_np(points: np.ndarray, max_points: int) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points) <= max_points:
        return points.astype(np.float32)
    rng = np.random.default_rng(0)
    ids = rng.choice(len(points), size=max_points, replace=False)
    return points[ids].astype(np.float32)


def _extract_static_scene_points(
    seq_ctx: Optional[dict],
    max_points: int = 120000,
) -> Tuple[np.ndarray, np.ndarray]:
    if seq_ctx is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points = None
    loader = seq_ctx.get("loader", None)
    if loader is not None and hasattr(loader, "p3_array"):
        arr = np.asarray(loader.p3_array)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            points = arr[:, :3]

    if points is None:
        sdp_global = seq_ctx.get("sdp_global", None)
        if sdp_global is not None and len(sdp_global) > 0:
            points = np.asarray(sdp_global)[:, :3]

    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    points = _subsample_points_np(points, max_points=max_points)
    colors = np.full((len(points), 3), 0.78, dtype=np.float32)
    return points, colors


def _collect_initial_view_points(
    timed_obbs: Dict[int, ObbTW],
    timeline_ts: List[int],
    seq_ctx: Optional[dict],
) -> np.ndarray:
    chunks: List[np.ndarray] = []

    if seq_ctx is not None:
        traj = seq_ctx.get("traj", None)
        if traj is not None and len(traj) > 0:
            traj_points: List[np.ndarray] = []
            stride = max(1, len(traj) // 400)
            for pose in traj[::stride]:
                try:
                    t = pose.t.reshape(-1).detach().cpu().numpy().astype(np.float32)
                except Exception:
                    continue
                if int(t.shape[0]) >= 3 and bool(np.isfinite(t[:3]).all()):
                    traj_points.append(t[:3])
            if len(traj_points) > 0:
                chunks.append(np.stack(traj_points, axis=0))

        static_points, _ = _extract_static_scene_points(seq_ctx, max_points=10000)
        if len(static_points) > 0:
            chunks.append(static_points.astype(np.float32))

    if len(timeline_ts) > 0:
        n_frames = min(len(timeline_ts), 60)
        frame_ids = np.linspace(0, len(timeline_ts) - 1, num=n_frames, dtype=np.int64)
        centers: List[np.ndarray] = []
        for frame_id in frame_ids.tolist():
            ts = int(timeline_ts[int(frame_id)])
            obbs = _as_2d_obbs(timed_obbs.get(ts, _empty_obbs()))
            if len(obbs) == 0:
                continue
            try:
                probs = obbs.prob.squeeze(-1).detach().cpu().numpy()
                keep = probs >= 0.25
                if not bool(np.any(keep)):
                    keep = probs >= 0.0

                corners = obbs.bb3corners_world.detach().cpu().numpy()
                if int(corners.shape[0]) == int(keep.shape[0]):
                    corners = corners[keep]
                if len(corners) == 0:
                    continue

                c = corners.mean(axis=1).astype(np.float32)
                c = c[np.isfinite(c).all(axis=1)]
                if len(c) > 0:
                    centers.append(_subsample_points_np(c, max_points=120))
            except Exception:
                continue

        if len(centers) > 0:
            chunks.append(np.concatenate(centers, axis=0))

    if len(chunks) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(chunks, axis=0).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return _subsample_points_np(points, max_points=20000)


def _estimate_initial_camera_pose(
    timed_obbs: Dict[int, ObbTW],
    timeline_ts: List[int],
    seq_ctx: Optional[dict],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    points = _collect_initial_view_points(
        timed_obbs=timed_obbs,
        timeline_ts=timeline_ts,
        seq_ctx=seq_ctx,
    )
    if len(points) == 0:
        return None

    low = np.percentile(points, 2.0, axis=0).astype(np.float32)
    high = np.percentile(points, 98.0, axis=0).astype(np.float32)
    span = high - low

    if (not bool(np.isfinite(span).all())) or float(np.max(span)) < 1e-4:
        low = np.min(points, axis=0).astype(np.float32)
        high = np.max(points, axis=0).astype(np.float32)
        span = high - low

    span = np.maximum(span, 1e-3)
    center = ((low + high) * 0.5).astype(np.float32)

    # Infer a plausible vertical axis from the tightest span; this keeps
    # initial up-direction reasonable across z-up/y-up datasets.
    anisotropy = float(np.max(span) / max(np.min(span), 1e-6))
    up_axis = 2 if anisotropy < 1.35 else int(np.argmin(span))
    horiz_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
    h_major = horiz_axes[int(np.argmax(span[horiz_axes]))]
    h_minor = horiz_axes[0] if horiz_axes[0] != h_major else horiz_axes[1]

    view_dir = np.zeros(3, dtype=np.float32)
    view_dir[h_major] = 1.20
    view_dir[h_minor] = -0.85
    view_dir[up_axis] = 0.65
    view_norm = float(np.linalg.norm(view_dir))
    if view_norm < 1e-6:
        view_dir = np.array([1.0, -1.0, 0.7], dtype=np.float32)
        view_norm = float(np.linalg.norm(view_dir))
    view_dir = view_dir / max(view_norm, 1e-6)

    up_dir = np.zeros(3, dtype=np.float32)
    up_dir[up_axis] = 1.0

    scene_diag = float(np.linalg.norm(span))
    cam_dist = max(1.5, 1.2 * scene_diag)
    position = (center + view_dir * cam_dist).astype(np.float32)
    look_at = center.astype(np.float32)

    near = float(max(0.01, cam_dist * 0.01))
    far = float(max(200.0, cam_dist * 8.0))
    return position, look_at, up_dir, near, far


def _project_obbs_for_rgb_overlay(
    obbs: ObbTW,
    cam,
    T_wr,
    *,
    img_w: int,
    img_h: int,
    is_nebula: bool,
    conf_thresh: float,
    color_mode: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]]:
    if obbs is None or len(obbs) == 0:
        return []

    probs_t = obbs.prob.squeeze(-1)
    keep_t = probs_t >= float(conf_thresh)
    if not bool(keep_t.any()):
        return []

    kept_obbs = obbs[keep_t]
    probs = probs_t[keep_t].detach().cpu().numpy()

    sem_ids = kept_obbs.sem_id.squeeze(-1).detach().cpu().numpy()
    text_labels = kept_obbs.text_string()
    if isinstance(text_labels, str):
        text_labels = [text_labels]
    text_labels = [str(s).strip() for s in text_labels]

    corners = kept_obbs.bb3corners_world
    n_boxes = int(len(kept_obbs))
    n_sub = 10
    edge_idx = torch.tensor(BB3D_LINE_ORDERS, dtype=torch.long, device=corners.device)

    p0 = corners[:, edge_idx[:, 0], :]
    p1 = corners[:, edge_idx[:, 1], :]
    t_interp = torch.linspace(0, 1, n_sub + 1, device=corners.device)
    edge_pts = (
        p0[:, :, None, :] * (1 - t_interp[None, None, :, None])
        + p1[:, :, None, :] * t_interp[None, None, :, None]
    )

    s = n_sub + 1
    pts_world = edge_pts.reshape(-1, 3)
    T_world_cam = T_wr @ cam.T_camera_rig.inverse()
    pts_cam = T_world_cam.inverse().transform(pts_world)

    proj_cam = cam
    try:
        cam_w = float(cam.size.reshape(-1).detach().cpu().numpy()[0])
        cam_h = float(cam.size.reshape(-1).detach().cpu().numpy()[1])
        if abs(cam_w - img_w) > 1.0 or abs(cam_h - img_h) > 1.0:
            proj_cam = cam.scale_to_size((img_w, img_h))
    except Exception:
        proj_cam = cam

    pts_2d, valid = proj_cam.project(
        pts_cam.unsqueeze(0),
        fov_deg=140.0 if is_nebula else 120.0,
    )
    pts_2d = pts_2d.squeeze(0).detach().cpu().numpy()
    valid = valid.squeeze(0).detach().cpu().numpy()

    if not is_nebula:
        try:
            proj_h = float(proj_cam.size.reshape(-1).detach().cpu().numpy()[1])
        except Exception:
            proj_h = float(img_h)
        old_x = pts_2d[:, 0].copy()
        old_y = pts_2d[:, 1].copy()
        pts_2d[:, 0] = float(proj_h - 1) - old_y
        pts_2d[:, 1] = old_x

    pts_2d = pts_2d.reshape(n_boxes, 12, s, 2)
    valid = valid.reshape(n_boxes, 12, s)

    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = []
    for i in range(n_boxes):
        if color_mode == "probability":
            color = _prob_color(float(probs[i]))
        else:
            label = (
                text_labels[i]
                if i < len(text_labels) and text_labels[i]
                else f"sem_{int(sem_ids[i])}"
            )
            color = _label_color(label)

        label = (
            text_labels[i]
            if i < len(text_labels) and text_labels[i]
            else f"sem_{int(sem_ids[i])}"
        )
        out.append((pts_2d[i], valid[i], color, label, float(probs[i])))
    return out


def _draw_rgb_overlays(
    image_rgb: np.ndarray,
    projected_obbs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]],
    owl_overlay: Optional[dict],
) -> np.ndarray:
    if image_rgb is None:
        return image_rgb

    img_h, img_w = image_rgb.shape[:2]
    out_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for edge_pts, edge_valid, color_rgb, label, prob in projected_obbs:
        bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        for e in range(edge_pts.shape[0]):
            for s in range(edge_pts.shape[1] - 1):
                if bool(edge_valid[e, s]) and bool(edge_valid[e, s + 1]):
                    x0 = int(np.clip(round(float(edge_pts[e, s, 0])), 0, img_w - 1))
                    y0 = int(np.clip(round(float(edge_pts[e, s, 1])), 0, img_h - 1))
                    x1 = int(np.clip(round(float(edge_pts[e, s + 1, 0])), 0, img_w - 1))
                    y1 = int(np.clip(round(float(edge_pts[e, s + 1, 1])), 0, img_h - 1))
                    cv2.line(out_bgr, (x0, y0), (x1, y1), bgr, 2, lineType=cv2.LINE_AA)

        if np.any(edge_valid):
            pts = edge_pts[edge_valid]
            cx = int(np.clip(round(float(np.mean(pts[:, 0]))), 0, img_w - 1))
            cy = int(np.clip(round(float(np.mean(pts[:, 1]))), 0, img_h - 1))
            cv2.putText(
                out_bgr,
                f"{label} {prob:.2f}",
                (cx, max(0, cy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bgr,
                1,
                cv2.LINE_AA,
            )

    if owl_overlay is not None:
        boxes = np.asarray(owl_overlay.get("boxes", np.zeros((0, 4))))
        scores = np.asarray(owl_overlay.get("scores", np.zeros((0,))))
        accepted = np.asarray(owl_overlay.get("accepted", np.zeros((0,), dtype=bool)))

        for i in range(len(boxes)):
            x1, x2, y1, y2 = boxes[i]
            xa = int(np.clip(round(float(min(x1, x2))), 0, img_w - 1))
            xb = int(np.clip(round(float(max(x1, x2))), 0, img_w - 1))
            ya = int(np.clip(round(float(min(y1, y2))), 0, img_h - 1))
            yb = int(np.clip(round(float(max(y1, y2))), 0, img_h - 1))

            is_accepted = bool(accepted[i]) if i < len(accepted) else False
            color = (30, 220, 30) if is_accepted else (50, 120, 255)
            cv2.rectangle(out_bgr, (xa, ya), (xb, yb), color, 2, lineType=cv2.LINE_AA)

            score = float(scores[i]) if i < len(scores) else 0.0
            tag = "3D" if is_accepted else "2D"
            cv2.putText(
                out_bgr,
                f"{tag} {score:.2f}",
                (xa, max(0, ya - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


class _ViserTimelineViewer:
    def __init__(
        self,
        timed_obbs: Dict[int, ObbTW],
        timeline_ts: List[int],
        *,
        seq_name: str,
        host: str,
        port: int,
        seq_ctx: Optional[dict] = None,
        title: str = "Boxer Viser Viewer",
        show_trajectory: bool = False,
        show_camera: bool = False,
        show_rgb: bool = False,
        seek_debounce_sec: float = 0.0,
        apply_conf_filter: bool = True,
    ):
        viser, tf = _require_viser()
        self.viser = viser
        self.tf = tf
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.seq_name = seq_name
        self.seq_ctx = seq_ctx
        self.timed_obbs = timed_obbs
        self.timeline_ts = sorted([int(x) for x in timeline_ts])
        if len(self.timeline_ts) == 0:
            raise ValueError("timeline is empty")

        self.seek_debounce_sec = float(max(0.0, seek_debounce_sec))
        self.apply_conf_filter = bool(apply_conf_filter)
        self._scene_update_lock = threading.RLock()
        self._pending_frame_idx = 0
        self._pending_frame_dirty = False
        self._pending_frame_event_ts = 0.0

        self._initial_camera_pose = _estimate_initial_camera_pose(
            timed_obbs=self.timed_obbs,
            timeline_ts=self.timeline_ts,
            seq_ctx=self.seq_ctx,
        )
        self._apply_initial_camera_pose()

        self._line_handle = self.server.scene.add_line_segments(
            f"{SCENE_ROOT}/obbs",
            points=_empty_segments(),
            colors=_empty_colors(),
            line_width=2.0,
        )
        self._line_node_version = 0
        self._traj_handle = None
        self._frustum_handle = None
        self._label_handles = []
        self._last_frame_idx = 0

        self.gui_title = self.server.gui.add_text("Scene", initial_value=title)
        self.gui_frame = self.server.gui.add_slider(
            "Image Index",
            min=1,
            max=len(self.timeline_ts),
            step=1,
            initial_value=1,
        )
        self.gui_play = self.server.gui.add_checkbox("Playing", initial_value=False)
        self.gui_fps = self.server.gui.add_slider(
            "Playback FPS", min=0.5, max=60.0, step=0.5, initial_value=10.0
        )
        self.gui_conf = self.server.gui.add_slider(
            "Confidence Threshold", min=0.0, max=1.0, step=0.01, initial_value=0.5
        )
        self.gui_line_width = self.server.gui.add_slider(
            "Line Width", min=1.0, max=8.0, step=0.5, initial_value=2.0
        )
        self.gui_color_mode = self.server.gui.add_dropdown(
            "Color Mode",
            options=["semantic", "probability"],
            initial_value="semantic",
        )
        self.gui_show_labels = self.server.gui.add_checkbox("Show Labels", initial_value=True)
        self.gui_show_traj = self.server.gui.add_checkbox(
            "Show Trajectory", initial_value=show_trajectory
        )
        self.gui_show_camera = self.server.gui.add_checkbox(
            "Show Current Camera", initial_value=show_camera
        )
        self.gui_show_rgb = self.server.gui.add_checkbox(
            "Show Current RGB", initial_value=show_rgb
        )
        self.gui_status = self.server.gui.add_text("Status", initial_value="Ready")

        self.gui_rgb = self.server.gui.add_image(
            np.zeros((24, 24, 3), dtype=np.uint8), label="Current RGB"
        )
        self.gui_rgb.visible = bool(show_rgb)

        @self.gui_frame.on_update
        def _(_evt) -> None:
            self._queue_frame_update(int(self.gui_frame.value) - 1)

        @self.gui_conf.on_update
        def _(_evt) -> None:
            self._flush_pending_frame_update(force=True)
            self._update_frame(self._last_frame_idx)

        @self.gui_color_mode.on_update
        def _(_evt) -> None:
            self._flush_pending_frame_update(force=True)
            self._update_frame(self._last_frame_idx)

        @self.gui_line_width.on_update
        def _(_evt) -> None:
            # Keep line updates in a single message to avoid per-property
            # update deduplication dropping points/colors.
            self._set_obb_line_segments(
                self._line_handle.points,
                self._line_handle.colors,
                float(self.gui_line_width.value),
            )

        @self.gui_show_labels.on_update
        def _(_evt) -> None:
            self._flush_pending_frame_update(force=True)
            self._update_frame(self._last_frame_idx)

        @self.gui_show_traj.on_update
        def _(_evt) -> None:
            self._refresh_trajectory()

        @self.gui_show_camera.on_update
        def _(_evt) -> None:
            self._flush_pending_frame_update(force=True)
            self._update_frame(self._last_frame_idx)

        @self.gui_show_rgb.on_update
        def _(_evt) -> None:
            self.gui_rgb.visible = bool(self.gui_show_rgb.value)
            self._flush_pending_frame_update(force=True)
            self._update_frame(self._last_frame_idx)

        self._refresh_trajectory()
        self._update_frame(0)

        self._last_play_tick = time.time()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def _refresh_trajectory(self) -> None:
        with self._scene_update_lock:
            if self._traj_handle is not None:
                try:
                    self._traj_handle.remove()
                except Exception:
                    pass
                self._traj_handle = None

            if not bool(self.gui_show_traj.value):
                return

            traj_segments = _trajectory_segments(self.seq_ctx)
            if len(traj_segments) == 0:
                return

            self._traj_handle = self.server.scene.add_line_segments(
                f"{SCENE_ROOT}/trajectory",
                points=traj_segments,
                colors=np.array([90, 180, 255], dtype=np.uint8),
                line_width=1.5,
            )

    def _queue_frame_update(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
        if self.seek_debounce_sec <= 0.0:
            self._update_frame(idx)
            return

        self._pending_frame_idx = idx
        self._pending_frame_dirty = True
        self._pending_frame_event_ts = time.time()
        if bool(self.gui_play.value):
            self._flush_pending_frame_update(force=True)

    def _flush_pending_frame_update(self, force: bool = False) -> None:
        if not self._pending_frame_dirty:
            return
        now = time.time()
        if force or bool(self.gui_play.value) or (now - self._pending_frame_event_ts >= self.seek_debounce_sec):
            self._pending_frame_dirty = False
            self._update_frame(self._pending_frame_idx)

    def _load_rgb_for_view(self, ts_ns: int) -> Optional[np.ndarray]:
        return _load_rgb_image(self.seq_ctx, ts_ns)

    def _apply_initial_camera_pose(self) -> None:
        if self._initial_camera_pose is None:
            return

        pos, look_at, up_dir, near, far = self._initial_camera_pose
        try:
            self.server.initial_camera.position = tuple(float(x) for x in pos.tolist())
            self.server.initial_camera.look_at = tuple(float(x) for x in look_at.tolist())
            self.server.initial_camera.up = tuple(float(x) for x in up_dir.tolist())
            self.server.initial_camera.near = float(near)
            self.server.initial_camera.far = float(far)
        except Exception:
            pass

    def _set_obb_line_segments(
        self,
        segments: np.ndarray,
        colors: np.ndarray,
        line_width: float,
    ) -> None:
        # Use unique node names for each refresh to avoid same-name
        # create/remove dedup races in the websocket buffer.
        with self._scene_update_lock:
            old_handle = self._line_handle
            self._line_node_version += 1
            node_name = f"{SCENE_ROOT}/obbs_{self._line_node_version}"
            self._line_handle = self.server.scene.add_line_segments(
                node_name,
                points=segments,
                colors=colors,
                line_width=float(line_width),
            )
            try:
                old_handle.remove()
            except Exception:
                pass

    def _update_frame(self, idx: int) -> None:
        with self._scene_update_lock:
            idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
            ts = int(self.timeline_ts[idx])
            self._last_frame_idx = idx

            obbs = self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
            segments, colors, labels = _obbs_to_segments(
                obbs,
                conf_thresh=(float(self.gui_conf.value) if self.apply_conf_filter else 0.0),
                color_mode=str(self.gui_color_mode.value),
            )

            self._set_obb_line_segments(
                segments,
                colors,
                float(self.gui_line_width.value),
            )

            for h in self._label_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._label_handles = []

            if bool(self.gui_show_labels.value):
                for center, label in labels:
                    self._label_handles.append(
                        self.server.scene.add_label(
                            name=f"{SCENE_ROOT}/labels/{len(self._label_handles)}",
                            text=label,
                            position=tuple(center.tolist()),
                            anchor="center-center",
                            font_screen_scale=0.9,
                        )
                    )

            if self._frustum_handle is not None:
                try:
                    self._frustum_handle.remove()
                except Exception:
                    pass
                self._frustum_handle = None

            if bool(self.gui_show_camera.value) and self.seq_ctx is not None:
                cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
                if cam is not None and T_wr is not None:
                    T_wc = T_wr @ cam.T_camera_rig.inverse()
                    R = T_wc.R.reshape(3, 3).detach().cpu().numpy()
                    t = T_wc.t.reshape(3).detach().cpu().numpy()
                    q = self.tf.SO3.from_matrix(R).wxyz

                    c = cam.c.reshape(-1).detach().cpu().numpy()
                    f = cam.f.reshape(-1).detach().cpu().numpy()
                    h = float(cam.size.reshape(-1).detach().cpu().numpy()[1])
                    w = float(cam.size.reshape(-1).detach().cpu().numpy()[0])
                    fy = max(float(f[1]), 1e-6)
                    fov = float(2.0 * np.arctan2(h / 2.0, fy))
                    aspect = float(w / max(h, 1.0))

                    self._frustum_handle = self.server.scene.add_camera_frustum(
                        name=f"{SCENE_ROOT}/current_camera",
                        fov=fov,
                        aspect=aspect,
                        scale=0.25,
                        wxyz=q,
                        position=t,
                        line_width=1.5,
                    )

            if bool(self.gui_show_rgb.value):
                img = self._load_rgb_for_view(ts)
                if img is not None:
                    self.gui_rgb.image = img

            n_boxes = len(obbs)
            n_segments = int(len(segments))
            self.gui_status.value = (
                f"ts={ts} frame={idx + 1}/{len(self.timeline_ts)} boxes={n_boxes} segments={n_segments}"
            )

    def _playback_loop(self) -> None:
        while True:
            if bool(self.gui_play.value):
                now = time.time()
                fps = float(max(self.gui_fps.value, 0.1))
                if now - self._last_play_tick >= 1.0 / fps:
                    self._last_play_tick = now
                    nxt = int(self.gui_frame.value) + 1
                    if nxt > len(self.timeline_ts):
                        nxt = 1
                    self.gui_frame.value = nxt
            self._flush_pending_frame_update(force=False)
            time.sleep(0.01)

    def run_forever(self) -> None:
        while True:
            time.sleep(5.0)


class _ViserPromptViewer(_ViserTimelineViewer):
    def __init__(
        self,
        timed_obbs: Dict[int, ObbTW],
        timeline_ts: List[int],
        *,
        seq_name: str,
        host: str,
        port: int,
        seq_ctx: dict,
        boxernet,
        owl,
        device: str,
        precision_dtype,
    ):
        self.boxernet = boxernet
        self.owl = owl
        self.device = device
        self.precision_dtype = precision_dtype
        # Base __init__ triggers _update_frame(0). Guard prompt-specific
        # logic until this subclass finishes creating its GUI controls.
        self._prompt_gui_ready = False
        self._owl_overlay_by_ts: Dict[int, dict] = {}
        self._sdp_handle = None
        self._sdp_uploaded_once = False
        self._last_dynamic_sdp_ts: Optional[int] = None
        self._raw_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._disp_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._rgb_cache_cap = 12
        self._static_scene_points, self._static_scene_colors = _extract_static_scene_points(
            seq_ctx,
            max_points=120000,
        )

        super().__init__(
            timed_obbs,
            timeline_ts,
            seq_name=seq_name,
            host=host,
            port=port,
            seq_ctx=seq_ctx,
            title="Boxer Prompt Viewer (Viser)",
            show_trajectory=True,
            show_camera=True,
            show_rgb=True,
            seek_debounce_sec=0.12,
            apply_conf_filter=False,
        )

        self.gui_prompt_text = self.server.gui.add_text("Text Prompt", initial_value="chair")
        self.gui_detect_btn = self.server.gui.add_button("Detect + Lift (OWL -> BoxerNet)")
        self.gui_clear_btn = self.server.gui.add_button("Clear Current Frame Boxes")
        self.gui_show_sdp = self.server.gui.add_checkbox(
            "Show Scene Point Cloud", initial_value=True
        )
        self.gui_sdp_point_size = self.server.gui.add_slider(
            "SDP Point Size", min=0.001, max=0.03, step=0.001, initial_value=0.004
        )
        self.gui_show_rgb_3d_overlay = self.server.gui.add_checkbox(
            "Show RGB 3D Overlay", initial_value=True
        )
        self.gui_show_rgb_2d_overlay = self.server.gui.add_checkbox(
            "Show RGB 2D Overlay", initial_value=False
        )

        @self.gui_detect_btn.on_click
        def _(_evt) -> None:
            self._run_detect_lift()

        @self.gui_clear_btn.on_click
        def _(_evt) -> None:
            ts = int(self.timeline_ts[int(self.gui_frame.value) - 1])
            self.timed_obbs[ts] = ObbTW(torch.zeros(0, 165))
            self._owl_overlay_by_ts.pop(ts, None)
            self._update_frame(int(self.gui_frame.value) - 1)
            self._force_rebuild_current_obb_lines()
            self._flush_server_updates()

        @self.gui_show_sdp.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_sdp_point_size.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_show_rgb_3d_overlay.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_show_rgb_2d_overlay.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        self._prompt_gui_ready = True
        # Prompt scenes are easier to inspect with thicker lines + high-contrast
        # probability colormap by default.
        self.gui_line_width.value = 2.0
        self._line_handle.line_width = 2.0
        self.gui_color_mode.value = "probability"
        self._update_frame(int(self.gui_frame.value) - 1)

    def _flush_server_updates(self) -> None:
        # Force an immediate websocket flush so newly detected 3D boxes appear
        # without requiring a manual browser refresh.
        try:
            self.server.flush()
        except Exception:
            pass

    def _force_rebuild_current_obb_lines(self) -> None:
        with self._scene_update_lock:
            idx = int(np.clip(self._last_frame_idx, 0, len(self.timeline_ts) - 1))
            ts = int(self.timeline_ts[idx])
            obbs = self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
            segments, colors, _ = _obbs_to_segments(
                obbs,
                conf_thresh=(float(self.gui_conf.value) if self.apply_conf_filter else 0.0),
                color_mode=str(self.gui_color_mode.value),
            )
            self._set_obb_line_segments(
                segments,
                colors,
                float(self.gui_line_width.value),
            )

    def _update_prompt_sdp(self, ts_ns: int) -> None:
        if not hasattr(self, "gui_show_sdp") or not hasattr(self, "gui_sdp_point_size"):
            return

        if not bool(self.gui_show_sdp.value):
            if self._sdp_handle is not None:
                self._sdp_handle.visible = False
            return

        if len(self._static_scene_points) > 0:
            pts = self._static_scene_points
            cols = self._static_scene_colors
            if self._sdp_handle is None:
                self._sdp_handle = self.server.scene.add_point_cloud(
                    name=f"{SCENE_ROOT}/sdp_points",
                    points=pts,
                    colors=cols,
                    point_size=float(self.gui_sdp_point_size.value),
                    point_shape="circle",
                )
                self._sdp_uploaded_once = True
            else:
                # Static scene points do not change over time; avoid resending
                # large arrays every frame.
                if not self._sdp_uploaded_once:
                    self._sdp_handle.points = pts
                    self._sdp_handle.colors = cols
                    self._sdp_uploaded_once = True
                self._sdp_handle.visible = True
                try:
                    self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
                except Exception:
                    pass
            return
        else:
            ts_ns = int(ts_ns)
            if self._sdp_handle is not None and self._last_dynamic_sdp_ts == ts_ns:
                self._sdp_handle.visible = True
                try:
                    self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
                except Exception:
                    pass
                return
            sdp = _load_sdp(self.seq_ctx, ts_ns)
            if isinstance(sdp, torch.Tensor):
                pts = sdp.detach().cpu().numpy().reshape(-1, 3)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)
            pts = pts[np.isfinite(pts).all(axis=1)] if len(pts) > 0 else pts
            pts = _subsample_points_np(pts, max_points=30000)
            cols = np.full((len(pts), 3), 0.78, dtype=np.float32)
            self._last_dynamic_sdp_ts = ts_ns

        if self._sdp_handle is None:
            self._sdp_handle = self.server.scene.add_point_cloud(
                name=f"{SCENE_ROOT}/sdp_points",
                points=pts,
                colors=cols,
                point_size=float(self.gui_sdp_point_size.value),
                point_shape="circle",
            )
        else:
            self._sdp_handle.points = pts
            self._sdp_handle.colors = cols
            self._sdp_handle.visible = True
            try:
                self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
            except Exception:
                pass

    def _get_frame_images(self, ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ts_ns = int(ts_ns)
        raw = self._raw_rgb_cache.get(ts_ns, None)
        disp = self._disp_rgb_cache.get(ts_ns, None)
        if raw is not None and disp is not None:
            self._raw_rgb_cache.move_to_end(ts_ns)
            self._disp_rgb_cache.move_to_end(ts_ns)
            return raw, disp

        raw = _load_raw_rgb_image(self.seq_ctx, ts_ns)
        if raw is None:
            return None, None
        disp = _to_display_rgb(self.seq_ctx, raw)
        if disp is None:
            disp = raw.copy()

        self._raw_rgb_cache[ts_ns] = raw
        self._disp_rgb_cache[ts_ns] = disp
        while len(self._raw_rgb_cache) > self._rgb_cache_cap:
            k0, _ = self._raw_rgb_cache.popitem(last=False)
            self._disp_rgb_cache.pop(k0, None)
        return raw, disp

    def _load_rgb_for_view(self, ts_ns: int) -> Optional[np.ndarray]:
        # Prompt viewer renders its own cached RGB with overlays.
        return None

    def _update_frame(self, idx: int) -> None:
        super()._update_frame(idx)

        if not bool(getattr(self, "_prompt_gui_ready", False)):
            return

        idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
        ts = int(self.timeline_ts[idx])

        self._update_prompt_sdp(ts)

        if not bool(self.gui_show_rgb.value):
            return

        img_raw, img_disp = self._get_frame_images(ts)
        if img_raw is None or img_disp is None:
            return

        img = img_disp
        seq_ctx_dict = self.seq_ctx if isinstance(self.seq_ctx, dict) else {}

        if bool(self.gui_show_rgb_3d_overlay.value) or bool(self.gui_show_rgb_2d_overlay.value):
            cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
            projected = []
            if bool(self.gui_show_rgb_3d_overlay.value) and cam is not None and T_wr is not None:
                obbs = self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
                projected = _project_obbs_for_rgb_overlay(
                    obbs,
                    cam,
                    T_wr,
                    img_w=img_raw.shape[1],
                    img_h=img_raw.shape[0],
                    is_nebula=bool(seq_ctx_dict.get("is_nebula", True)),
                    conf_thresh=0.0,
                    color_mode=str(self.gui_color_mode.value),
                )
            owl_overlay = self._owl_overlay_by_ts.get(ts, None) if bool(self.gui_show_rgb_2d_overlay.value) else None
            img = _draw_rgb_overlays(img, projected, owl_overlay)

        self.gui_rgb.image = img

    def _run_detect_lift(self) -> None:
        self._flush_pending_frame_update(force=True)
        ts = int(self.timeline_ts[int(self._last_frame_idx)])
        text = str(self.gui_prompt_text.value).strip()
        if not text:
            self.gui_status.value = "Prompt is empty"
            return

        cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
        img_raw, _img_disp = self._get_frame_images(ts)
        if cam is None or T_wr is None or img_raw is None:
            self.gui_status.value = "Failed to load frame/camera/pose"
            return

        seq_ctx_dict = self.seq_ctx if isinstance(self.seq_ctx, dict) else {}

        rotated = bool(not seq_ctx_dict.get("is_nebula", True)) if seq_ctx_dict.get("source") == "aria" else False

        self.owl.set_text_prompts([text])
        img_torch_255 = torch.from_numpy(img_raw).permute(2, 0, 1).float()[None]
        boxes, scores2d, _label_ints, _ = self.owl.forward(
            img_torch_255,
            rotated,
            resize_to_HW=(906, 906),
        )
        if len(boxes) == 0:
            self.gui_status.value = f"OWL found 0 detections for '{text}'"
            return

        H, W = img_raw.shape[:2]
        bxr_hw = int(self.boxernet.hw)
        scale_x = bxr_hw / float(W)
        scale_y = bxr_hw / float(H)

        img_resized = torch.from_numpy(img_raw).permute(2, 0, 1).float()[None] / 255.0
        img_resized = torch.nn.functional.interpolate(
            img_resized,
            size=(bxr_hw, bxr_hw),
            mode="bilinear",
            align_corners=False,
        )[0]

        bb2d = boxes.clone()
        bb2d[:, 0] *= scale_x
        bb2d[:, 1] *= scale_x
        bb2d[:, 2] *= scale_y
        bb2d[:, 3] *= scale_y

        cam_data = cam._data.clone()
        cam_scaled = cam.__class__(cam_data)
        cam_scaled._data[0] = bxr_hw
        cam_scaled._data[1] = bxr_hw
        cam_scaled._data[2] *= scale_x
        cam_scaled._data[3] *= scale_y
        cam_scaled._data[4] *= scale_x
        cam_scaled._data[5] *= scale_y

        sdp_w = _load_sdp(self.seq_ctx, ts)

        datum = {
            "img0": img_resized,
            "cam0": cam_scaled.float(),
            "T_world_rig0": T_wr.float(),
            "rotated0": torch.tensor([rotated]),
            "sdp_w": sdp_w.float(),
            "bb2d": bb2d,
        }

        for k, v in list(datum.items()):
            if isinstance(v, torch.Tensor):
                datum[k] = v.to(self.device)
            elif hasattr(v, "to"):
                datum[k] = v.to(self.device)

        if self.device == "mps":
            outputs = self.boxernet.forward(datum)
        else:
            with torch.autocast(device_type=self.device, dtype=self.precision_dtype):
                outputs = self.boxernet.forward(datum)

        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
        keep = obb_pr_w.prob.squeeze(-1) >= float(self.gui_conf.value)
        accepted = obb_pr_w[keep].clone()

        keep_np = keep.detach().cpu().numpy().astype(bool)
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        if seq_ctx_dict.get("source") == "aria" and not bool(seq_ctx_dict.get("is_nebula", True)):
            boxes_np = _transform_boxes_raw_to_display(boxes_np, raw_w=W)
        scores_np = scores2d.detach().cpu().numpy()
        if len(keep_np) != len(boxes_np):
            kk = np.zeros((len(boxes_np),), dtype=bool)
            n_min = min(len(kk), len(keep_np))
            kk[:n_min] = keep_np[:n_min]
            keep_np = kk
        self._owl_overlay_by_ts[ts] = {
            "boxes": boxes_np,
            "scores": scores_np,
            "accepted": keep_np,
            "prompt": text,
        }

        if len(accepted) == 0:
            self.gui_status.value = (
                f"Detected {len(boxes)} 2D, accepted 0 3D above {self.gui_conf.value:.2f}"
            )
            self._update_frame(int(self._last_frame_idx))
            return

        if ts in self.timed_obbs and len(self.timed_obbs[ts]) > 0:
            old_data = self.timed_obbs[ts]._data
            new_data = accepted._data
            if old_data is not None and new_data is not None:
                self.timed_obbs[ts] = ObbTW(torch.cat([old_data, new_data], dim=0))
            else:
                self.timed_obbs[ts] = accepted
        else:
            self.timed_obbs[ts] = accepted

        self.gui_status.value = (
            f"Detected {len(boxes)} 2D, accepted {len(accepted)} 3D ({text})"
        )
        self._update_frame(int(self._last_frame_idx))
        self._force_rebuild_current_obb_lines()
        self._flush_server_updates()


class _ViserFusionViewer(_ViserTimelineViewer):
    def __init__(
        self,
        timed_obbs: Dict[int, ObbTW],
        timeline_ts: List[int],
        *,
        seq_name: str,
        host: str,
        port: int,
        seq_ctx: Optional[dict],
        show_rgb: bool,
        show_sdp: bool,
        show_rgb_3d_overlay: bool,
        sdp_point_size: float,
        seek_debounce_sec: float,
    ):
        self._raw_timed_obbs = {
            int(ts): _as_2d_obbs(obbs) for ts, obbs in timed_obbs.items()
        }
        self._display_timed_obbs = {int(ts): _empty_obbs() for ts in timeline_ts}
        self._fused_obbs: ObbTW = _empty_obbs()
        self._last_fusion_count = 0
        self.fusion_iou_threshold = 0.3
        self.fusion_min_detections = 4
        self.fusion_semantic_threshold = 0.7
        self.fusion_confidence_weighting = "robust"
        self.fusion_samp_per_dim = 8
        self.fusion_enable_nms = True
        self.fusion_nms_iou_threshold = 0.6
        self._fusion_gui_ready = False
        self._enable_rgb_3d_overlay = bool(show_rgb_3d_overlay)
        self._enable_sdp = bool(show_sdp)
        self._sdp_handle = None
        self._sdp_uploaded_once = False
        self._last_dynamic_sdp_ts: Optional[int] = None
        self._raw_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._disp_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._rgb_cache_cap = 12
        if seq_ctx is not None:
            self._static_scene_points, self._static_scene_colors = _extract_static_scene_points(
                seq_ctx,
                max_points=120000,
            )
        else:
            self._static_scene_points = np.zeros((0, 3), dtype=np.float32)
            self._static_scene_colors = np.zeros((0, 3), dtype=np.float32)

        super().__init__(
            self._display_timed_obbs,
            timeline_ts,
            seq_name=seq_name,
            host=host,
            port=port,
            seq_ctx=seq_ctx,
            title="Boxer Fusion Viewer (Viser)",
            show_trajectory=False,
            show_camera=False,
            show_rgb=bool(show_rgb and seq_ctx is not None),
            seek_debounce_sec=float(max(0.0, seek_debounce_sec)),
            apply_conf_filter=False,
        )

        self.gui_show_raw_set = self.server.gui.add_checkbox(
            "Show Per-Frame 3DBBs", initial_value=True
        )
        self.gui_show_fused_set = self.server.gui.add_checkbox(
            "Show Fused 3DBB", initial_value=True
        )
        self.gui_fusion_iou = self.server.gui.add_slider(
            "Fusion IoU", min=0.0, max=1.0, step=0.01, initial_value=self.fusion_iou_threshold
        )
        self.gui_fusion_min_det = self.server.gui.add_slider(
            "Fusion Min Detections", min=1, max=20, step=1, initial_value=self.fusion_min_detections
        )
        self.gui_fusion_sem = self.server.gui.add_slider(
            "Fusion Semantic Thresh", min=0.0, max=1.0, step=0.01, initial_value=self.fusion_semantic_threshold
        )
        self.gui_run_fusion = self.server.gui.add_button("Run Fusion")
        self.gui_fusion_status = self.server.gui.add_text(
            "Fusion", initial_value="not run"
        )

        @self.gui_show_raw_set.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_show_fused_set.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_fusion_iou.on_update
        def _(_evt) -> None:
            self.fusion_iou_threshold = float(self.gui_fusion_iou.value)

        @self.gui_fusion_min_det.on_update
        def _(_evt) -> None:
            self.fusion_min_detections = int(self.gui_fusion_min_det.value)

        @self.gui_fusion_sem.on_update
        def _(_evt) -> None:
            self.fusion_semantic_threshold = float(self.gui_fusion_sem.value)

        @self.gui_run_fusion.on_click
        def _(_evt) -> None:
            self._run_fusion_from_raw()
            self._update_frame(int(self.gui_frame.value) - 1)

        if self._enable_sdp and seq_ctx is not None:
            self.gui_show_sdp = self.server.gui.add_checkbox(
                "Show Scene Point Cloud", initial_value=True
            )
            self.gui_sdp_point_size = self.server.gui.add_slider(
                "SDP Point Size",
                min=0.001,
                max=0.03,
                step=0.001,
                initial_value=float(np.clip(sdp_point_size, 0.001, 0.03)),
            )

            @self.gui_show_sdp.on_update
            def _(_evt) -> None:
                self._update_frame(int(self.gui_frame.value) - 1)

            @self.gui_sdp_point_size.on_update
            def _(_evt) -> None:
                self._update_frame(int(self.gui_frame.value) - 1)

        if bool(self.gui_show_rgb.value):
            self.gui_show_rgb_3d_overlay = self.server.gui.add_checkbox(
                "Show RGB 3D Overlay", initial_value=bool(self._enable_rgb_3d_overlay)
            )

            @self.gui_show_rgb_3d_overlay.on_update
            def _(_evt) -> None:
                self._update_frame(int(self.gui_frame.value) - 1)

        self._fusion_gui_ready = True
        self._run_fusion_from_raw()
        self._update_frame(int(self.gui_frame.value) - 1)

    def _stack_all_raw_obbs(self) -> ObbTW:
        all_data = []
        for ts in sorted(self._raw_timed_obbs.keys()):
            obbs = _as_2d_obbs(self._raw_timed_obbs.get(ts, _empty_obbs()))
            if len(obbs) == 0 or obbs._data is None:
                continue
            all_data.append(obbs._data)
        if len(all_data) == 0:
            return _empty_obbs()
        return ObbTW(torch.cat(all_data, dim=0))

    def _run_fusion_from_raw(self) -> None:
        all_obbs = self._stack_all_raw_obbs()
        if len(all_obbs) == 0:
            self._fused_obbs = _empty_obbs()
            self._last_fusion_count = 0
            self.gui_fusion_status.value = "no detections"
            return

        mask = (all_obbs.prob >= float(self.gui_conf.value)).reshape(-1)
        filtered = all_obbs[mask]
        if len(filtered) == 0:
            self._fused_obbs = _empty_obbs()
            self._last_fusion_count = 0
            self.gui_fusion_status.value = "0 filtered detections"
            return

        try:
            from utils.fuse_3d_boxes import BoundingBox3DFuser, precompute_semantic_embeddings

            semantic_embeddings = precompute_semantic_embeddings(filtered)
            fuser = BoundingBox3DFuser(
                iou_threshold=float(self.fusion_iou_threshold),
                min_detections=int(self.fusion_min_detections),
                confidence_weighting=str(self.fusion_confidence_weighting),
                samp_per_dim=int(self.fusion_samp_per_dim),
                semantic_threshold=float(self.fusion_semantic_threshold),
                enable_nms=bool(self.fusion_enable_nms),
                nms_iou_threshold=float(self.fusion_nms_iou_threshold),
                conf_threshold=0.0,
            )
            fused_instances = fuser.fuse(
                filtered,
                semantic_embeddings=semantic_embeddings,
            )
            fused_data: List[torch.Tensor] = []
            for inst in fused_instances:
                if not hasattr(inst, "obb"):
                    continue
                obb = _as_2d_obbs(getattr(inst, "obb", None))
                if obb._data is not None and len(obb) > 0:
                    fused_data.append(obb._data)
            self._fused_obbs = ObbTW(torch.cat(fused_data, dim=0)) if len(fused_data) > 0 else _empty_obbs()
            self._last_fusion_count = int(len(self._fused_obbs))
            self.gui_fusion_status.value = (
                f"fused {self._last_fusion_count} instances from {len(filtered)} detections"
            )
        except Exception as exc:
            self._fused_obbs = _empty_obbs()
            self._last_fusion_count = 0
            self.gui_fusion_status.value = f"fusion failed: {exc}"

    def _compose_display_obbs(self, ts: int) -> ObbTW:
        show_raw = bool(getattr(self, "gui_show_raw_set", None) and self.gui_show_raw_set.value)
        show_fused = bool(getattr(self, "gui_show_fused_set", None) and self.gui_show_fused_set.value)

        raw = _as_2d_obbs(self._raw_timed_obbs.get(int(ts), _empty_obbs())) if show_raw else _empty_obbs()
        if len(raw) > 0:
            mask = (raw.prob >= float(self.gui_conf.value)).reshape(-1)
            raw = raw[mask] if int(mask.sum()) > 0 else _empty_obbs()
        fused = _as_2d_obbs(self._fused_obbs) if show_fused else _empty_obbs()

        if len(raw) == 0 and len(fused) == 0:
            return _empty_obbs()
        if len(raw) == 0:
            return fused
        if len(fused) == 0:
            return raw
        if raw._data is None:
            return fused
        if fused._data is None:
            return raw
        return ObbTW(torch.cat([raw._data, fused._data], dim=0))

    def _get_frame_images(self, ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ts_ns = int(ts_ns)
        raw = self._raw_rgb_cache.get(ts_ns, None)
        disp = self._disp_rgb_cache.get(ts_ns, None)
        if raw is not None and disp is not None:
            self._raw_rgb_cache.move_to_end(ts_ns)
            self._disp_rgb_cache.move_to_end(ts_ns)
            return raw, disp

        raw = _load_raw_rgb_image(self.seq_ctx, ts_ns)
        if raw is None:
            return None, None
        disp = _to_display_rgb(self.seq_ctx, raw)
        if disp is None:
            disp = raw.copy()

        self._raw_rgb_cache[ts_ns] = raw
        self._disp_rgb_cache[ts_ns] = disp
        while len(self._raw_rgb_cache) > self._rgb_cache_cap:
            k0, _ = self._raw_rgb_cache.popitem(last=False)
            self._disp_rgb_cache.pop(k0, None)
        return raw, disp

    def _load_rgb_for_view(self, ts_ns: int) -> Optional[np.ndarray]:
        # Fusion viewer draws optional overlays from cached RGB in _update_frame.
        return None

    def _update_fusion_sdp(self, ts_ns: int) -> None:
        if not self._enable_sdp or self.seq_ctx is None:
            return
        if not hasattr(self, "gui_show_sdp") or not hasattr(self, "gui_sdp_point_size"):
            return

        if not bool(self.gui_show_sdp.value):
            if self._sdp_handle is not None:
                self._sdp_handle.visible = False
            return

        if len(self._static_scene_points) > 0:
            pts = self._static_scene_points
            cols = self._static_scene_colors
            if self._sdp_handle is None:
                self._sdp_handle = self.server.scene.add_point_cloud(
                    name=f"{SCENE_ROOT}/sdp_points",
                    points=pts,
                    colors=cols,
                    point_size=float(self.gui_sdp_point_size.value),
                    point_shape="circle",
                )
                self._sdp_uploaded_once = True
            else:
                if not self._sdp_uploaded_once:
                    self._sdp_handle.points = pts
                    self._sdp_handle.colors = cols
                    self._sdp_uploaded_once = True
                self._sdp_handle.visible = True
                try:
                    self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
                except Exception:
                    pass
            return

        ts_ns = int(ts_ns)
        if self._sdp_handle is not None and self._last_dynamic_sdp_ts == ts_ns:
            self._sdp_handle.visible = True
            try:
                self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
            except Exception:
                pass
            return

        sdp = _load_sdp(self.seq_ctx, ts_ns)
        if isinstance(sdp, torch.Tensor):
            pts = sdp.detach().cpu().numpy().reshape(-1, 3)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)] if len(pts) > 0 else pts
        pts = _subsample_points_np(pts, max_points=30000)
        cols = np.full((len(pts), 3), 0.78, dtype=np.float32)
        self._last_dynamic_sdp_ts = ts_ns

        if self._sdp_handle is None:
            self._sdp_handle = self.server.scene.add_point_cloud(
                name=f"{SCENE_ROOT}/sdp_points",
                points=pts,
                colors=cols,
                point_size=float(self.gui_sdp_point_size.value),
                point_shape="circle",
            )
        else:
            self._sdp_handle.points = pts
            self._sdp_handle.colors = cols
            self._sdp_handle.visible = True
            try:
                self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
            except Exception:
                pass

    def _update_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
        ts = int(self.timeline_ts[idx])
        self.timed_obbs[ts] = self._compose_display_obbs(ts)

        super()._update_frame(idx)

        if not bool(getattr(self, "_fusion_gui_ready", False)):
            return

        self._update_fusion_sdp(ts)

        if self.seq_ctx is None or not bool(self.gui_show_rgb.value):
            return

        img_raw, img_disp = self._get_frame_images(ts)
        if img_raw is None or img_disp is None:
            return

        img = img_disp
        overlay_on = bool(
            hasattr(self, "gui_show_rgb_3d_overlay")
            and bool(self.gui_show_rgb_3d_overlay.value)
        )
        if overlay_on:
            cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
            if cam is not None and T_wr is not None:
                seq_ctx_dict = self.seq_ctx if isinstance(self.seq_ctx, dict) else {}
                obbs = self.timed_obbs.get(ts, _empty_obbs())
                projected = _project_obbs_for_rgb_overlay(
                    obbs,
                    cam,
                    T_wr,
                    img_w=img_raw.shape[1],
                    img_h=img_raw.shape[0],
                    is_nebula=bool(seq_ctx_dict.get("is_nebula", True)),
                    conf_thresh=(float(self.gui_conf.value) if self.apply_conf_filter else 0.0),
                    color_mode=str(self.gui_color_mode.value),
                )
                img = _draw_rgb_overlays(img, projected, owl_overlay=None)
        self.gui_rgb.image = img


class _ViserTrackerViewer(_ViserTimelineViewer):
    def __init__(
        self,
        timed_obbs: Dict[int, ObbTW],
        timeline_ts: List[int],
        *,
        seq_name: str,
        host: str,
        port: int,
        seq_ctx: dict,
        init_freeze_tracker: bool,
        seek_debounce_sec: float,
        param_apply_delay_sec: float,
        show_sdp: bool,
        show_rgb_3d_overlay: bool,
    ):
        self._det_timed_obbs = {int(ts): obbs for ts, obbs in timed_obbs.items()}
        self._display_timed_obbs = {int(ts): _empty_obbs() for ts in timeline_ts}
        self.freeze_tracker = bool(init_freeze_tracker)
        self.tracker_iou_threshold = 0.25
        self.tracker_min_hits = 8
        self.tracker_conf_threshold = 0.55
        self.raw_conf_threshold = 0.55
        self.tracker_merge_iou = 0.5
        self.tracker_merge_sem = 0.7
        self.tracker_merge_iou_2d = 0.7
        self.tracker_merge_interval = 5
        self.tracker_min_conf_mass = 4.0
        self.tracker_max_missed = 45
        self.tracker_min_obs_points = 4
        self.tracker_verbose = False

        self._tracker = self._make_tracker()
        self._tracker_frame_idx = 0
        self._params_dirty_time: Optional[float] = None
        self._seek_dirty_time: Optional[float] = None
        self._seek_target_frame = 0
        self._param_apply_delay_sec = float(max(0.0, param_apply_delay_sec))
        self._cached_params_snapshot = self._get_params_snapshot()

        self._last_raw_count = 0
        self._last_active_track_count = 0
        self._last_visible_track_count = 0
        self._last_param_apply_ts: Optional[float] = None

        self._enable_sdp = bool(show_sdp)
        self._enable_rgb_3d_overlay = bool(show_rgb_3d_overlay)
        self._sdp_handle = None
        self._sdp_uploaded_once = False
        self._last_dynamic_sdp_ts: Optional[int] = None
        self._raw_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._disp_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._rgb_cache_cap = 12
        self._static_scene_points, self._static_scene_colors = _extract_static_scene_points(
            seq_ctx,
            max_points=120000,
        )
        self._tracker_gui_ready = False
        self._current_raw_obbs: ObbTW = _empty_obbs()
        self._current_tracked_obbs: ObbTW = _empty_obbs()
        self._raw_line_handle = None
        self._raw_line_node_version = 0

        super().__init__(
            self._display_timed_obbs,
            timeline_ts,
            seq_name=seq_name,
            host=host,
            port=port,
            seq_ctx=seq_ctx,
            title="Boxer Tracker Viewer (Viser)",
            show_trajectory=True,
            show_camera=True,
            show_rgb=True,
            seek_debounce_sec=float(max(0.0, seek_debounce_sec)),
            apply_conf_filter=False,
        )

        self._raw_line_handle = self.server.scene.add_line_segments(
            f"{SCENE_ROOT}/raw_obbs",
            points=_empty_segments(),
            colors=_empty_colors(),
            line_width=1.0,
        )

        self.gui_freeze_tracker = self.server.gui.add_checkbox(
            "Freeze Tracker", initial_value=bool(self.freeze_tracker)
        )
        self.gui_tracker_conf = self.server.gui.add_slider(
            "Tracked 3DBB Conf", min=0.0, max=1.0, step=0.01, initial_value=self.tracker_conf_threshold
        )
        self.gui_raw_conf = self.server.gui.add_slider(
            "Per-Frame 3DBB Conf", min=0.0, max=1.0, step=0.01, initial_value=self.raw_conf_threshold
        )
        self.gui_tracker_iou = self.server.gui.add_slider(
            "Tracker IoU Threshold", min=0.0, max=1.0, step=0.01, initial_value=self.tracker_iou_threshold
        )
        self.gui_tracker_min_hits = self.server.gui.add_slider(
            "Tracker Min Hits", min=1, max=10, step=1, initial_value=self.tracker_min_hits
        )
        self.gui_tracker_max_missed = self.server.gui.add_slider(
            "Tracker Max Missed", min=1, max=120, step=1, initial_value=self.tracker_max_missed
        )
        self.gui_show_raw_set = self.server.gui.add_checkbox(
            "Show Per-Frame 3DBBs", initial_value=True
        )
        self.gui_show_tracked_set = self.server.gui.add_checkbox(
            "Show Tracked 3DBBs", initial_value=True
        )

        if self._enable_sdp:
            self.gui_show_sdp = self.server.gui.add_checkbox(
                "Show Scene Point Cloud", initial_value=True
            )
            self.gui_sdp_point_size = self.server.gui.add_slider(
                "SDP Point Size", min=0.001, max=0.03, step=0.001, initial_value=0.004
            )

        self.gui_show_rgb_3d_overlay = self.server.gui.add_checkbox(
            "Show RGB 3D Overlay", initial_value=bool(self._enable_rgb_3d_overlay)
        )

        @self.gui_freeze_tracker.on_update
        def _(_evt) -> None:
            self.freeze_tracker = bool(self.gui_freeze_tracker.value)
            self._mark_params_dirty()

        @self.gui_tracker_conf.on_update
        def _(_evt) -> None:
            self.tracker_conf_threshold = float(self.gui_tracker_conf.value)
            self._mark_params_dirty()

        @self.gui_raw_conf.on_update
        def _(_evt) -> None:
            self.raw_conf_threshold = float(self.gui_raw_conf.value)
            self._mark_params_dirty()

        @self.gui_tracker_iou.on_update
        def _(_evt) -> None:
            self.tracker_iou_threshold = float(self.gui_tracker_iou.value)
            self._mark_params_dirty()

        @self.gui_tracker_min_hits.on_update
        def _(_evt) -> None:
            self.tracker_min_hits = int(self.gui_tracker_min_hits.value)
            self._mark_params_dirty()

        @self.gui_tracker_max_missed.on_update
        def _(_evt) -> None:
            self.tracker_max_missed = int(self.gui_tracker_max_missed.value)
            self._mark_params_dirty()

        @self.gui_show_raw_set.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_show_tracked_set.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        if self._enable_sdp:
            @self.gui_show_sdp.on_update
            def _(_evt) -> None:
                self._update_frame(int(self.gui_frame.value) - 1)

            @self.gui_sdp_point_size.on_update
            def _(_evt) -> None:
                self._update_frame(int(self.gui_frame.value) - 1)

        @self.gui_show_rgb_3d_overlay.on_update
        def _(_evt) -> None:
            self._update_frame(int(self.gui_frame.value) - 1)

        self._tracker_gui_ready = True
        self._update_frame(int(self.gui_frame.value) - 1)

    def _make_tracker(self):
        from utils.track_3d_boxes import BoundingBox3DTracker

        return BoundingBox3DTracker(
            iou_threshold=float(self.tracker_iou_threshold),
            min_hits=int(self.tracker_min_hits),
            conf_threshold=float(self.tracker_conf_threshold),
            force_cpu=False,
            merge_iou_threshold=float(self.tracker_merge_iou),
            merge_semantic_threshold=float(self.tracker_merge_sem),
            merge_iou_2d_threshold=float(self.tracker_merge_iou_2d),
            merge_interval=int(self.tracker_merge_interval),
            min_confidence_mass=float(self.tracker_min_conf_mass),
            max_missed=int(self.tracker_max_missed),
            min_obs_points=int(self.tracker_min_obs_points),
            verbose=bool(self.tracker_verbose),
        )

    def _get_params_snapshot(self) -> Tuple:
        return (
            bool(self.freeze_tracker),
            float(self.tracker_iou_threshold),
            int(self.tracker_min_hits),
            float(self.tracker_conf_threshold),
            float(self.raw_conf_threshold),
            float(self.tracker_merge_iou),
            float(self.tracker_merge_sem),
            float(self.tracker_merge_iou_2d),
            int(self.tracker_merge_interval),
            float(self.tracker_min_conf_mass),
            int(self.tracker_max_missed),
            int(self.tracker_min_obs_points),
        )

    def _mark_params_dirty(self) -> None:
        self._params_dirty_time = time.time()

    def _reset_tracker(self) -> None:
        self._tracker = self._make_tracker()
        self._tracker_frame_idx = 0
        self._cached_params_snapshot = self._get_params_snapshot()
        self._display_timed_obbs = {int(ts): _empty_obbs() for ts in self.timeline_ts}
        self.timed_obbs = self._display_timed_obbs
        self._current_raw_obbs = _empty_obbs()
        self._current_tracked_obbs = _empty_obbs()

    def _filter_frame_obbs(self, obbs: ObbTW) -> ObbTW:
        if obbs is None or len(obbs) == 0:
            return _empty_obbs()
        mask = (obbs.prob >= float(self.raw_conf_threshold)).reshape(-1)
        if int(mask.sum()) == 0:
            return _empty_obbs()
        return obbs[mask]

    def _compose_tracker_display_obbs(self, raw: ObbTW, tracked: ObbTW) -> ObbTW:
        tracked = _as_2d_obbs(tracked)
        show_tracked = bool(
            getattr(self, "gui_show_tracked_set", None)
            and self.gui_show_tracked_set.value
        )
        return tracked if show_tracked else _empty_obbs()

    def _set_raw_line_segments(
        self,
        segments: np.ndarray,
        colors: np.ndarray,
        line_width: float,
    ) -> None:
        with self._scene_update_lock:
            old_handle = self._raw_line_handle
            self._raw_line_node_version += 1
            node_name = f"{SCENE_ROOT}/raw_obbs_{self._raw_line_node_version}"
            self._raw_line_handle = self.server.scene.add_line_segments(
                node_name,
                points=segments,
                colors=colors,
                line_width=float(line_width),
            )
            if old_handle is not None:
                try:
                    old_handle.remove()
                except Exception:
                    pass

    def _update_tracker_raw_overlay(self) -> None:
        show_raw = bool(getattr(self, "gui_show_raw_set", None) and self.gui_show_raw_set.value)
        if not show_raw:
            self._set_raw_line_segments(_empty_segments(), _empty_colors(), 1.0)
            return

        raw = _as_2d_obbs(self._current_raw_obbs)
        segments, _colors, _labels = _obbs_to_segments(
            raw,
            conf_thresh=0.0,
            color_mode="semantic",
        )
        if len(segments) == 0:
            self._set_raw_line_segments(_empty_segments(), _empty_colors(), 1.0)
            return

        raw_gray = np.full((segments.shape[0], 2, 3), 0.62, dtype=np.float32)
        self._set_raw_line_segments(segments, raw_gray, 1.0)

    def _get_observed_points(self, ts_ns: int) -> Optional[torch.Tensor]:
        sdp = _load_sdp(self.seq_ctx, int(ts_ns))
        if not isinstance(sdp, torch.Tensor) or sdp.numel() == 0:
            return None
        pts = sdp.reshape(-1, 3).float()
        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if len(pts) == 0:
            return None
        max_pts = 30000
        if len(pts) > max_pts:
            ids = torch.randperm(len(pts), device=pts.device)[:max_pts]
            pts = pts[ids]
        return pts

    def _tracker_update_one(self, frame_idx: int) -> ObbTW:
        ts = int(self.timeline_ts[frame_idx])
        dets = self._filter_frame_obbs(self._det_timed_obbs.get(ts, _empty_obbs()))
        self._last_raw_count = int(len(dets))
        cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
        obs_pts = self._get_observed_points(ts)
        self._tracker.update(
            dets,
            int(frame_idx),
            cam=cam,
            T_world_rig=T_wr,
            observed_points=obs_pts,
        )

        active_tracks = self._tracker._get_active_tracks()
        shown_tracks = []
        for track in active_tracks:
            avg_conf = float(track.accumulated_weight / max(track.support_count, 1))
            if avg_conf >= float(self.tracker_conf_threshold):
                shown_tracks.append(track)

        self._last_active_track_count = int(len(shown_tracks))
        if len(shown_tracks) == 0:
            self._last_visible_track_count = 0
            return _empty_obbs()

        tracked_obbs = _stack_obbs([track.obb for track in shown_tracks])
        visible_count = 0
        if cam is not None and T_wr is not None and len(tracked_obbs) > 0:
            try:
                _, bb2_valid = tracked_obbs.get_pseudo_bb2(
                    cam.unsqueeze(0),
                    T_wr.unsqueeze(0),
                    num_samples_per_edge=10,
                    valid_ratio=0.16667,
                )
                visible_mask = bb2_valid.squeeze(0)
                visible_count = int(visible_mask.sum().item())
            except Exception:
                visible_count = 0
        self._last_visible_track_count = int(visible_count)
        return tracked_obbs

    def _tracker_step_to_frame(self, target_idx: int) -> None:
        target_idx = int(np.clip(target_idx, 0, len(self.timeline_ts) - 1))
        ts = int(self.timeline_ts[target_idx])
        raw_dets = self._filter_frame_obbs(self._det_timed_obbs.get(ts, _empty_obbs()))
        self._last_raw_count = int(len(raw_dets))

        if self.freeze_tracker:
            self._current_raw_obbs = raw_dets
            self._current_tracked_obbs = _empty_obbs()
            self.timed_obbs[ts] = self._compose_tracker_display_obbs(raw_dets, _empty_obbs())
            self._last_active_track_count = 0
            self._last_visible_track_count = 0
            self._tracker_frame_idx = target_idx
            return

        if target_idx == self._tracker_frame_idx + 1:
            tracked_obbs = self._tracker_update_one(target_idx)
        elif target_idx != self._tracker_frame_idx:
            # Match local TrackerViewer semantics for seek/jump:
            # reset and update only the target frame.
            self._reset_tracker()
            tracked_obbs = self._tracker_update_one(target_idx)
        else:
            active_tracks = self._tracker._get_active_tracks()
            shown_tracks = []
            for track in active_tracks:
                avg_conf = float(track.accumulated_weight / max(track.support_count, 1))
                if avg_conf >= float(self.tracker_conf_threshold):
                    shown_tracks.append(track)
            tracked_obbs = _stack_obbs([track.obb for track in shown_tracks]) if len(shown_tracks) > 0 else _empty_obbs()
            self._last_active_track_count = int(len(shown_tracks))
            self._last_visible_track_count = int(len(shown_tracks))

        tracked_obbs = tracked_obbs if isinstance(tracked_obbs, ObbTW) else _empty_obbs()
        self._current_raw_obbs = raw_dets
        self._current_tracked_obbs = tracked_obbs
        self.timed_obbs[ts] = self._compose_tracker_display_obbs(raw_dets, tracked_obbs)
        self._tracker_frame_idx = target_idx

    def _apply_deferred_tracker_params(self) -> None:
        if self._params_dirty_time is None:
            return
        now = time.time()
        if now - self._params_dirty_time < self._param_apply_delay_sec:
            return
        params_snapshot = self._get_params_snapshot()
        if params_snapshot == self._cached_params_snapshot:
            self._params_dirty_time = None
            return

        target = int(self._seek_target_frame if self._seek_dirty_time is not None else self._last_frame_idx)
        target = int(np.clip(target, 0, len(self.timeline_ts) - 1))
        self._params_dirty_time = None
        self._seek_dirty_time = None
        self._pending_frame_dirty = False
        was_playing = bool(self.gui_play.value)
        self.gui_play.value = False
        self._reset_tracker()
        self._tracker_step_to_frame(target)
        self._last_param_apply_ts = time.time()
        self._update_frame(target)
        self.gui_play.value = was_playing

    def _queue_frame_update(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
        self._seek_target_frame = idx
        self._seek_dirty_time = time.time()
        self._pending_frame_idx = idx
        self._pending_frame_dirty = True
        self._pending_frame_event_ts = self._seek_dirty_time
        if bool(self.gui_play.value):
            self._flush_pending_frame_update(force=True)

    def _flush_pending_frame_update(self, force: bool = False) -> None:
        if not self._pending_frame_dirty:
            return
        now = time.time()
        if force or bool(self.gui_play.value) or (now - self._pending_frame_event_ts >= self.seek_debounce_sec):
            self._pending_frame_dirty = False
            self._seek_dirty_time = None
            self._update_frame(self._pending_frame_idx)

    def _playback_loop(self) -> None:
        while True:
            if bool(self.gui_play.value):
                now = time.time()
                fps = float(max(self.gui_fps.value, 0.1))
                if now - self._last_play_tick >= 1.0 / fps:
                    self._last_play_tick = now
                    nxt = int(self.gui_frame.value) + 1
                    if nxt > len(self.timeline_ts):
                        nxt = 1
                    self.gui_frame.value = nxt

            self._apply_deferred_tracker_params()
            self._flush_pending_frame_update(force=False)
            time.sleep(0.01)

    def _get_frame_images(self, ts_ns: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ts_ns = int(ts_ns)
        raw = self._raw_rgb_cache.get(ts_ns, None)
        disp = self._disp_rgb_cache.get(ts_ns, None)
        if raw is not None and disp is not None:
            self._raw_rgb_cache.move_to_end(ts_ns)
            self._disp_rgb_cache.move_to_end(ts_ns)
            return raw, disp

        raw = _load_raw_rgb_image(self.seq_ctx, ts_ns)
        if raw is None:
            return None, None
        disp = _to_display_rgb(self.seq_ctx, raw)
        if disp is None:
            disp = raw.copy()

        self._raw_rgb_cache[ts_ns] = raw
        self._disp_rgb_cache[ts_ns] = disp
        while len(self._raw_rgb_cache) > self._rgb_cache_cap:
            k0, _ = self._raw_rgb_cache.popitem(last=False)
            self._disp_rgb_cache.pop(k0, None)
        return raw, disp

    def _load_rgb_for_view(self, ts_ns: int) -> Optional[np.ndarray]:
        return None

    def _update_tracker_sdp(self, ts_ns: int) -> None:
        if not self._enable_sdp or not hasattr(self, "gui_show_sdp"):
            return
        if not bool(self.gui_show_sdp.value):
            if self._sdp_handle is not None:
                self._sdp_handle.visible = False
            return

        if len(self._static_scene_points) > 0:
            pts = self._static_scene_points
            cols = self._static_scene_colors
            if self._sdp_handle is None:
                self._sdp_handle = self.server.scene.add_point_cloud(
                    name=f"{SCENE_ROOT}/sdp_points",
                    points=pts,
                    colors=cols,
                    point_size=float(self.gui_sdp_point_size.value),
                    point_shape="circle",
                )
                self._sdp_uploaded_once = True
            else:
                if not self._sdp_uploaded_once:
                    self._sdp_handle.points = pts
                    self._sdp_handle.colors = cols
                    self._sdp_uploaded_once = True
                self._sdp_handle.visible = True
                try:
                    self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
                except Exception:
                    pass
            return

        ts_ns = int(ts_ns)
        if self._sdp_handle is not None and self._last_dynamic_sdp_ts == ts_ns:
            self._sdp_handle.visible = True
            try:
                self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
            except Exception:
                pass
            return

        sdp = _load_sdp(self.seq_ctx, ts_ns)
        if isinstance(sdp, torch.Tensor):
            pts = sdp.detach().cpu().numpy().reshape(-1, 3)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)] if len(pts) > 0 else pts
        pts = _subsample_points_np(pts, max_points=30000)
        cols = np.full((len(pts), 3), 0.78, dtype=np.float32)
        self._last_dynamic_sdp_ts = ts_ns

        if self._sdp_handle is None:
            self._sdp_handle = self.server.scene.add_point_cloud(
                name=f"{SCENE_ROOT}/sdp_points",
                points=pts,
                colors=cols,
                point_size=float(self.gui_sdp_point_size.value),
                point_shape="circle",
            )
        else:
            self._sdp_handle.points = pts
            self._sdp_handle.colors = cols
            self._sdp_handle.visible = True
            try:
                self._sdp_handle.point_size = float(self.gui_sdp_point_size.value)
            except Exception:
                pass

    def _update_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.timeline_ts) - 1))
        with self._scene_update_lock:
            self._tracker_step_to_frame(idx)
            super()._update_frame(idx)

            self._update_tracker_raw_overlay()

            if not bool(getattr(self, "_tracker_gui_ready", False)):
                return

            ts = int(self.timeline_ts[idx])
            self._update_tracker_sdp(ts)

            if bool(self.gui_show_rgb.value):
                img_raw, img_disp = self._get_frame_images(ts)
                if img_raw is not None and img_disp is not None:
                    img = img_disp
                    if bool(self.gui_show_rgb_3d_overlay.value):
                        cam, T_wr = _get_cam_pose(self.seq_ctx, ts)
                        if cam is not None and T_wr is not None:
                            seq_ctx_dict = self.seq_ctx if isinstance(self.seq_ctx, dict) else {}
                            obbs = self.timed_obbs.get(ts, _empty_obbs())
                            projected = _project_obbs_for_rgb_overlay(
                                obbs,
                                cam,
                                T_wr,
                                img_w=img_raw.shape[1],
                                img_h=img_raw.shape[0],
                                is_nebula=bool(seq_ctx_dict.get("is_nebula", True)),
                                conf_thresh=0.0,
                                color_mode=str(self.gui_color_mode.value),
                            )
                            img = _draw_rgb_overlays(img, projected, owl_overlay=None)
                    self.gui_rgb.image = img

            seg_count = int(len(self._line_handle.points)) if hasattr(self._line_handle, "points") else 0
            param_ts = "-" if self._last_param_apply_ts is None else f"{self._last_param_apply_ts:.2f}"
            self.gui_status.value = (
                f"ts={ts} frame={idx + 1}/{len(self.timeline_ts)} "
                f"raw={self._last_raw_count} active={self._last_active_track_count} "
                f"visible={self._last_visible_track_count} segments={seg_count} param_apply={param_ts}"
            )


def run_viser_fusion_viewer(
    timed_obbs: Dict[int, ObbTW],
    *,
    seq_name: str,
    seq_ctx: Optional[dict] = None,
    show_rgb: bool = False,
    show_sdp: bool = False,
    show_rgb_3d_overlay: bool = True,
    sdp_point_size: float = 0.004,
    seek_debounce_sec: float = 0.08,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    timeline = sorted([int(ts) for ts in timed_obbs.keys()])
    viewer = _ViserFusionViewer(
        timed_obbs,
        timeline,
        seq_name=seq_name,
        host=host,
        port=port,
        seq_ctx=seq_ctx,
        show_rgb=bool(show_rgb),
        show_sdp=bool(show_sdp),
        show_rgb_3d_overlay=bool(show_rgb_3d_overlay),
        sdp_point_size=float(sdp_point_size),
        seek_debounce_sec=float(seek_debounce_sec),
    )
    print(f"==> Viser fusion viewer running on http://{host}:{port}")
    viewer.run_forever()


def run_viser_tracker_viewer(
    timed_obbs: Dict[int, ObbTW],
    *,
    seq_name: str,
    seq_ctx: dict,
    init_freeze_tracker: bool = False,
    seek_debounce_sec: float = 0.12,
    param_apply_delay_sec: float = 0.3,
    show_sdp: bool = True,
    show_rgb_3d_overlay: bool = True,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    timeline = sorted([int(ts) for ts in timed_obbs.keys()])
    viewer = _ViserTrackerViewer(
        timed_obbs,
        timeline,
        seq_name=seq_name,
        host=host,
        port=port,
        seq_ctx=seq_ctx,
        init_freeze_tracker=bool(init_freeze_tracker),
        seek_debounce_sec=float(seek_debounce_sec),
        param_apply_delay_sec=float(param_apply_delay_sec),
        show_sdp=bool(show_sdp),
        show_rgb_3d_overlay=bool(show_rgb_3d_overlay),
    )
    print(f"==> Viser tracker viewer running on http://{host}:{port}")
    viewer.run_forever()


def run_viser_prompt_viewer(
    timed_obbs: Dict[int, ObbTW],
    *,
    seq_name: str,
    seq_ctx: dict,
    boxernet,
    owl,
    device: str,
    precision_dtype,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    timeline = sorted([int(ts) for ts in timed_obbs.keys()])
    if len(timeline) == 0:
        timeline = sorted([int(ts) for ts in seq_ctx.get("rgb_timestamps", [])])

    viewer = _ViserPromptViewer(
        timed_obbs,
        timeline,
        seq_name=seq_name,
        host=host,
        port=port,
        seq_ctx=seq_ctx,
        boxernet=boxernet,
        owl=owl,
        device=device,
        precision_dtype=precision_dtype,
    )
    print(f"==> Viser prompt viewer running on http://{host}:{port}")
    viewer.run_forever()
