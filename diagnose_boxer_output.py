#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Geometric quality metrics for Boxer 3D outputs — no ground truth needed.

Three orthogonal self-consistency checks per 3D box:

  A. dist_to_cloud  Distance from box center to the nearest scene-cloud point.
                    Catches boxes placed in empty space.

  B. iou2d          Reproject the box's 8 corners into the image with the same
                    (K, pose) BoxerNet saw, take the axis-aligned 2D bbox,
                    compute IoU with the OWL prompt bbox.
                    Catches K / pose / 3D-position inconsistency.

  C. depth_gap      Camera-frame Z of the box's 8 corners vs. the Z distribution
                    of SDP points whose projection lands inside the OWL 2D bbox.
                    Catches "right angular direction, wrong distance" — the
                    failure mode systematic K bias / per-segment scale drift
                    produces, which A and B both tend to miss.

Each metric is orthogonal: one can pass and the other two fail. Combining all
three as a filter drops clearly-bad per-frame boxes before `fuse_3d_boxes.py`
clusters them, producing a cleaner scene graph.

Usage:
    python diagnose_boxer_output.py --input <seq>
    python diagnose_boxer_output.py --input <seq> --filter
    python diagnose_boxer_output.py --input <seq> --filter \\
        --max_dist_to_cloud 0.5 --min_iou2d 0.1 --max_depth_gap 0.5

Outputs (in `<output_dir>/<seq>/`):
    diagnose_by_box.csv       per-box metrics
    diagnose_summary.json     aggregate distributions
    <write_name>_3dbbs_filtered.csv   (only with --filter) bad boxes removed
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import torch

from utils.demo_utils import DEFAULT_SEQ, EVAL_PATH
from utils.file_io import load_bb2d_csv, read_obb_csv
from utils.viewer_3d import build_seq_ctx, resolve_input


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _project_points(
    pts_world: np.ndarray,   # [N, 3]
    T_wc: np.ndarray,         # [4, 4] world_from_camera
    K: np.ndarray,            # [3, 3] pinhole
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv [N, 2], z [N]). z > 0 means in front of camera."""
    T_cw = np.linalg.inv(T_wc)
    pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = pts_h @ T_cw.T
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
        v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _box_corners_from_obb(obb) -> np.ndarray:
    """Return the 8 world-frame corners of an ObbTW as [8, 3] float32."""
    bb3 = obb.bb3_object.squeeze(0).numpy()          # [6] xmin, xmax, ymin, ymax, zmin, zmax
    T_wo = obb.T_world_object                        # PoseTW
    R_wo = T_wo.R.squeeze(0).numpy()                 # [3, 3]
    t_wo = T_wo.t.squeeze(0).numpy()                 # [3]
    xs = (bb3[0], bb3[1]); ys = (bb3[2], bb3[3]); zs = (bb3[4], bb3[5])
    corners_local = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32
    )
    return (corners_local @ R_wo.T + t_wo[None]).astype(np.float32)


def _match_bb2d_for_obb(
    label: str,
    proj_center_uv: np.ndarray,   # [2]
    bb2d_entries: list[dict],     # from load_bb2d_csv per-frame list
    max_dist_px: float = 1e9,
) -> Optional[np.ndarray]:
    """Find the OWL 2D bbox with matching label whose center is closest in the image.

    Returns xyxy array or None. run_boxer's thresh_3d filter strips the index
    correspondence between BoxerNet outputs and OWL prompts, so we recover the
    link heuristically by nearest-projected-center within the same label class.
    """
    best = None; best_d = max_dist_px
    for d in bb2d_entries:
        if d.get("label") != label:
            continue
        bb = d["bb2d"]                                 # (x1, y1, x2, y2) numpy array
        c = np.array([0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])], dtype=np.float32)
        dist = float(np.linalg.norm(c - proj_center_uv))
        if dist < best_d:
            best = np.asarray(bb, dtype=np.float32); best_d = dist
    return best


# ---------------------------------------------------------------------------
# Per-frame context: pose + K + SDP lookup
# ---------------------------------------------------------------------------


def _pose_K_for_time(ctx: dict, time_ns: int) -> Optional[tuple[np.ndarray, np.ndarray, int, int]]:
    """Find the closest (pose, K) to a given timestamp. Returns (T_wc, K, H, W).

    build_seq_ctx's outputs differ per dataset, but all expose pose_ts + traj
    (camera poses) and calibs (CameraTW). We pick by nearest-timestamp.
    """
    pose_ts = ctx.get("pose_ts")
    traj = ctx.get("traj")
    calibs = ctx.get("calibs")
    calib_ts = ctx.get("calib_ts", pose_ts)
    if pose_ts is None or traj is None or calibs is None:
        return None

    # Traj may be a list of PoseTW (CA-1M path) or a tensor / list (Aria).
    idx = int(np.argmin(np.abs(np.asarray(pose_ts) - time_ns)))
    pose = traj[idx] if isinstance(traj, list) else traj[idx]
    # pose can be a PoseTW or a 4x4 tensor; normalize.
    if hasattr(pose, "_data"):
        R = pose.R.numpy() if hasattr(pose.R, "numpy") else np.asarray(pose.R)
        t = pose.t.numpy() if hasattr(pose.t, "numpy") else np.asarray(pose.t)
        if R.ndim == 3:
            R = R[0]; t = t[0]
        T_wc = np.eye(4, dtype=np.float32)
        T_wc[:3, :3] = R; T_wc[:3, 3] = t
    else:
        T_wc = np.asarray(pose, dtype=np.float32)
        if T_wc.shape == (3, 4):
            T_wc = np.vstack([T_wc, [0, 0, 0, 1]]).astype(np.float32)

    cidx = int(np.argmin(np.abs(np.asarray(calib_ts) - time_ns)))
    cam = calibs[cidx] if isinstance(calibs, list) else calibs
    # CameraTW stores [w, h, fx, fy, cx, cy, ...].
    data = cam._data if hasattr(cam, "_data") else cam
    data = data.squeeze().numpy() if hasattr(data, "numpy") else np.asarray(data).squeeze()
    W = int(round(float(data[0]))); H = int(round(float(data[1])))
    fx = float(data[2]); fy = float(data[3]); cx = float(data[4]); cy = float(data[5])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return T_wc, K, H, W


def _resolve_scene_cloud(ctx: dict, max_points: int = 80_000) -> np.ndarray:
    """Best-effort scene cloud from the sequence context (may be empty)."""
    # CA-1M exposes sdp_global.
    sdp = ctx.get("sdp_global")
    if sdp is not None and len(sdp) > 0:
        pts = np.asarray(sdp, dtype=np.float32)
    else:
        # Aria path: uid_to_p3 maps unique IDs to 3-vectors.
        uid_to_p3 = ctx.get("uid_to_p3")
        if uid_to_p3:
            pts = np.asarray(list(uid_to_p3.values()), dtype=np.float32)
        else:
            return np.zeros((0, 3), dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=DEFAULT_SEQ, help="path to the sequence folder")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Where CSVs live (default: ~/viz_boxer)")
    parser.add_argument("--write_name", default="boxer", type=str, help="CSV prefix (default: boxer)")
    parser.add_argument("--filter", action="store_true", help="write a *_filtered.csv with bad boxes dropped")
    parser.add_argument("--max_dist_to_cloud", type=float, default=0.5, help="filter threshold (m)")
    parser.add_argument("--min_iou2d", type=float, default=0.1, help="filter threshold")
    parser.add_argument("--max_depth_gap", type=float, default=0.5, help="filter threshold (m)")
    args = parser.parse_args()

    input_path, dataset_type, seq_name = resolve_input(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)

    obb_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    bb2d_csv = os.path.join(log_dir, "owl_2dbbs.csv")
    missing = [p for p in (obb_csv, bb2d_csv) if not os.path.exists(p)]
    if missing:
        print("[diagnose] missing prerequisite file(s):")
        for p in missing:
            print(f"  - {p}")
        print(f"\nGenerate them with:\n  python run_boxer.py --input {seq_name} --skip_viz")
        sys.exit(1)

    print(f"[diagnose] loading OBBs from {obb_csv}")
    timed_obbs = read_obb_csv(obb_csv)
    total_obbs = sum(len(o) for o in timed_obbs.values())
    print(f"[diagnose] loading 2D bboxes from {bb2d_csv}")
    timed_bb2d = load_bb2d_csv(bb2d_csv)

    print(f"[diagnose] building sequence context ({dataset_type})")
    ctx = build_seq_ctx(input_path, dataset_type)
    cloud = _resolve_scene_cloud(ctx)
    has_cloud = cloud.shape[0] > 0
    if not has_cloud:
        print("[diagnose] no scene cloud available for this dataset — A and C will be NaN")

    bb2d_times = np.array(sorted(timed_bb2d.keys()), dtype=np.int64) if timed_bb2d else np.zeros(0, dtype=np.int64)

    rows_out: list[dict] = []
    for time_ns in sorted(timed_obbs.keys()):
        obbs = timed_obbs[time_ns]
        if not obbs:
            continue

        ctx_pose_K = _pose_K_for_time(ctx, time_ns)
        if ctx_pose_K is None:
            continue
        T_wc, K, H, W = ctx_pose_K

        # Match the frame's 2D bboxes by nearest timestamp (within 50 ms).
        bb2d_entries: list[dict] = []
        if bb2d_times.size > 0:
            bb_idx = int(np.searchsorted(bb2d_times, time_ns))
            bb_idx = min(bb_idx, bb2d_times.size - 1)
            nearest = int(bb2d_times[bb_idx])
            if abs(nearest - time_ns) < 50_000_000:
                entry = timed_bb2d.get(nearest, {})
                n = len(entry.get("bb2d", []))
                for i in range(n):
                    bb2d_entries.append({
                        "label": entry["labels"][i] if "labels" in entry else "",
                        "bb2d": np.asarray(entry["bb2d"][i], dtype=np.float32),
                    })

        for obb in obbs:
            # Label as a string
            label = ""
            try:
                txt = obb.text_string()
                label = txt[0] if isinstance(txt, (list, tuple)) else str(txt)
            except Exception:
                pass

            center = obb.T_world_object.t.squeeze(0).numpy().astype(np.float32)
            corners = _box_corners_from_obb(obb)

            # Metric A
            dist_to_cloud = float("nan")
            if has_cloud:
                d2 = np.sum((cloud - center[None, :]) ** 2, axis=1)
                dist_to_cloud = float(np.sqrt(d2.min()))

            # Project center for OWL matching
            proj_center_uv, proj_center_z = _project_points(center[None, :], T_wc, K)
            owl_bb = None
            if proj_center_z[0] > 1e-3 and np.isfinite(proj_center_uv).all():
                owl_bb = _match_bb2d_for_obb(label, proj_center_uv[0], bb2d_entries)

            # Metric B
            iou2d = float("nan")
            if owl_bb is not None:
                uv, z = _project_points(corners, T_wc, K)
                mask = (z > 1e-3) & np.isfinite(uv).all(axis=1)
                if mask.sum() >= 2:
                    u = uv[mask, 0]; v = uv[mask, 1]
                    proj_bb = np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32)
                    iou2d = _iou_xyxy(proj_bb, owl_bb)

            # Metric C
            depth_gap = float("nan")
            if owl_bb is not None and has_cloud:
                uv, z = _project_points(cloud, T_wc, K)
                in_front = z > 1e-3
                in_bbox = (
                    (uv[:, 0] >= owl_bb[0]) & (uv[:, 0] <= owl_bb[2]) &
                    (uv[:, 1] >= owl_bb[1]) & (uv[:, 1] <= owl_bb[3])
                )
                keep = in_front & in_bbox & np.isfinite(uv).all(axis=1)
                zs_in = z[keep]
                if zs_in.size >= 30:
                    q10, q50, q90 = np.percentile(zs_in, [10, 50, 90]).tolist()
                    _, z_box = _project_points(corners, T_wc, K)
                    z_box = z_box[np.isfinite(z_box)]
                    if z_box.size > 0:
                        z_box_min, z_box_max = float(z_box.min()), float(z_box.max())
                        depth_gap = float(max(0.0, q10 - z_box_max, z_box_min - q90))

            rows_out.append({
                "time_ns": time_ns,
                "label": label,
                "prob": float(obb.prob.item()) if hasattr(obb, "prob") else float("nan"),
                "cx": float(center[0]), "cy": float(center[1]), "cz": float(center[2]),
                "dist_to_cloud": dist_to_cloud,
                "iou2d": iou2d,
                "owl_matched": owl_bb is not None,
                "depth_gap": depth_gap,
            })

    # Write per-box CSV.
    out_csv = os.path.join(log_dir, "diagnose_by_box.csv")
    os.makedirs(log_dir, exist_ok=True)
    if rows_out:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
    print(f"[diagnose] wrote {len(rows_out)} rows → {out_csv}")

    # Aggregate summary.
    def _stats(key: str) -> Optional[dict]:
        vals = np.array(
            [r[key] for r in rows_out if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        )
        if vals.size == 0:
            return None
        return {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
        }
    summary = {
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "n_boxes": len(rows_out),
        "n_owl_matched": sum(1 for r in rows_out if r["owl_matched"]),
        "has_scene_cloud": bool(has_cloud),
        "dist_to_cloud": _stats("dist_to_cloud"),
        "iou2d": _stats("iou2d"),
        "depth_gap": _stats("depth_gap"),
    }
    summary_path = os.path.join(log_dir, "diagnose_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Optional filter — write a reduced CSV for downstream fuse_3d_boxes.py.
    if args.filter:
        filtered_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_filtered.csv")
        key_to_row = {
            (r["time_ns"], round(r["cx"], 5), round(r["cy"], 5), round(r["cz"], 5)): r
            for r in rows_out
        }
        kept, dropped = 0, 0
        with open(obb_csv) as fin, open(filtered_csv, "w", newline="") as fout:
            header = fin.readline()
            fout.write(header)
            for line in fin:
                parts = line.split(",")
                key = (
                    int(parts[0]),
                    round(float(parts[1]), 5),
                    round(float(parts[2]), 5),
                    round(float(parts[3]), 5),
                )
                r = key_to_row.get(key)
                ok = True
                if r is not None:
                    if not math.isnan(r["dist_to_cloud"]) and r["dist_to_cloud"] > args.max_dist_to_cloud:
                        ok = False
                    elif not math.isnan(r["iou2d"]) and r["iou2d"] < args.min_iou2d:
                        ok = False
                    elif not math.isnan(r["depth_gap"]) and r["depth_gap"] > args.max_depth_gap:
                        ok = False
                if ok:
                    fout.write(line); kept += 1
                else:
                    dropped += 1
        print(f"[diagnose] filter: kept {kept}, dropped {dropped} → {filtered_csv}")
        print(f"[diagnose]   thresholds: max_dist_to_cloud={args.max_dist_to_cloud} "
              f"min_iou2d={args.min_iou2d} max_depth_gap={args.max_depth_gap}")
        print(f"[diagnose]   now run: python -c \"from utils.fuse_3d_boxes import fuse_obbs_from_csv; "
              f"fuse_obbs_from_csv('{filtered_csv}')\"")


if __name__ == "__main__":
    main()
