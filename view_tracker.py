#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""View pre-computed BoxerNet results in 3D tracker mode."""

import argparse
import os

from utils.file_io import read_obb_csv
from utils.viser_viewer import run_viser_tracker_viewer
from utils.viewer_3d import (
    TrackerViewer,
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    resolve_bb2d_csv,
    scale_factor,
    subsample_timed_obbs,
)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="View BoxerNet results (tracker mode)")
    add_common_args(parser)
    parser.add_argument("--init_follow", action="store_true", help="Initialize with Follow View enabled")
    parser.add_argument("--init_follow_behind", type=float, default=None, help="Follow-view behind distance (meters)")
    parser.add_argument("--init_follow_above", type=float, default=6.0, help="Follow-view above distance (meters)")
    parser.add_argument("--init_follow_look_ahead", type=float, default=None, help="Follow-view look-ahead distance (meters)")
    parser.add_argument("--init_show_obs", action="store_true", help="Initially show observed points")
    parser.add_argument("--autoplay", action="store_true", help="Automatically start playback")
    parser.add_argument("--autorecord", action="store_true")
    parser.add_argument("--record_fps", type=float, default=0.0, help="Recording FPS (0 = auto)")
    parser.add_argument("--teaser", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tracker logging")
    parser.add_argument("--bb2d_csv", type=str, default="", help="2D BB CSV filename (relative to log_dir)")
    parser.add_argument("--viewer_backend", type=str, default="local", choices=["local", "viser"], help="Viewer backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--init_freeze_tracker", action="store_true", help="Start viser tracker with freeze mode enabled")
    parser.add_argument("--viser_param_apply_delay", type=float, default=0.3, help="Viser tracker parameter apply delay in seconds")
    parser.add_argument("--viser_seek_debounce_sec", type=float, default=0.12, help="Viser tracker seek debounce in seconds")
    parser.add_argument("--viser_no_sdp", action="store_true", help="Disable scene point cloud in viser tracker")
    parser.add_argument("--viser_no_rgb_3d_overlay", action="store_true", help="Disable 3D overlay on RGB in viser tracker")
    # fmt: on
    args = parser.parse_args()

    input_path, dataset_type, seq_name, log_dir, view_path, load_view_data = (
        load_common(args)
    )

    # Load OBBs
    tracked_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_tracked.csv")
    raw_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")

    timed_obbs = None
    csv_path = None
    if os.path.exists(tracked_csv):
        tracked_obbs = read_obb_csv(tracked_csv)
        if len(tracked_obbs) <= 1 and os.path.exists(raw_csv):
            print(
                f"[WARN] {tracked_csv} has only {len(tracked_obbs)} timestamp(s). "
                f"Falling back to raw detections: {raw_csv}"
            )
            csv_path = raw_csv
            timed_obbs = read_obb_csv(raw_csv)
        else:
            csv_path = tracked_csv
            timed_obbs = tracked_obbs
    elif os.path.exists(raw_csv):
        csv_path = raw_csv
        timed_obbs = read_obb_csv(raw_csv)

    if csv_path is None or timed_obbs is None:
        raise IOError(f"3D BB CSV not found: {raw_csv} or {tracked_csv}")

    if args.verbose:
        print(f"==> Loading OBBs from {csv_path}")
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    if args.verbose:
        total_dets = sum(len(obbs) for obbs in timed_obbs.values())
        print(f"==> Loaded {len(timed_obbs)} frames, {total_dets} detections")

    seq_ctx = build_seq_ctx(input_path, dataset_type)

    if args.viewer_backend == "viser":
        run_viser_tracker_viewer(
            timed_obbs,
            seq_name=seq_name,
            seq_ctx=seq_ctx,
            init_freeze_tracker=bool(args.init_freeze_tracker),
            seek_debounce_sec=float(args.viser_seek_debounce_sec),
            param_apply_delay_sec=float(args.viser_param_apply_delay),
            show_sdp=not bool(args.viser_no_sdp),
            show_rgb_3d_overlay=not bool(args.viser_no_rgb_3d_overlay),
            host=args.host,
            port=args.port,
        )
        return

    bb2d_csv_path = resolve_bb2d_csv(log_dir, args.bb2d_csv, args.write_name)

    default_w, default_h = 2250 * scale_factor, 1100 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class Viewer(TrackerViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                bb2d_csv_path=bb2d_csv_path,
                init_rgb_text_scale=args.init_rgb_text_scale,
                init_color_mode=args.init_color_mode,
                init_follow=args.init_follow,
                init_follow_behind=args.init_follow_behind,
                init_follow_above=args.init_follow_above,
                init_follow_look_ahead=args.init_follow_look_ahead,
                init_show_obs=args.init_show_obs,
                init_image_panel_width=args.init_image_panel_width,
                autorecord=args.autorecord,
                record_fps=args.record_fps,
                teaser=args.teaser,
                verbose=args.verbose,
                load_view_data=load_view_data,
                view_save_path=view_path,
                scannet_scene=args.scannet_scene,
                scannet_annotation_path=args.scannet_annotation_path,
                **kw,
            )
            if args.autoplay:
                self.is_playing = True
                self.follow_view = True

    launch_viewer(Viewer)


if __name__ == "__main__":
    main()
