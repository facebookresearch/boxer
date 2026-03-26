#! /usr/bin/env python3

# pyre-unsafe
"""View pre-computed BoxerNet results in 3D tracker mode."""

import argparse
import os

from view_boxer import (
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    read_obb_csv,
    resolve_bb2d_csv,
    subsample_timed_obbs,
)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="View BoxerNet results (tracker mode)")
    add_common_args(parser)
    parser.add_argument("--init_follow", action="store_true", help="Initialize with Follow View enabled")
    parser.add_argument("--init_follow_behind", type=float, default=None, help="Follow-view behind distance (meters)")
    parser.add_argument("--init_follow_above", type=float, default=None, help="Follow-view above distance (meters)")
    parser.add_argument("--init_follow_look_ahead", type=float, default=None, help="Follow-view look-ahead distance (meters)")
    parser.add_argument("--init_overlay2d", action="store_true", help="Overlay 2D BBs on RGB panel")
    parser.add_argument("--init_show_obs", action="store_true", help="Initially show observed points")
    parser.add_argument("--autorecord", action="store_true")
    parser.add_argument("--record_fps", type=float, default=0.0, help="Recording FPS (0 = auto)")
    parser.add_argument("--teaser", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tracker logging")
    parser.add_argument("--bb2d_csv", type=str, default="", help="2D BB CSV filename (relative to log_dir)")
    # fmt: on
    args = parser.parse_args()

    input_path, dataset_type, seq_name, log_dir, view_path, load_view_data = (
        load_common(args)
    )

    # Load OBBs — prefer tracked CSV if available
    tracked_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_tracked.csv")
    raw_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    csv_path = tracked_csv if os.path.exists(tracked_csv) else raw_csv
    if not os.path.exists(tracked_csv):
        print(f"==> Tracked CSV not found, falling back to {csv_path}")

    print(f"==> Loading OBBs from {csv_path}")
    timed_obbs = read_obb_csv(csv_path)
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    total_dets = sum(len(obbs) for obbs in timed_obbs.values())
    print(f"==> Loaded {len(timed_obbs)} frames, {total_dets} detections")

    seq_ctx = build_seq_ctx(input_path, dataset_type)
    bb2d_csv_path = resolve_bb2d_csv(log_dir, args.bb2d_csv, args.write_name)

    default_w, default_h = 2250, 1100
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    from utils.viz_boxer import TrackerViewer

    class Viewer(TrackerViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                bb2d_csv_path=bb2d_csv_path if os.path.exists(bb2d_csv_path) else "",
                freeze_tracker=False,
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

    launch_viewer(Viewer)


if __name__ == "__main__":
    main()
