#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""View pre-computed BoxerNet 3D bounding boxes in a minimal 3D viewer."""

import argparse
import os

import torch

from utils.demo_utils import DEFAULT_SEQ, EVAL_PATH
from utils.file_io import read_obb_csv
from utils.viewer_3d import (
    OBBViewer,
    launch_viewer,
    resolve_input,
    scale_factor,
    subsample_timed_obbs,
)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="View BoxerNet 3D bounding boxes")
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
    # fmt: on
    args = parser.parse_args()

    input_path, dataset_type, seq_name = resolve_input(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)

    # Load OBBs — prefer fused CSV if available
    fused_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_fused.csv")
    raw_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    csv_path = fused_csv if os.path.exists(fused_csv) else raw_csv

    print(f"==> Loading OBBs from {csv_path}")
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] CSV not found: {csv_path}")
        print("\nRun BoxerNet first to generate it:\n")
        cmd = f"  python run_boxer.py --input {seq_name} --skip_viz"
        if args.write_name != "boxer":
            cmd += f" --write_name {args.write_name}"
        if args.output_dir != EVAL_PATH:
            cmd += f" --output_dir {args.output_dir}"
        print(cmd + "\n")
        raise SystemExit(1)
    timed_obbs = read_obb_csv(csv_path)
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    total_dets = sum(len(obbs) for obbs in timed_obbs.values())
    print(f"==> Loaded {len(timed_obbs)} frames, {total_dets} detections")

    # Stack all OBBs
    from utils.tw.obb import ObbTW

    all_obbs_list = []
    for ts in sorted(timed_obbs.keys()):
        all_obbs_list.extend(timed_obbs[ts])
    all_obbs = (
        torch.stack(all_obbs_list) if all_obbs_list else ObbTW(torch.zeros(0, 165))
    )

    # Resolve view file
    view_path = os.path.join(log_dir, "camera_view.pt")
    load_view_data = None
    if args.load_view is not None:
        target = view_path if args.load_view == "DEFAULT" else args.load_view
        if os.path.exists(target):
            load_view_data = torch.load(target, weights_only=False)
            print(f"==> Loaded camera view from {target}")

    default_w, default_h = 1400 * scale_factor, 900 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class Viewer(OBBViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                all_obbs=all_obbs,
                timed_obbs=timed_obbs,
                root_path=log_dir,
                init_color_mode=args.init_color_mode,
                load_view_data=load_view_data,
                view_save_path=view_path,
                seq_name=seq_name,
                **kw,
            )
            self.show_raw_set = True
            self.show_tracked_all_set = True
            self.show_tracked_visible_set = False

        def _render_main_controls(self):
            self._render_fusion_controls()

        def _render_common_visual_controls(self, **kw):
            super()._render_common_visual_controls(
                raw_checkbox_label="Show Per-Frame 3DBBs",
                tracked_all_checkbox_label="Show Fused 3DBB",
                show_visible_checkbox=False,
                show_sets_header=False,
            )

    launch_viewer(Viewer)


if __name__ == "__main__":
    main()
