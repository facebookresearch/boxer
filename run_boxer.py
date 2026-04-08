#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import argparse
import os
import re
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from boxernet.boxernet import BoxerNet
from loaders.ca_loader import CALoader
from loaders.omni_loader import OMNI3D_DATASETS, OmniLoader
from loaders.scannet_loader import ScanNetLoader
from utils.demo_utils import (
    CKPT_PATH,
    DEFAULT_SEQ,
    EVAL_PATH,
    SAMPLE_DATA_PATH,
    CudaTimer,
)
from utils.file_io import ObbCsvWriter2, load_bb2d_csv, read_obb_csv, save_bb2d_csv
from utils.image import draw_bb3s, put_text, render_bb2, render_depth_patches, torch2cv2
from utils.taxonomy import load_text_labels
from utils.tw.tensor_utils import (
    pad_string,
    string2tensor,
    tensor2string,
    unpad_string,
)
from utils.video import make_mp4, safe_delete_folder


def jet_color(val):
    """Map a scalar in [0, 1] to an RGB tuple via OpenCV's JET colormap."""
    val = max(0.0, min(1.0, float(val)))
    bgr = cv2.applyColorMap(np.uint8([[int(val * 255)]]), cv2.COLORMAP_JET)[0, 0]
    return (float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0)


def jet_colors_bgr(scores):
    """Vectorized: map array of scores in [0,1] to list of BGR (int) tuples."""
    if len(scores) == 0:
        return []
    vals = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
    u8 = (vals * 255).astype(np.uint8).reshape(1, -1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0]  # (N, 3)
    return [tuple(int(c) for c in row) for row in bgr]


TAB20 = [
    (0.122, 0.467, 0.706),
    (0.682, 0.780, 0.910),
    (1.000, 0.498, 0.055),
    (1.000, 0.733, 0.471),
    (0.173, 0.627, 0.173),
    (0.596, 0.875, 0.541),
    (0.839, 0.153, 0.157),
    (1.000, 0.596, 0.588),
    (0.580, 0.404, 0.741),
    (0.773, 0.690, 0.835),
    (0.549, 0.337, 0.294),
    (0.769, 0.612, 0.580),
    (0.890, 0.467, 0.761),
    (0.969, 0.714, 0.824),
    (0.498, 0.498, 0.498),
    (0.780, 0.780, 0.780),
    (0.737, 0.741, 0.133),
    (0.859, 0.859, 0.553),
    (0.090, 0.745, 0.812),
    (0.620, 0.855, 0.898),
]


def comma_separated_list(value):
    # Handle empty string gracefully
    if not value:
        return []
    return value.split(",")


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_SEQ, help="path to the sequence folder")
    parser.add_argument("--skip_n", type=int, default=1, help="skip n frames")
    parser.add_argument("--start_n", type=int, default=1, help="start from n-th frame")
    parser.add_argument("--max_n", type=int, default=99999, help="run for max n frames")
    parser.add_argument("--pinhole", action="store_true", help="rectify to pinhole")
    parser.add_argument("--camera", type=str, default="rgb", choices=["rgb", "slaml", "slamr"], help="camera to use (default: rgb)")
    parser.add_argument("--detector", type=str, default="owl", choices=["owl"], help="2D detector to use (default: owl)")
    parser.add_argument("--thresh2d", type=float, default=0.25, help="detection confidence for 2d detector")
    parser.add_argument("--thresh3d", type=float, default=0.5, help="detection confidence for boxer")
    parser.add_argument("--labels", type=comma_separated_list, nargs="?", const=[], default=["lvisplus"], help="Optional comma-separated list of text prompts (e.g. --labels=small or --labels=chair,table,lamp)")
    parser.add_argument("--detector_hw", type=int, default=960, help="resize images before going into 2D detector")
    parser.add_argument("--write_name", default="boxer", type=str, help="name prefix for outputs")
    parser.add_argument("--skip_viz", action="store_true", help="disable headless visualization (on by default)")
    parser.add_argument("--cache2d", action="store_true", help="load 2D BBs from CSV instead of running detector")
    parser.add_argument("--cache3d", action="store_true", help="load 3D BBs from CSV instead of running BoxerNet")
    parser.add_argument("--no_sdp", action="store_true", help="turn off SDP input")
    parser.add_argument("--no_csv", action="store_true", help="skip CSV writing")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU")
    parser.add_argument("--gt2d", action="store_true", help="use GT pseudo 2DBB as input")
    parser.add_argument("--fuse", action="store_true", help="run offline 3D box fusion after processing")
    parser.add_argument("--track", action="store_true", help="run online 3D box tracking and show tracked boxes in Top Down View")
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_PATH, "boxernet_hw960in4x6d768-wssxpf9p.ckpt"), help="path to BoxerNet checkpoint")
    parser.add_argument("--force_precision", type=str, default=None, choices=["float32", "bfloat16"], help="Override auto-detected inference precision")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Output directory for results (default: output/)")
    args = parser.parse_args()

    if args.fuse and args.track:
        parser.error("--fuse and --track are mutually exclusive")
    if args.cache3d:
        args.cache2d = True
    args.viz_headless = not args.skip_viz
    print(args)
    # fmt: on

    DEBUG = os.environ.get("DEBUG", "0") == "1"
    _t_start = time.perf_counter()
    _t_prev = _t_start

    def _dbg(label):
        nonlocal _t_prev
        if not DEBUG:
            return
        now = time.perf_counter()
        print(
            f"  [init] {label}: {(now - _t_prev) * 1000:.0f}ms (total: {(now - _t_start) * 1000:.0f}ms)",
            flush=True,
        )
        _t_prev = now

    # Determine dataset type and seq_name from input string
    if bool(re.search(r"scene\d{4}_\d{2}", args.input)) or "/scannet/" in args.input:
        dataset_type = "scannet"
        seq_name = os.path.basename(args.input.rstrip("/"))
    elif args.input in OMNI3D_DATASETS:
        dataset_type = "omni3d"
        seq_name = args.input
    elif args.input.startswith("ca1m"):
        dataset_type = "ca1m"
        seq_name = args.input
    else:
        dataset_type = "aria"
        remote_root = args.input
        # Resolve bare sequence names: try sample_data/ first, then ~/boxy_data/
        if not os.path.isabs(remote_root) and not os.path.exists(remote_root):
            sample = os.path.join(SAMPLE_DATA_PATH, remote_root)
            legacy = os.path.expanduser(os.path.join("~/boxy_data", remote_root))
            if os.path.exists(sample):
                remote_root = sample
            elif os.path.exists(legacy):
                remote_root = legacy
        seq_name = remote_root.rstrip("/").split("/")[-1]

    # get name of containing directory
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    csv2d_out_path = os.path.join(log_dir, "owl_2dbbs.csv")
    print(f"==> Created output folder {log_dir}")
    _dbg("setup")

    # --cache3d: skip detection + BoxerNet + loader, go straight to post-processing
    if args.cache3d:
        print(f"==> Loading cached 3D BBs from {csv_path}")
        cached_timed_obbs = read_obb_csv(csv_path)
        total_dets = sum(len(obbs) for obbs in cached_timed_obbs.values())
        print(
            f"==> Loaded {len(cached_timed_obbs)} frames, {total_dets} detections from cache"
        )

        if args.fuse:
            from utils.fuse_3d_boxes import fuse_obbs_from_csv

            print(f"\n==> Running fusion on {csv_path}")
            fuse_obbs_from_csv(csv_path)

        if os.path.exists(csv2d_out_path):
            print(f"==> 2D BB CSV exists: {csv2d_out_path}")
        else:
            print(
                f"==> No 2D BB CSV found (run without --cache3d to generate: {csv2d_out_path})"
            )
        return

    # Create data loader
    if dataset_type == "scannet":
        loader = ScanNetLoader(
            scene_dir=args.input,
            annotation_path=os.path.join(
                SAMPLE_DATA_PATH, "scannet", "full_annotations.json"
            ),
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            start_frame=args.start_n,
        )
        seq_name = loader.scene_id
    elif dataset_type == "omni3d":
        print(f"==> Loading Omni3D dataset: {args.input} (val)")
        loader = OmniLoader(
            dataset_name=args.input,
            split="val",
            max_images=args.max_n,
            skip_images=args.skip_n,
        )
        # Disable fusion for Omni3D (single images, not video)
        if args.fuse:
            print(
                "==> Warning: --fuse is disabled for Omni3D (single images, not video)"
            )
            args.fuse = False
        if args.track:
            print(
                "==> Warning: --track is disabled for Omni3D (single images, not video)"
            )
            args.track = False
    elif dataset_type == "ca1m":
        loader = CALoader(
            seq_name,
            start_frame=args.start_n,
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            resize=(args.detector_hw, args.detector_hw),
        )
    else:
        from loaders.aria_loader import AriaLoader

        print(f"==> Sequence name: '{seq_name}'")
        loader = AriaLoader(
            remote_root,
            camera=args.camera,
            with_traj=True,
            with_sdp=True,
            with_obb=args.gt2d,
            pinhole=args.pinhole,
            resize=None,
            unrotate=False,
            skip_n=args.skip_n,
            max_n=args.max_n,
            start_n=args.start_n,
        )

    _dbg("loader")

    # choose a model checkpoint
    if torch.backends.mps.is_available() and not args.force_cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"==> Using device {device}")

    # Load text labels if they match special strings.
    text_labels = load_text_labels(args.labels)
    # Track taxonomy name for visualization
    taxonomy_name = args.labels[0] if args.labels else "custom"
    if not args.gt2d:
        print(f"==> Using text prompts ({taxonomy_name}):")
        if len(text_labels) > 64:
            print(text_labels[:64])
            print(
                f"    ... and {len(text_labels) - 64} more (total: {len(text_labels)})"
            )
        else:
            print(text_labels)

    # Load 2D detector (skip if --cache2d)
    if args.cache2d:
        print(f"==> Loading cached 2D BBs from {csv2d_out_path}")
        bb2d_cache = load_bb2d_csv(csv2d_out_path)
        bb2d_cache_timestamps = np.array(sorted(bb2d_cache.keys()), dtype=np.int64)
        print(f"==> Loaded {len(bb2d_cache)} frames of 2D BBs from cache")
        method = "CACHED"
    elif args.gt2d:
        method = "GT2D"
    else:
        from owl.owl_wrapper import OwlWrapper

        owl = OwlWrapper(
            device,
            text_prompts=text_labels,
            min_confidence=args.thresh2d,
            precision=args.force_precision,
        )
        method = "OWLv2"
    _dbg("owl")

    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    loader.resize = boxernet.hw
    # Re-trigger prefetch so the first frame uses the correct resize.
    loader.index = 0
    loader._init_prefetch()
    print(f"==> Will resize images to {loader.resize}x{loader.resize} for boxernet")
    _dbg("boxernet")

    # Print model architecture
    total_params = sum(p.numel() for p in boxernet.parameters())
    print("=" * 50)
    print(f"  BOXERNET ARCHITECTURE ({total_params / 1e6:.2f}M params)")
    print("=" * 50)
    for name, module in boxernet.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({n_params / 1e6:.2f}M)")
    print("=" * 50)

    _dbg("arch_print")

    video_dir = os.path.join(log_dir, f"{args.write_name}_viz")
    if args.viz_headless:
        safe_delete_folder(
            video_dir, extensions=[".png"], keep_folder=True, recursive=True
        )
        os.makedirs(video_dir, exist_ok=True)
        print(
            f"==> Current frame: {os.path.join(log_dir, f'{args.write_name}_viz_current.jpg')}"
        )

    colors = {
        label: (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255),
        )
        for label in text_labels
    }

    if args.gt2d:
        sem_name_to_id = loader.sem_name_to_id
        sem_id_to_name = {val: key for key, val in sem_name_to_id.items()}
    else:
        sem_name_to_id = {label: i for i, label in enumerate(text_labels)}
        sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}

    writer = None if args.no_csv else ObbCsvWriter2(csv_path)

    tracker = None
    if args.track:
        from utils.track_3d_boxes import BoundingBox3DTracker

        tracker = BoundingBox3DTracker(
            iou_threshold=0.25,
            min_hits=8,
            conf_threshold=args.thresh3d,
            samp_per_dim=8,
            max_missed=90,
            force_cpu=args.force_cpu,
            verbose=False,
        )

    def write_empty_frame(img_np, HH, WW, ii):
        panels = [img_np, img_np]
        if args.track:
            panels.append(img_np)
        final = np.hstack(panels)
        _, jpg_buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = jpg_buf.tobytes()
        out_path = os.path.join(video_dir, f"{args.write_name}_viz_{ii:05d}.jpg")
        with open(out_path, "wb") as f:
            f.write(jpg_bytes)
        out_path = os.path.join(log_dir, f"{args.write_name}_viz_current.jpg")
        with open(out_path, "wb") as f:
            f.write(jpg_bytes)

    timestamps_ns = []  # Collect timestamps to compute FPS
    timer = CudaTimer(device)
    pbar = tqdm(range(len(loader)), desc="BoxerNet")
    DEBUG_VIZ = os.environ.get("DEBUG_VIZ", "0") == "1"
    _dbg("ready")

    for ii in pbar:
        # Data loading
        timer.start("load")
        if DEBUG_VIZ:
            _tl0 = time.perf_counter()
        try:
            datum = next(loader)
        except StopIteration:
            break

        if datum is False:
            pbar.set_postfix_str("Skipped (time misalignment)")
            continue

        if DEBUG_VIZ:
            _tl1 = time.perf_counter()

        # Collect timestamp for FPS calculation
        if args.viz_headless and "time_ns0" in datum:
            timestamps_ns.append(int(datum["time_ns0"]))

        img_torch = datum["img0"]
        rotated = datum["rotated0"]
        HH, WW = img_torch.shape[2], img_torch.shape[3]
        img_np = torch2cv2(img_torch, rotate=rotated, ensure_rgb=True)
        t_load = timer.stop("load")

        if DEBUG_VIZ:
            _tl2 = time.perf_counter()
            print(
                f"  [load] next(loader): {(_tl1 - _tl0) * 1000:.1f}ms  torch2cv2: {(_tl2 - _tl1) * 1000:.1f}ms",
                flush=True,
            )

        sdp_w_viz = datum["sdp_w"].float()  # Keep original SDP for visualization
        if args.no_sdp:
            # TURN OFF SDP inputs by removing them
            print("==> Removing SDP inputs")
            datum["sdp_w"] = torch.zeros(0, 3)

        # 2D Detection
        timer.start("owl")
        if args.cache2d:
            # Look up cached 2D BBs by timestamp
            time_ns = int(datum["time_ns0"])
            cache_entry = None
            if time_ns in bb2d_cache:
                cache_entry = bb2d_cache[time_ns]
            else:
                # Find nearest timestamp
                idx = np.searchsorted(bb2d_cache_timestamps, time_ns)
                idx = min(idx, len(bb2d_cache_timestamps) - 1)
                nearest_ts = int(bb2d_cache_timestamps[idx])
                if abs(nearest_ts - time_ns) < 50_000_000:  # within 50ms
                    cache_entry = bb2d_cache[nearest_ts]
            if cache_entry is not None and len(cache_entry["bb2d"]) > 0:
                # Convert from numpy (x1,y1,x2,y2) to torch boxer format (x1,x2,y1,y2)
                bb2d_np = cache_entry["bb2d"]
                bb2d = torch.from_numpy(bb2d_np[:, [0, 2, 1, 3]]).float()
                scores2d = torch.from_numpy(cache_entry["scores"]).float()
                labels2d = list(cache_entry["labels"])
            else:
                bb2d = torch.zeros(0, 4)
                scores2d = torch.zeros(0)
                labels2d = []
        elif args.gt2d:
            # Check if there are any valid GT objects for this frame
            obbs_valid = datum["obbs"].remove_padding()
            if len(obbs_valid) == 0:
                t_owl = timer.stop("owl")
                pbar.set_postfix_str(
                    f"0 GT obbs | load:{t_load:.0f}ms owl:{t_owl:.0f}ms"
                )
                if args.viz_headless:
                    write_empty_frame(img_np, HH, WW, ii)

                continue

            # Use pre-computed 2D bounding boxes (CA-1M, Omni3D, or SST)
            bb2d = datum["bb2d0"]
            # Get labels: from gt_labels if available, otherwise from full obbs
            # Note: bb2d0 comes from full obbs, so labels must match full obbs length
            if "gt_labels" in datum and len(datum["gt_labels"]) == len(bb2d):
                labels2d = datum["gt_labels"]
            else:
                labels2d = datum["obbs"].text_string()
            # Filter out invalid entries: NaN or xmin == -1 (invalid OBB bb2)
            valid_mask = ~torch.isnan(bb2d).any(dim=1) & (bb2d[:, 0] >= 0)
            bb2d = bb2d[valid_mask]
            labels2d = [labels2d[i] for i in range(len(valid_mask)) if valid_mask[i]]
            if len(bb2d) == 0:
                t_owl = timer.stop("owl")
                pbar.set_postfix_str(
                    f"0 valid bb2d | load:{t_load:.0f}ms owl:{t_owl:.0f}ms"
                )
                if args.viz_headless:
                    write_empty_frame(img_np, HH, WW, ii)
                continue

            scores2d = 0.5 * torch.ones(bb2d.shape[0])
        else:
            img_torch_255 = img_torch.clone() * 255.0
            bb2d, scores2d, label_ints, _ = owl.forward(
                img_torch_255,
                rotated.item(),
                resize_to_HW=(args.detector_hw, args.detector_hw),
            )
            labels2d = [text_labels[label_int] for label_int in label_ints]

        t_owl = timer.stop("owl")

        if bb2d.shape[0] == 0:
            pbar.set_postfix_str(f"0 dets | load:{t_load:.0f}ms owl:{t_owl:.0f}ms")
            if args.viz_headless:
                write_empty_frame(img_np, HH, WW, ii)
            continue

        # 3D BoxerNet
        timer.start("boxer")
        sdp_w = datum["sdp_w"].float()
        cam = datum["cam0"].float()
        T_wr = datum["T_world_rig0"].float()
        datum["bb2d"] = bb2d
        if args.force_precision is not None:
            precision_dtype = (
                torch.bfloat16 if args.force_precision == "bfloat16" else torch.float32
            )
        elif device == "cuda" and torch.cuda.is_bf16_supported():
            precision_dtype = torch.bfloat16
        else:
            precision_dtype = torch.float32
        # MPS does not support torch.autocast
        if device == "mps":
            outputs = boxernet.forward(datum)
        else:
            with torch.autocast(device_type=device, dtype=precision_dtype):
                outputs = boxernet.forward(datum)
        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

        # Populate sem_id with text labels from 2d detector.
        assert len(obb_pr_w) == len(labels2d)
        sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
        for i in range(len(labels2d)):
            label = labels2d[i]
            if label in sem_name_to_id:
                sem_ids[i] = sem_name_to_id[label]
            else:
                # Dynamically add new labels to the mapping
                new_id = len(sem_name_to_id)
                sem_name_to_id[label] = new_id
                sem_id_to_name[new_id] = label
                sem_ids[i] = new_id
        obb_pr_w.set_sem_id(sem_ids)

        # Confidence: filter by 3D confidence and combine with 2D scores
        scores3d = obb_pr_w.prob.squeeze(-1).clone()
        keepers = obb_pr_w.prob.squeeze(-1) >= args.thresh3d
        obb_pr_w = obb_pr_w[keepers].clone()
        scores3d = scores3d[keepers].clone()
        labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
        mean_scores = (scores2d[keepers] + scores3d) / 2.0
        obb_pr_w.set_prob(mean_scores)

        # Set text description in ObbTW.
        if len(labels3d) > 0:
            text_data = torch.stack(
                [string2tensor(pad_string(lab, max_len=128)) for lab in labels3d]
            )
            obb_pr_w.set_text(text_data)
        t_boxer = timer.stop("boxer")

        # Visualization (includes writing CSV and images)
        timer.start("csv")
        time_ns = int(datum["time_ns0"])
        if writer is not None:
            writer.write(obb_pr_w, time_ns, sem_id_to_name=sem_id_to_name)

            # Convert bb2d from boxer format (x1, x2, y1, y2) to standard (x1, y1, x2, y2)
            bb2d_xyxy = bb2d[:, [0, 2, 1, 3]]
            save_bb2d_csv(
                csv2d_out_path,
                frame_id=ii,
                bb2d=bb2d_xyxy,
                scores=scores2d,
                labels=labels2d,
                sem_name_to_id=sem_name_to_id,
                append=(ii > 0),
                time_ns=time_ns,
                img_width=WW,
                img_height=HH,
                sensor=loader.camera if hasattr(loader, "camera") else "unknown",
                device=loader.device_name
                if hasattr(loader, "device_name")
                else "unknown",
            )
        t_csv = timer.stop("csv")

        active_tracks = None
        if tracker is not None:
            timer.start("track")
            active_tracks = tracker.update(
                obb_pr_w, ii, cam=cam, T_world_rig=T_wr, observed_points=sdp_w
            )
            t_track = timer.stop("track")

        if args.viz_headless:
            timer.start("viz")
            _t0 = time.perf_counter()

            bb2_texts = [f"{l[:10]} {s:.2f}" for s, l in zip(scores2d, labels2d)]
            bb2_colors = jet_colors_bgr(scores2d)
            bb3_texts = [f"{l[:10]} {s:.2f}" for s, l in zip(scores3d, labels3d)]
            bb3_colors = jet_colors_bgr(scores3d)

            if DEBUG_VIZ:
                _t1 = time.perf_counter()
                print(f"  [viz] colors: {(_t1 - _t0) * 1000:.1f}ms", flush=True)

            viz_2d = img_np.copy()

            viz_2d = render_bb2(
                viz_2d,
                bb2d,
                rotated=rotated,
                texts=bb2_texts,
                clr=bb2_colors,
            )
            put_text(
                viz_2d,
                f"2D Detections ({method} {args.detector_hw}x{args.detector_hw})",
                scale=0.6,
                line=0,
            )
            t_sec = int(datum["time_ns0"]) / 1e9
            put_text(viz_2d, f"frame {ii}, t={t_sec:.3f}s", scale=0.5, line=2)
            max_labels = 64
            if len(text_labels) > max_labels:
                line = -1
            else:
                line = -1 - len(text_labels)
                for jj, label in enumerate(text_labels[:max_labels]):
                    put_text(
                        viz_2d, label, scale=0.4, line=-1 - jj, color=colors[label]
                    )
            if args.gt2d:
                put_text(viz_2d, f"{len(bb2d)} 2DBB PROMPTS", scale=0.4, line=line)
            else:
                put_text(
                    viz_2d,
                    f"{len(text_labels)} TEXT PROMPTS ({taxonomy_name})",
                    scale=0.4,
                    line=line,
                )

            if DEBUG_VIZ:
                _t2 = time.perf_counter()
                print(f"  [viz] 2d_panel: {(_t2 - _t1) * 1000:.1f}ms", flush=True)

            # 3D BB Viz on image.
            viz_3d = img_np.copy()

            # Overlay sparse depth patches on middle frame
            if "sdp_patch0" in outputs:
                if DEBUG_VIZ:
                    _ts0 = time.perf_counter()
                sdp_median = outputs["sdp_patch0"][0].cpu()
                if DEBUG_VIZ:
                    _ts1 = time.perf_counter()
                HH, WW = viz_3d.shape[:2]
                viz_sdp, sdp_resized = render_depth_patches(
                    sdp_median, rotated=rotated, HH=HH, WW=WW
                )
                if DEBUG_VIZ:
                    _ts2 = time.perf_counter()
                viz_sdp = np.ascontiguousarray(viz_sdp)
                mask3 = mask[:, :, None] if (mask := sdp_resized > 0.1).any() else None
                if DEBUG_VIZ:
                    _ts3 = time.perf_counter()
                if mask3 is not None:
                    # Single-pass fused blend+mask: 0.2 ≈ 51/256, 0.8 ≈ 205/256
                    viz_3d = np.where(
                        mask3,
                        (
                            (
                                viz_sdp.astype(np.uint16) * 51
                                + viz_3d.astype(np.uint16) * 205
                            )
                            >> 8
                        ).astype(np.uint8),
                        viz_3d,
                    )
                if DEBUG_VIZ:
                    _ts4 = time.perf_counter()
                    print(
                        f"    [sdp] cpu: {(_ts1 - _ts0) * 1000:.1f}ms  render: {(_ts2 - _ts1) * 1000:.1f}ms  mask: {(_ts3 - _ts2) * 1000:.1f}ms  blend: {(_ts4 - _ts3) * 1000:.1f}ms",
                        flush=True,
                    )

            if DEBUG_VIZ:
                _t3 = time.perf_counter()
                print(f"  [viz] sdp: {(_t3 - _t2) * 1000:.1f}ms", flush=True)

            viz_3d = draw_bb3s(
                viz=viz_3d,
                T_world_rig=T_wr,
                cam=cam,
                obbs=obb_pr_w,
                already_rotated=rotated,
                rotate_label=rotated,
                colors=bb3_colors,
                texts=bb3_texts,
            )
            put_text(
                viz_3d,
                f"3D Detections (Boxer {boxernet.hw}x{boxernet.hw})",
                scale=0.6,
                line=0,
            )
            put_text(
                viz_3d,
                f"Device: '{loader.device_name}', Camera: '{loader.camera}'",
                scale=0.5,
                line=-1,
            )

            if DEBUG_VIZ:
                _t4 = time.perf_counter()
                print(f"  [viz] draw_bb3s: {(_t4 - _t3) * 1000:.1f}ms", flush=True)

            panels = [viz_2d, viz_3d]

            if tracker is not None and active_tracks is not None:
                viz_track = img_np.copy()
                if len(active_tracks) > 0:
                    tracked_obbs = torch.stack([t.obb for t in active_tracks])
                    track_colors = [
                        (np.array(TAB20[t.track_id % len(TAB20)]) * 255).tolist()
                        for t in active_tracks
                    ]
                    track_texts = [
                        f"{t.cached_text[:10]} {t.accumulated_weight / max(t.support_count, 1):.2f}"
                        for t in active_tracks
                    ]
                    viz_track = draw_bb3s(
                        viz=viz_track,
                        T_world_rig=T_wr,
                        cam=cam,
                        obbs=tracked_obbs,
                        already_rotated=rotated,
                        rotate_label=rotated,
                        colors=track_colors,
                        texts=track_texts,
                    )
                put_text(
                    viz_track,
                    f"3D Tracks: {len(active_tracks)} Objects",
                    scale=0.6,
                    line=0,
                )
                panels.append(viz_track)

            final = np.hstack(panels)

            if DEBUG_VIZ:
                _t5 = time.perf_counter()
                print(f"  [viz] panels+hstack: {(_t5 - _t4) * 1000:.1f}ms", flush=True)

            _, jpg_buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_bytes = jpg_buf.tobytes()
            out_path = os.path.join(video_dir, f"{args.write_name}_viz_{ii:05d}.jpg")
            with open(out_path, "wb") as f:
                f.write(jpg_bytes)
            out_path = os.path.join(log_dir, f"{args.write_name}_viz_current.jpg")
            with open(out_path, "wb") as f:
                f.write(jpg_bytes)

            if DEBUG_VIZ:
                _t6 = time.perf_counter()
                print(f"  [viz] imwrite: {(_t6 - _t5) * 1000:.1f}ms", flush=True)
                print(f"  [viz] TOTAL: {(_t6 - _t0) * 1000:.1f}ms", flush=True)

            t_viz = timer.stop("viz")

        timing_str = f"load:{t_load:.0f}ms owl:{t_owl:.0f}ms boxer:{t_boxer:.0f}ms"
        if tracker is not None:
            timing_str += f" track:{t_track:.0f}ms"
        timing_str += f" csv:{t_csv:.0f}ms"
        if args.viz_headless:
            timing_str += f" viz:{t_viz:.0f}ms"
        pbar.set_postfix_str(f"{len(bb2d)} 2D, {obb_pr_w.shape[0]} 3D | " + timing_str)

    if writer is not None:
        writer.close()
        print(f"==> Saved 3D BBs to {csv_path}")
        print(f"==> Saved 2D BBs to {csv2d_out_path}")

    if args.viz_headless:
        # Calculate FPS from RGB timestamps
        if dataset_type in ("omni3d", "scannet"):
            # Omni3D/ScanNet: no real nanosecond timestamps, use fixed framerate
            fps = 10
        elif len(timestamps_ns) >= 2:
            total_time_ns = timestamps_ns[-1] - timestamps_ns[0]
            if total_time_ns > 0:
                fps = max(1, round((len(timestamps_ns) - 1) * 1e9 / total_time_ns))
            else:
                fps = 10  # fallback
        else:
            fps = 10  # fallback

        make_mp4(
            video_dir,
            fps,
            output_dir=log_dir,
            image_glob=f"{args.write_name}_viz_*.jpg",
            output_name=f"{args.write_name}_viz_final.mp4",
        )

    if args.fuse:
        from utils.fuse_3d_boxes import fuse_obbs_from_csv

        print(f"\n==> Running fusion on {csv_path}")
        fuse_obbs_from_csv(csv_path)

    if tracker is not None:
        active_tracks = tracker._get_active_tracks()
        print(f"==> {len(active_tracks)} active tracks from inline tracker")

        if len(active_tracks) > 0:
            base, ext = os.path.splitext(csv_path)
            track_output_path = f"{base}_tracked{ext}"

            tracked_obbs = torch.stack([t.obb for t in active_tracks])
            ids = torch.tensor([t.track_id for t in active_tracks], dtype=torch.int32)
            tracked_obbs.set_inst_id(ids)

            rounded_prob = torch.round(tracked_obbs.prob * 100) / 100
            tracked_obbs.set_prob(rounded_prob.squeeze(-1), use_mask=False)

            track_sem = {}
            for obb in tracked_obbs:
                sid = int(obb.sem_id.item())
                if sid not in track_sem:
                    track_sem[sid] = unpad_string(tensor2string(obb.text.int()))
            track_writer = ObbCsvWriter2(track_output_path)
            track_writer.write(tracked_obbs, timestamps_ns=0, sem_id_to_name=track_sem)
            track_writer.close()
            print(f"==> Saved {len(active_tracks)} tracked OBBs to {track_output_path}")


if __name__ == "__main__":
    main()
