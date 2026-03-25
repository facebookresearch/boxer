#! /usr/bin/env python3

# pyre-unsafe
import argparse
import os

import re

import cv2
import numpy as np
import torch
from utils.settings import CKPT_PATH, EVAL_PATH
from boxernet.boxernet import BoxerNet
from utils.demo_utils import (
    CudaTimer,
    DEFAULT_SEQ,
    expand_seq_shorthand,
    handle_input,
)


from utils.file_io import ObbCsvWriter2, read_obb_csv, load_bb2d_csv, save_bb2d_csv
from utils.image import put_text, torch2cv2

from utils.render import (
    draw_bb3s,
    render_bb2,
    render_depth_patches,
)
from utils.taxonomy import load_text_labels
from utils.tensor_utils import (
    pad_string,
    string2tensor,
    tensor2string,
    unpad_string,
)
from utils.video import make_mp4, safe_delete_folder
from tqdm import tqdm

from loaders.ca_loader import CALoader
from loaders.omni_loader import OMNI3D_DATASETS, OmniLoader
from loaders.scannet_loader import ScanNetLoader
from loaders.aria_loader import AriaLoader


def jet_color(val):
    """Map a scalar in [0, 1] to an RGB tuple via OpenCV's JET colormap."""
    val = max(0.0, min(1.0, float(val)))
    bgr = cv2.applyColorMap(np.uint8([[int(val * 255)]]), cv2.COLORMAP_JET)[0, 0]
    return (float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0)


TAB20 = [
    (0.122, 0.467, 0.706), (0.682, 0.780, 0.910),
    (1.000, 0.498, 0.055), (1.000, 0.733, 0.471),
    (0.173, 0.627, 0.173), (0.596, 0.875, 0.541),
    (0.839, 0.153, 0.157), (1.000, 0.596, 0.588),
    (0.580, 0.404, 0.741), (0.773, 0.690, 0.835),
    (0.549, 0.337, 0.294), (0.769, 0.612, 0.580),
    (0.890, 0.467, 0.761), (0.969, 0.714, 0.824),
    (0.498, 0.498, 0.498), (0.780, 0.780, 0.780),
    (0.737, 0.741, 0.133), (0.859, 0.859, 0.553),
    (0.090, 0.745, 0.812), (0.620, 0.855, 0.898),
]

def detect_frame(datum, det2d, boxernet, text_labels, sem_name_to_id, sem_id_to_name,
                  thresh2d, thresh3d, device, no_sdp=False):
    """Run 2D detection + 3D BoxerNet on a single frame datum.

    Returns: (obb_pr_w: ObbTW, time_ns: int) or (None, time_ns) if no detections.
    """
    time_ns = int(datum["time_ns0"])
    img_torch = datum["img0"]

    if no_sdp:
        datum["sdp_w"] = torch.zeros(0, 3)

    # 2D detection
    img_torch_255 = img_torch.clone() * 255.0
    rotated = datum["rotated0"]
    bb2d, scores2d, label_ints, _ = det2d.forward(
        img_torch_255,
        rotated.item(),
        resize_to_HW=(800, 800),
    )
    labels2d = [text_labels[label_int] for label_int in label_ints]

    if bb2d.shape[0] == 0:
        return (None, time_ns)

    # 3D BoxerNet
    datum["bb2d"] = bb2d
    if device == "mps":
        outputs = boxernet.forward(datum)
    else:
        with torch.autocast(device_type=device, dtype=torch.float32):
            outputs = boxernet.forward(datum)
    obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

    # Set semantic IDs from 2D labels
    assert len(obb_pr_w) == len(labels2d)
    sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
    for i in range(len(labels2d)):
        label = labels2d[i]
        if label in sem_name_to_id:
            sem_ids[i] = sem_name_to_id[label]
        else:
            new_id = len(sem_name_to_id)
            sem_name_to_id[label] = new_id
            sem_id_to_name[new_id] = label
            sem_ids[i] = new_id
    obb_pr_w.set_sem_id(sem_ids)

    # Filter by 3D confidence
    scores3d = obb_pr_w.prob.squeeze(-1).clone()
    keepers = obb_pr_w.prob.squeeze(-1) >= thresh3d
    obb_pr_w = obb_pr_w[keepers].clone()
    scores3d = scores3d[keepers].clone()
    labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
    mean_scores = (scores2d[keepers] + scores3d) / 2.0
    obb_pr_w.set_prob(mean_scores)

    # Set text labels
    if len(labels3d) > 0:
        text_data = torch.stack(
            [string2tensor(pad_string(lab, max_len=128)) for lab in labels3d]
        )
        obb_pr_w.set_text(text_data)

    return (obb_pr_w, time_ns)


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
    parser.add_argument("--thresh2d", type=float, default=0.2, help="detection confidence for 2d detector")
    parser.add_argument("--thresh3d", type=float, default=0.5, help="detection confidence for boxer")
    parser.add_argument("--labels", type=comma_separated_list, nargs="?", const=[], default=["lvisplus"], help="Optional comma-separated list of text prompts (e.g. --labels=small or --labels=chair,table,lamp)")
    parser.add_argument("--detector_hw", type=int, default=800, help="resize images before going into 2D detector")
    parser.add_argument("--write_name", default="boxer", type=str, help="name prefix for outputs")
    parser.add_argument("--viz_headless", action="store_true", help="run OpenCV 2D panel visualization")
    parser.add_argument("--viz_3d", action="store_true", help="launch interactive 3D viewer after pipeline")
    parser.add_argument("--cache2d", action="store_true", help="load 2D BBs from CSV instead of running detector")
    parser.add_argument("--cache3d", action="store_true", help="load 3D BBs from CSV instead of running BoxerNet")
    parser.add_argument("--no_sdp", action="store_true", help="turn off SDP input")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU")
    parser.add_argument("--gt2d", action="store_true", help="use GT pseudo 2DBB as input")
    parser.add_argument("--fuse", action="store_true", help="run 3D box fusion after processing")
    parser.add_argument("--track", action="store_true", help="run online 3D box tracking and show tracked boxes in Top Down View")
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_PATH, "bxr_alln2nw12bs12hw960in2x6d768ni1_Nov20.ckpt"), help="path to BoxerNet checkpoint")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"], help="Inference precision (default: float32)")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Output directory for results (default: ~/viz_boxer)")
    args = parser.parse_args()

    if args.fuse and args.track:
        parser.error("--fuse and --track are mutually exclusive")
    if args.cache3d:
        args.cache2d = True
    if args.viz_3d and not args.track and not args.fuse:
        args.fuse = True
    print(args)
    # fmt: on


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
        remote_root = handle_input(expand_seq_shorthand(args.input))
        # Resolve bare sequence names to ~/boxy_data/<name>
        if not os.path.isabs(remote_root) and not os.path.exists(remote_root):
            resolved = os.path.expanduser(os.path.join("~/boxy_data", remote_root))
            if os.path.exists(resolved):
                remote_root = resolved
        seq_name = remote_root.rstrip("/").split("/")[-1]

    # get name of containing directory
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    csv2d_out_path = os.path.join(log_dir, f"{args.write_name}_2dbbs.csv")
    print(f"==> Created output folder {log_dir}")

    # --cache3d: skip detection + BoxerNet + loader, go straight to post-processing
    if args.cache3d:
        print(f"==> Loading cached 3D BBs from {csv_path}")
        cached_timed_obbs = read_obb_csv(csv_path)
        total_dets = sum(len(obbs) for obbs in cached_timed_obbs.values())
        print(f"==> Loaded {len(cached_timed_obbs)} frames, {total_dets} detections from cache")

        if args.fuse:
            from utils.fuse_3d_boxes import fuse_obbs_from_csv
            print(f"\n==> Running fusion on {csv_path}")
            fuse_obbs_from_csv(csv_path)

        if args.viz_3d:
            _launch_3d_viewer(args, None, dataset_type, seq_name, log_dir, csv_path, csv2d_out_path)
        return

    # Create data loader
    if dataset_type == "scannet":
        loader = ScanNetLoader(
            scene_dir=args.input,
            annotation_path=os.path.expanduser("~/data/scannet/full_annotations.json"),
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
            pinhole=args.pinhole,
            resize=(args.detector_hw, args.detector_hw),
            use_canny=False,
        )
    else:
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

    # choose a model checkpoint
    if torch.backends.mps.is_available() and not args.force_cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"==> Using device {device}")

    # Handle --labels=gt: extract unique names from loader's sem_id_to_name
    if args.labels == ["gt"]:
        if hasattr(loader, "sem_id_to_name") and loader.sem_id_to_name:
            # Extract unique names from sem_id_to_name, filtering out structural classes
            loader_label_names = []
            for _sem_id, name in loader.sem_id_to_name.items():
                # Skip floor/wall structural classes
                if name.lower() in ["floor", "wall"]:
                    continue
                if name not in loader_label_names:
                    loader_label_names.append(name)
            if len(loader_label_names) == 0:
                raise ValueError(
                    "No valid semantic labels found in loader.sem_id_to_name after filtering"
                )
            args.labels = loader_label_names
            print(
                f"==> Using {len(loader_label_names)} labels from loader.sem_id_to_name"
            )
        else:
            raise ValueError(
                "--labels=gt requires the loader to have sem_id_to_name attribute. "
                "This is available for CALoader (ca1m sequences) and AriaLoader with --gt2d."
            )

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
        from detectors.owl_wrapper import OwlWrapper
        det2d = OwlWrapper(
            device, text_prompts=text_labels, min_confidence=args.thresh2d,
            precision=args.precision,
        )
        method = "OWLv2"

    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    loader.resize = boxernet.hw
    print(f"==> Will resize images to {loader.resize}x{loader.resize} for boxernet")

    # Print model architecture
    total_params = sum(p.numel() for p in boxernet.parameters())
    print("=" * 50)
    print(f"  BOXERNET ARCHITECTURE ({total_params / 1e6:.2f}M params)")
    print("=" * 50)
    for name, module in boxernet.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({n_params / 1e6:.2f}M)")
    print("=" * 50)

    video_dir = os.path.join(log_dir, f"{args.write_name}_viz")
    if args.viz_headless:
        safe_delete_folder(
            video_dir, extensions=[".png"], keep_folder=True, recursive=True
        )
        os.makedirs(video_dir, exist_ok=True)
        print(
            f"==> Current frame: {os.path.join(log_dir, f'{args.write_name}_viz_current.png')}"
        )

    colors = {
        label: (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
        for label in text_labels
    }

    if args.gt2d:
        sem_name_to_id = loader.sem_name_to_id
        sem_id_to_name = {val: key for key, val in sem_name_to_id.items()}
    else:
        sem_name_to_id = {label: i for i, label in enumerate(text_labels)}
        sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}

    # Online mode: skip batch loop, launch viewer with detect_fn
    if args.viz_3d and not args.cache3d:
        def _make_detect_fn(det2d_model, boxernet_model, text_labels, sem_name_to_id, sem_id_to_name, args, device):
            def detect_fn(datum):
                return detect_frame(datum, det2d_model, boxernet_model, text_labels,
                                    sem_name_to_id, sem_id_to_name,
                                    args.thresh2d, args.thresh3d, device, args.no_sdp)
            return detect_fn
        detect_fn = _make_detect_fn(det2d, boxernet, text_labels, sem_name_to_id, sem_id_to_name, args, device)
        online_writer = ObbCsvWriter2(csv_path)
        _launch_3d_viewer(args, loader, dataset_type, seq_name, log_dir, csv_path, csv2d_out_path,
                          detect_fn=detect_fn, loader_iter=iter(loader), writer=online_writer)
        return

    writer = ObbCsvWriter2(csv_path)

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
        out_path = os.path.join(video_dir, f"{args.write_name}_viz_{ii:05d}.png")
        cv2.imwrite(out_path, final)
        out_path = os.path.join(log_dir, f"{args.write_name}_viz_current.png")
        cv2.imwrite(out_path, final)

    timestamps_ns = []  # Collect timestamps to compute FPS
    timer = CudaTimer(device)
    pbar = tqdm(range(len(loader)), desc="BoxerNet")
    for ii in pbar:
        # Data loading
        timer.start("load")
        try:
            datum = next(loader)
        except StopIteration:
            break

        if datum is False:
            pbar.set_postfix_str("Skipped (time misalignment)")
            continue

        # Collect timestamp for FPS calculation
        if args.viz_headless and "time_ns0" in datum:
            timestamps_ns.append(int(datum["time_ns0"]))

        img_torch = datum["img0"]
        rotated = datum["rotated0"]
        HH, WW = img_torch.shape[2], img_torch.shape[3]
        img_np = torch2cv2(img_torch, rotate=rotated, ensure_rgb=True)
        t_load = timer.stop("load")

        sdp_w_viz = datum["sdp_w"].float()  # Keep original SDP for visualization
        if args.no_sdp:
            # TURN OFF SDP inputs by removing them
            print("==> Removing SDP inputs")
            datum["sdp_w"] = torch.zeros(0, 3)

        # 2D Detection
        timer.start("det2d")
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
                t_det2d = timer.stop("det2d")
                pbar.set_postfix_str(
                    f"0 GT obbs | load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms"
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
                t_det2d = timer.stop("det2d")
                pbar.set_postfix_str(
                    f"0 valid bb2d | load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms"
                )
                if args.viz_headless:
                    write_empty_frame(img_np, HH, WW, ii)
                continue

            scores2d = 0.5 * torch.ones(bb2d.shape[0])
        else:
            img_torch_255 = img_torch.clone() * 255.0
            bb2d, scores2d, label_ints, _ = det2d.forward(
                img_torch_255,
                rotated.item(),
                resize_to_HW=(args.detector_hw, args.detector_hw),
            )
            labels2d = [text_labels[label_int] for label_int in label_ints]

        t_det2d = timer.stop("det2d")

        if bb2d.shape[0] == 0:
            pbar.set_postfix_str(f"0 dets | load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms")
            if args.viz_headless:
                write_empty_frame(img_np, HH, WW, ii)
            continue

        # 3D BoxerNet
        timer.start("boxer")
        sdp_w = datum["sdp_w"].float()
        cam = datum["cam0"].float()
        T_wr = datum["T_world_rig0"].float()
        datum["bb2d"] = bb2d
        precision_dtype = (
            torch.bfloat16 if args.precision == "bfloat16" else torch.float32
        )
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
        timer.start("viz")
        time_ns = int(datum["time_ns0"])
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

        active_tracks = None
        if tracker is not None:
            timer.start("track")
            active_tracks = tracker.update(
                obb_pr_w, ii, cam=cam, T_world_rig=T_wr, observed_points=sdp_w
            )
            t_track = timer.stop("track")

        if args.viz_headless:
            bb2_texts = [f"{l} {s:.2f}" for s, l in zip(scores2d, labels2d)]
            bb2_colors = [(np.array(jet_color(1.0 - s)) * 255).tolist() for s in scores2d]
            bb3_texts = [f"{l} {s:.2f}" for s, l in zip(scores3d, labels3d)]
            bb3_colors = [(np.array(jet_color(1.0 - s)) * 255).tolist() for s in scores3d]

            viz_2d = img_np.copy()

            viz_2d = render_bb2(
                viz_2d,
                bb2d,
                rotated=rotated,
                texts=bb2_texts,
                clr=bb2_colors,
            )
            put_text(viz_2d, f"{method} [{args.detector_hw}x{args.detector_hw}]", scale=0.6, line=0)
            put_text(viz_2d, f"frame {ii}, t={int(datum['time_ns0'])}", scale=0.5, line=1)
            max_labels = 64
            if len(text_labels) > max_labels:
                line = -1
            else:
                line = -1 - len(text_labels)
                for jj, label in enumerate(text_labels[:max_labels]):
                    put_text(viz_2d, label, scale=0.4, line=-1 - jj, color=colors[label])
            if args.gt2d:
                put_text(viz_2d, f"{len(bb2d)} 2DBB PROMPTS", scale=0.4, line=line)
            else:
                put_text(viz_2d, f"{len(text_labels)} TEXT PROMPTS ({taxonomy_name})", scale=0.4, line=line)

            # 3D BB Viz on image.
            viz_3d = img_np.copy()

            # Overlay sparse depth patches on middle frame
            if "sdp_patch0" in outputs:
                sdp_median = outputs["sdp_patch0"][0].cpu()
                HH, WW = viz_3d.shape[:2]
                viz_sdp = render_depth_patches(sdp_median, rotated=rotated, HH=HH, WW=WW)
                viz_sdp = np.ascontiguousarray(viz_sdp)  # already BGR from cv2.applyColorMap
                sdp_resized = torch.nn.functional.interpolate(
                    sdp_median[None], size=(HH, WW), mode="nearest"
                )[0, 0].numpy()
                if rotated:
                    sdp_resized = np.rot90(sdp_resized, k=-1)  # 90 degrees CW
                mask = sdp_resized > 0.1
                viz_3d[mask] = (
                    viz_sdp[mask] * 0.2 + viz_3d[mask].astype(np.float32) * 0.8
                ).astype(np.uint8)

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
            put_text(viz_3d, f"BoxerNet [{boxernet.hw}x{boxernet.hw}], {obb_pr_w.shape[0]} 3DBBs", scale=0.6, line=0)
            put_text(viz_3d, f"Device: '{loader.device_name}', Camera: '{loader.camera}'", scale=0.5,
                line=-1,
            )

            panels = [viz_2d, viz_3d]

            if tracker is not None and active_tracks is not None:
                viz_track = img_np.copy()
                confirmed = [t for t in active_tracks if t.state.name == "ACTIVE"]
                if len(confirmed) > 0:
                    tracked_obbs = torch.stack([t.obb for t in confirmed])
                    track_colors = [
                        (np.array(TAB20[t.track_id % len(TAB20)]) * 255).tolist()
                        for t in confirmed
                    ]
                    track_texts = [
                        f"{t.cached_text} (n={t.support_count})"
                        for t in confirmed
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
                put_text(viz_track, f"Tracked: {len(confirmed)} objects", scale=0.6, line=0)
                panels.append(viz_track)

            final = np.hstack(panels)

            out_path = os.path.join(video_dir, f"{args.write_name}_viz_{ii:05d}.png")
            cv2.imwrite(out_path, final)
            out_path = os.path.join(log_dir, f"{args.write_name}_viz_current.png")
            cv2.imwrite(out_path, final)
        t_viz = timer.stop("viz")

        timing_str = f"load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms boxer:{t_boxer:.0f}ms"
        if tracker is not None:
            timing_str += f" track:{t_track:.0f}ms"
        timing_str += f" viz:{t_viz:.0f}ms"
        pbar.set_postfix_str(f"{len(bb2d)} 2D, {obb_pr_w.shape[0]} 3D | " + timing_str)

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
            image_glob=f"{args.write_name}_viz_*.png",
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

    if args.viz_3d:
        _launch_3d_viewer(args, loader, dataset_type, seq_name, log_dir, csv_path, csv2d_out_path)


def _build_seq_ctx(loader, dataset_type):
    """Build viewer-compatible sequence context from existing loader."""
    if dataset_type == "aria":
        rgb_stream_id = loader.stream_id[0]
        rgb_num_frames = loader.provider.get_num_data(rgb_stream_id)
        rgb_timestamps = (
            np.array(loader.pose_ts, dtype=np.int64)
            if getattr(loader, "pose_ts", None) is not None and len(loader.pose_ts) > 0
            else np.array([0], dtype=np.int64)
        )
        data = {
            "source": "aria",
            "loader": loader,
            "rgb_num_frames": rgb_num_frames,
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,
            "is_nebula": bool(loader.is_nebula),
            "traj": loader.traj,
            "pose_ts": loader.pose_ts,
            "calibs": loader.calibs[0],
            "calib_ts": loader.calib_ts,
            "time_to_uids_slaml": getattr(loader, "time_to_uids_slaml", None),
            "time_to_uids_slamr": getattr(loader, "time_to_uids_slamr", None),
            "uid_to_p3": getattr(loader, "uid_to_p3", None),
        }
        return data
    elif dataset_type == "ca1m":
        rgb_timestamps = np.array(loader.timestamp_ns)
        n = len(rgb_timestamps)
        traj = [(loader.Ts_wc[i] @ loader.cams[i].T_camera_rig).float() for i in range(n)]
        return {
            "source": "ca1m",
            "rgb_num_frames": n,
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": loader.rgb_images,
            "is_nebula": True,
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": loader.cams,
            "calib_ts": rgb_timestamps,
            "loader": loader,
            "time_to_uids_slaml": None,
            "time_to_uids_slamr": None,
            "uid_to_p3": None,
        }
    elif dataset_type == "scannet":
        from tw.camera import CameraTW
        from tw.pose import PoseTW
        frame_ids = list(loader.frame_ids)
        # Read first image to get dimensions
        first_fid = frame_ids[0]
        color_path = os.path.join(loader.scene_dir, "frames", "color", f"{first_fid}.png")
        if not os.path.exists(color_path):
            color_path = os.path.join(loader.scene_dir, "frames", "color", f"{first_fid}.jpg")
        first_bgr = cv2.imread(color_path)
        H, W = first_bgr.shape[:2]
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_data = torch.tensor(
            [W, H, loader.fx, loader.fy, loader.cx, loader.cy, -1, 1e-3, W, H],
            dtype=torch.float32,
        )
        cam_template = CameraTW(torch.cat([cam_data, T_cam_rig])).float()

        rgb_timestamps = np.array([int(fid) for fid in frame_ids], dtype=np.int64)
        traj = []
        calibs = []
        for fid in frame_ids:
            pose_path = os.path.join(loader.scene_dir, "frames", "pose", f"{fid}.txt")
            T_world_cam = np.loadtxt(pose_path).astype(np.float32)
            T_world_cam[:3, 3] -= loader.world_offset.astype(np.float32)
            R_flat = T_world_cam[:3, :3].reshape(-1)
            t_vec = T_world_cam[:3, 3]
            T_wr_data = torch.tensor([*R_flat, *t_vec], dtype=torch.float32)
            traj.append(PoseTW(T_wr_data).float())
            calibs.append(cam_template.clone())
        return {
            "source": "scannet",
            "loader": loader,
            "scannet_scene_dir": loader.scene_dir,
            "scannet_frame_ids": list(frame_ids),
            "rgb_num_frames": len(frame_ids),
            "rgb_timestamps": rgb_timestamps,
            "rgb_images": None,
            "is_nebula": True,
            "traj": traj,
            "pose_ts": rgb_timestamps,
            "calibs": calibs,
            "calib_ts": rgb_timestamps,
            "time_to_uids_slaml": None,
            "time_to_uids_slamr": None,
            "uid_to_p3": None,
        }
    else:
        return None


def _launch_3d_viewer(args, loader, dataset_type, seq_name, log_dir, csv_path, csv2d_out_path,
                      detect_fn=None, loader_iter=None, writer=None):
    """Launch interactive 3D viewer after pipeline completes."""
    import sys
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # prevent moderngl-window from consuming args

    from utils.viz_boxer import TrackerViewer
    import moderngl_window as mglw

    if detect_fn is not None:
        # Online mode: start with empty timed_obbs
        timed_obbs = {}
    else:
        timed_obbs = read_obb_csv(csv_path)
    seq_ctx = _build_seq_ctx(loader, dataset_type) if loader is not None else None

    bb2d_csv_path = csv2d_out_path if os.path.exists(csv2d_out_path) else ""

    class Viewer(TrackerViewer):
        window_size = (2250, 1100)
        def __init__(self, **kw):
            super().__init__(
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                bb2d_csv_path=bb2d_csv_path,
                freeze_tracker=not args.track,
                detect_fn=detect_fn,
                loader_iter=loader_iter,
                writer=writer,
                **kw,
            )
            if not args.track:
                self.show_raw_set = True

    try:
        mglw.run_window_config(Viewer)
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
