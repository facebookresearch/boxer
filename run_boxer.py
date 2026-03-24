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

from detectors.detic_wrapper import DeticWrapper

from utils.file_io import ObbCsvWriter2, save_bb2d_csv
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
    parser.add_argument("--detector", type=str, default="owl", choices=["owl", "detic"], help="2D detector to use (default: owl)")
    parser.add_argument("--thresh2d", type=float, default=0.2, help="detection confidence for 2d detector")
    parser.add_argument("--thresh3d", type=float, default=0.5, help="detection confidence for boxer")
    parser.add_argument("--labels", type=comma_separated_list, nargs="?", const=[], default=["lvisplus"], help="Optional comma-separated list of text prompts (e.g. --labels=small or --labels=chair,table,lamp)")
    parser.add_argument("--detector_hw", type=int, default=800, help="resize images before going into 2D detector")
    parser.add_argument("--write_name", default="boxer", type=str, help="name prefix for outputs")
    parser.add_argument("--skip_viz", action="store_true", help="skip visualization")
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
    print(args)
    # fmt: on


    # Determine dataset type from input string
    if bool(re.search(r"scene\d{4}_\d{2}", args.input)) or "/scannet/" in args.input:
        dataset_type = "scannet"
    elif args.input in OMNI3D_DATASETS:
        dataset_type = "omni3d"
    elif args.input.startswith("ca1m"):
        dataset_type = "ca1m"
    else:
        dataset_type = "aria"

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
        seq_name = args.input
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
        seq_name = args.input
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
        remote_root = handle_input(expand_seq_shorthand(args.input))
        seq_name = remote_root.rstrip("/").split("/")[-1]
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

    if args.gt2d:
        method = "GT2D"
    elif args.detector == "detic":
        det2d = DeticWrapper(
            device=device,
            model_width=args.detector_hw,
            model_height=args.detector_hw,
            selected_classes=text_labels,
            min_confidence=args.thresh2d,
        )
        method = "DETIC"
    else:
        from detectors.owl_wrapper import OwlWrapper
        det2d = OwlWrapper(
            device, text_prompts=text_labels, min_confidence=args.thresh2d
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

    # get name of containing directory
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)
    video_dir = os.path.join(log_dir, f"{args.write_name}_viz")
    if not args.skip_viz:
        safe_delete_folder(
            video_dir, extensions=[".png"], keep_folder=True, recursive=True
        )
        os.makedirs(video_dir, exist_ok=True)
        print(
            f"==> Current frame: {os.path.join(log_dir, f'{args.write_name}_viz_current.png')}"
        )
    print(f"==> Created output folder {log_dir}")

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

    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    writer = ObbCsvWriter2(csv_path)
    csv2d_out_path = os.path.join(log_dir, f"{args.write_name}_2dbbs.csv")

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
        if not args.skip_viz and "time_ns0" in datum:
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
        if args.gt2d:
            # Check if there are any valid GT objects for this frame
            obbs_valid = datum["obbs"].remove_padding()
            if len(obbs_valid) == 0:
                t_det2d = timer.stop("det2d")
                pbar.set_postfix_str(
                    f"0 GT obbs | load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms"
                )
                if not args.skip_viz:
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
                if not args.skip_viz:
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
            # DETIC returns class names directly, OWL returns indices into text_labels
            if args.detector == "detic":
                labels2d = list(label_ints)
            else:
                labels2d = [text_labels[label_int] for label_int in label_ints]

        t_det2d = timer.stop("det2d")

        if bb2d.shape[0] == 0:
            pbar.set_postfix_str(f"0 dets | load:{t_load:.0f}ms det2d:{t_det2d:.0f}ms")
            if not args.skip_viz:
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
                # For DETIC, dynamically add new labels to the mapping
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

        if not args.skip_viz:
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

    if not args.skip_viz:
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


if __name__ == "__main__":
    main()
