# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import contextlib
import os
import threading

# Silence VRS/projectaria_tools logging before importing
os.environ.setdefault("GLOG_minloglevel", "2")  # Suppress INFO and WARNING
os.environ.setdefault("VRS_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch
from projectaria_tools.core import data_provider

from utils.file_io import (
    load_closed_loop_trajectory,
    load_obbs_adt,
    load_online_calib,
    load_semidense,
    probe_gravity_direction,
)
from utils.tw.camera import CameraTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import find_nearest2


def get_T_zup_yup():
    """Get transformation from Y-up to Z-up coordinate system.

    Transforms coordinates so that:
    - Gravity (0, -9.81, 0) in Y-up becomes (0, 0, -9.81) in Z-up
    - new_x = old_x
    - new_y = -old_z
    - new_z = old_y
    """
    R_zup_yup = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    ).unsqueeze(0)
    t_zup_yup = torch.zeros(1, 3)
    return PoseTW.from_Rt(R_zup_yup, t_zup_yup)


@contextlib.contextmanager
def _suppress_stderr():
    """Context manager to suppress C++ library stderr output."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


from loaders.base_loader import BaseLoader


class AriaLoader(BaseLoader):
    def __init__(
        self,
        remote_root,
        camera="rgb",
        with_img=True,
        with_traj=True,
        with_sdp=True,
        with_obb=False,
        use_description=False,
        pinhole=False,
        pinhole_fxy=None,
        resize=None,
        unrotate=False,
        skip_n=1,
        max_n=999999,
        start_n=0,
        remove_structure=True,
        force_reload=False,
        target_time_ns=None,
        start_ts=None,
        is_adt="auto",
        obb_vis_thresh=0.0001,  # remove visibility 0 boxes.
        obb_min_size=0.02,  # remove small boxes based on 2D box size.
        obb_max_depth=None,  # Max distance (meters) from camera to keep OBBs
        restrict_range=True,  # restrict frames to intersection of modality time ranges.
    ):
        self.camera = camera
        self.with_img = with_img
        self.with_traj = with_traj
        self.with_sdp = with_sdp
        self.with_obb = with_obb
        self.use_description = use_description
        self.force_reload = force_reload
        self.obb_vis_thresh = obb_vis_thresh
        self.obb_min_size = obb_min_size
        self.obb_max_depth = obb_max_depth
        # Note: self.is_adt is set later after auto-detection
        self.restrict_range = restrict_range
        print("==> loading AriaLoader with the following settings:")
        print(f"camera: {camera}")
        print(f"with_img: {with_img}")
        print(f"with_traj: {with_traj}")
        print(f"with_sdp: {with_sdp}")
        print(f"with_obb: {with_obb}")
        print(f"use_description: {use_description}")
        print(f"pinhole: {pinhole}")
        print(f"resize: {resize}")
        print(f"unrotate: {unrotate}")
        print(f"skip_n: {skip_n}")
        print(f"max_n: {max_n}")
        print(f"start_n: {start_n}")
        print(f"start_ts: {start_ts}")
        print(f"is_adt: {is_adt} (auto-detected if 'auto')")
        print(f"restrict_range: {restrict_range}")

        if remote_root.endswith("vrs"):
            vrs_path = remote_root
            remote_root = os.path.dirname(vrs_path)
        else:
            vrs_path = remote_root + "/main.vrs"

        seq_name = os.path.basename(remote_root)
        local_root = os.path.expanduser("~/boxy_data")
        local_root = os.path.join(local_root, seq_name)
        # Data is assumed to be local
        self.vrs_path = vrs_path
        self.remote_root = remote_root
        self.local_root = local_root
        self.seq_name = seq_name
        self.pinhole = pinhole
        self.pinhole_fxy = pinhole_fxy
        self.resize = resize
        self.unrotate = unrotate
        self.skip_n = skip_n
        self.max_n = max_n

        print(
            f"==> SST loader using vrs file '{vrs_path}' and local seq '{self.local_root}'"
        )

        with _suppress_stderr():
            self.provider = data_provider.create_vrs_data_provider(vrs_path)

        try:
            device_type = self.provider.get_file_tags()["device_type"].lower()
        except Exception:
            device_type = "unknown"
        self.is_nebula = device_type in ["oatmeal", "aria gen 2"]
        self.device_name = "Aria Gen 2" if self.is_nebula else "Aria Gen 1"
        print(f"==> Device: {self.device_name} (type={device_type})")

        calib_path = os.path.join(remote_root, "online_calibration.jsonl")
        slaml_calibs, slamr_calibs, rgb_calibs, calib_ns = load_online_calib(calib_path)
        self.calibs = []
        self.stream_id = []
        # fmt: off
        if self.camera == "rgb" or self.camera == "frameset":
            self.stream_id.append(self.provider.get_stream_id_from_label("camera-rgb"))
            self.calibs.append(rgb_calibs)
        if self.camera == "slaml" or self.camera == "frameset":
            if self.is_nebula:
                self.stream_id.append(
                    self.provider.get_stream_id_from_label("slam-front-left")
                )
            else:
                self.stream_id.append(
                    self.provider.get_stream_id_from_label("camera-slam-left")
                )
            self.calibs.append(slaml_calibs)
        if self.camera == "slamr" or self.camera == "frameset":
            if self.is_nebula:
                self.stream_id.append(
                    self.provider.get_stream_id_from_label("slam-front-right")
                )
            else:
                self.stream_id.append(
                    self.provider.get_stream_id_from_label("camera-slam-right")
                )
            self.calibs.append(slamr_calibs)

        if len(self.calibs) == 0:
            raise ValueError("No valid camera found")

        # If target_time_ns is provided, binary search to find the frame
        if target_time_ns is not None:
            start_n = self._find_frame_by_timestamp(target_time_ns)
            max_n = 1
            skip_n = 1
            print(f"==> target_time_ns={target_time_ns} -> start_n={start_n}")

        # If start_ts is provided, find the corresponding frame index
        if start_ts is not None:
            start_n = self._find_frame_by_timestamp(start_ts)
            print(f"==> start_ts={start_ts} -> start_n={start_n}")

        total_frames = self.provider.get_num_data(self.stream_id[0])
        self.length = total_frames
        # account for start_n, skip_n, and max_n
        indices = list(range(start_n, total_frames, skip_n))
        indices = indices[:max_n]
        self.iter_length = len(indices)
        print("==> total number of frames in AriaLoader: %d" % total_frames)
        print("==> number of frames in AriaLoader: %d" % self.iter_length)

        self.calib_ts = calib_ns.numpy()

        # Auto-detect gravity direction if is_adt="auto"
        if is_adt == "auto":
            traj_path = remote_root + "/closed_loop_trajectory.csv"
            gravity_dir = probe_gravity_direction(traj_path)
            if gravity_dir == "y":
                print("=" * 60)
                print("AUTO-DETECTED Y-UP GRAVITY COORDINATE SYSTEM")
                print("=" * 60)
                print("  Gravity is primarily in the Y-direction (e.g., optitrack data)")
                print("  Automatically enabling is_adt=True for coordinate transform")
                print("=" * 60)
                self.is_adt = True
            else:
                self.is_adt = False
        else:
            self.is_adt = is_adt

        # Override obb_vis_thresh and obb_max_depth for ADT data
        if self.is_adt:
            self.obb_vis_thresh = 0.3
            self.obb_max_depth = 5
            print(f"==> is_adt=True: overriding obb_vis_thresh to {self.obb_vis_thresh}")
            print(f"==> is_adt=True: overriding obb_max_depth to {self.obb_max_depth}")

        if self.with_traj:
            print("==> loading trajectory")
            traj_path = remote_root + "/closed_loop_trajectory.csv"
            traj, pose_ts = load_closed_loop_trajectory(
                traj_path, subsample=5, grav_y=self.is_adt
            )
            # Transform Y-up to Z-up if needed
            if self.is_adt:
                print("==> Transforming trajectory from Y-up to Z-up")
                T_zup_yup = get_T_zup_yup().to(traj.R.dtype)
                traj = T_zup_yup @ traj
            self.traj = traj
            self.pose_ts = pose_ts.numpy()

        if self.with_sdp:
            print("==> loading points")
            global_path = os.path.join(remote_root, "semidense_points.csv.gz")
            obs_path = os.path.join(remote_root, "semidense_observations.csv.gz")
            time_to_uids_slaml, time_to_uids_slamr, uid_to_p3 = load_semidense(
                global_path, obs_path, calib_path, force_reload=self.force_reload
            )
            self.time_to_uids_slaml = time_to_uids_slaml
            self.time_to_uids_slamr = time_to_uids_slamr
            self.uid_to_p3 = uid_to_p3

            # Pre-compute combined time_to_uids for non-SLAM camera modes (done once)
            time_to_uids_combined = {}
            for k, v in time_to_uids_slaml.items():
                time_to_uids_combined[k] = v
            for k, v in time_to_uids_slamr.items():
                if k in time_to_uids_combined:
                    time_to_uids_combined[k] = list(set(time_to_uids_combined[k] + v))
                else:
                    time_to_uids_combined[k] = v
            self.time_to_uids_combined = time_to_uids_combined

            # Pre-compute sorted timestamp arrays for fast lookup
            self.sdp_times_slaml = np.array(sorted(time_to_uids_slaml.keys()))
            self.sdp_times_slamr = np.array(sorted(time_to_uids_slamr.keys()))
            self.sdp_times_combined = np.array(sorted(time_to_uids_combined.keys()))

            # Build uid_to_p3 as numpy array for fast lookup
            # Create a mapping from uid to index and store p3 values in array
            all_uids = list(uid_to_p3.keys())
            self.uid_to_idx = {uid: idx for idx, uid in enumerate(all_uids)}
            self.p3_array = np.array([uid_to_p3[uid] for uid in all_uids], dtype=np.float32)

            # Transform Y-up to Z-up if needed
            if self.is_adt:
                print("==> Transforming semidense points from Y-up to Z-up")
                # Apply rotation: new_x = old_x, new_y = -old_z, new_z = old_y
                old_x = self.p3_array[:, 0].copy()
                old_y = self.p3_array[:, 1].copy()
                old_z = self.p3_array[:, 2].copy()
                self.p3_array[:, 0] = old_x
                self.p3_array[:, 1] = -old_z
                self.p3_array[:, 2] = old_y

        if self.with_obb:
            print("==> loading 3DBB")

            timed_obbs, sem_name_to_id = load_obbs_adt(
                self.local_root, return_sem2id=True, use_description=self.use_description,
                force_reload=self.force_reload,
                view_filter=self.camera,
                visibility_thresh=self.obb_vis_thresh,
            )

            obb_ts, obbs = [], []
            for key, val in sorted(timed_obbs.items()):
                if remove_structure and len(val) > 0:
                    val = self.filter_obbs_by_name(val)
                obb_ts.append(key)
                obbs.append(val)
            if remove_structure:
                from loaders.base_loader import STRUCTURE_CLASSES
                print(f"==> filtering out structure classes by name: {sorted(STRUCTURE_CLASSES)}")

            # Filter OBBs by obb_min_size (minimum 2D bounding box dimension ratio)
            if self.obb_min_size > 0:
                # Get image dimensions from first calibration
                first_calib = self.calibs[0][0]  # First camera, first timestamp
                img_w, img_h = first_calib.size[0].item(), first_calib.size[1].item()
                avg_dim = (img_w + img_h) / 2.0
                min_bb2_dim = self.obb_min_size * avg_dim
                print(f"==> filtering OBBs with bb2 min dimension < {self.obb_min_size:.2%} of avg image dim ({min_bb2_dim:.1f} pixels)")

                # Track which instance IDs have at least one valid bb2 meeting the threshold
                valid_inst_ids = set()
                for obb_batch in obbs:
                    if len(obb_batch) == 0:
                        continue
                    for i in range(obb_batch.shape[0]):
                        obb = obb_batch[i]
                        inst_id = int(obb.inst_id.item())
                        if inst_id in valid_inst_ids:
                            continue  # Already validated
                        # Check all camera views
                        for bb2 in [obb.bb2_rgb, obb.bb2_slaml, obb.bb2_slamr]:
                            # bb2 format: [xmin, xmax, ymin, ymax], -1 if not visible
                            if bb2[0].item() >= 0:  # Valid bb2
                                width = bb2[1].item() - bb2[0].item()
                                height = bb2[3].item() - bb2[2].item()
                                min_dim = min(width, height)
                                if min_dim >= min_bb2_dim:
                                    valid_inst_ids.add(inst_id)
                                    break

                # Filter obbs to keep only those with valid instance IDs
                filtered_obbs = []
                total_removed = 0
                for obb_batch in obbs:
                    if len(obb_batch) == 0:
                        filtered_obbs.append(obb_batch)
                        continue
                    keep_mask = torch.tensor([
                        int(obb_batch[i].inst_id.item()) in valid_inst_ids
                        for i in range(obb_batch.shape[0])
                    ], dtype=torch.bool)
                    total_removed += (~keep_mask).sum().item()
                    filtered_obbs.append(obb_batch[keep_mask])
                obbs = filtered_obbs
                print(f"==> obb_min_size filter: kept {len(valid_inst_ids)} instances, removed {total_removed} OBB entries")

            self.obb_ts = np.array(obb_ts)
            self.obbs = obbs
            self.sem_name_to_id = sem_name_to_id

            # Transform Y-up to Z-up if needed
            if self.is_adt:
                print("==> Transforming OBBs from Y-up to Z-up")
                for i, obb in enumerate(self.obbs):
                    # Transform T_world_object by pre-multiplying with T_zup_yup
                    T_zup_yup = get_T_zup_yup().to(obb.T_world_object.R.dtype)
                    self.obbs[i].set_T_world_object(T_zup_yup @ obb.T_world_object)

        # Restrict frame range to intersection of all modality time ranges
        self.end_index = self.length - 1  # Default to full range
        if self.restrict_range:
            valid_range = self._compute_valid_time_range()
            if valid_range is not None:
                valid_start_ns, valid_end_ns = valid_range
                # Find frame indices that fall within the valid time range
                valid_start_frame = self._find_frame_by_timestamp(valid_start_ns)
                valid_end_frame = self._find_frame_by_timestamp(valid_end_ns)
                # Clamp valid_end to VRS bounds
                valid_end_frame = min(valid_end_frame, self.length - 1)
                # Apply start_n as minimum starting point
                new_start = max(valid_start_frame, start_n)
                new_end = valid_end_frame
                # Check if we have a valid range
                if new_start <= new_end:
                    # Recalculate indices with skip_n and max_n
                    old_iter_length = self.iter_length
                    indices = list(range(new_start, new_end + 1, skip_n))
                    indices = indices[:max_n]
                    self.iter_length = len(indices)
                    if self.iter_length > 0:
                        start_n = new_start
                        self.end_index = indices[-1]
                        print(
                            f"==> restrict_range: adjusted frame range from {old_iter_length} "
                            f"to {self.iter_length} frames (start={new_start}, end={self.end_index})"
                        )
                    else:
                        print("==> Warning: restrict_range resulted in 0 frames")
                else:
                    print(
                        f"==> Warning: no valid frame range (start={new_start} > end={new_end})"
                    )

        # Compute the actual sampled time range based on start_n, skip_n, iter_length
        sampled_range = None
        if self.iter_length > 0 and len(self.stream_id) > 0:
            stream_id = self.stream_id[0]
            # Get timestamp of first sampled frame
            _, first_record = self.provider.get_image_data_by_index(stream_id, start_n)
            first_sampled_ns = first_record.capture_timestamp_ns
            # Get timestamp of last sampled frame
            last_idx = start_n + (self.iter_length - 1) * skip_n
            last_idx = min(last_idx, self.end_index)
            _, last_record = self.provider.get_image_data_by_index(stream_id, last_idx)
            last_sampled_ns = last_record.capture_timestamp_ns
            sampled_range = (first_sampled_ns, last_sampled_ns)

        # Print modality time ranges visualization with sampled range markers
        self._print_modality_ranges(restricted_range=sampled_range)

        self.index = start_n
        self.count = 0

        # Prefetch: load next frame in background thread
        self._prefetch_result = None
        self._prefetch_thread = None
        self._start_prefetch()

    def _compute_valid_time_range(self):
        """Compute intersection of all enabled modality time ranges.

        Returns (start_ns, end_ns) tuple or None if no modalities are enabled.
        """
        ranges = []

        # img: Get first/last frame timestamps from VRS provider
        if self.with_img and len(self.stream_id) > 0:
            stream_id = self.stream_id[0]
            num_frames = self.provider.get_num_data(stream_id)
            if num_frames > 0:
                _, first_record = self.provider.get_image_data_by_index(stream_id, 0)
                _, last_record = self.provider.get_image_data_by_index(
                    stream_id, num_frames - 1
                )
                ranges.append(
                    (
                        first_record.capture_timestamp_ns,
                        last_record.capture_timestamp_ns,
                    )
                )

        # traj: Use self.pose_ts min/max
        if self.with_traj and hasattr(self, "pose_ts") and len(self.pose_ts) > 0:
            ranges.append((self.pose_ts.min(), self.pose_ts.max()))

        # sdp: Use self.sdp_times_combined min/max
        if (
            self.with_sdp
            and hasattr(self, "sdp_times_combined")
            and len(self.sdp_times_combined) > 0
        ):
            ranges.append(
                (self.sdp_times_combined.min(), self.sdp_times_combined.max())
            )

        # obb: Use self.obb_ts min/max
        if self.with_obb and hasattr(self, "obb_ts") and len(self.obb_ts) > 0:
            ranges.append((self.obb_ts.min(), self.obb_ts.max()))

        if not ranges:
            return None

        # Compute intersection: max of starts, min of ends
        intersection_start = max(start for start, _ in ranges)
        intersection_end = min(end for _, end in ranges)

        if intersection_start >= intersection_end:
            print("==> Warning: no overlapping time range between modalities")
            return None

        return (intersection_start, intersection_end)

    def _print_modality_ranges(self, restricted_range=None):
        """Print ASCII visualization of time ranges for all enabled modalities.

        Args:
            restricted_range: Optional (start_ns, end_ns) tuple to show as sampled range.
        """
        modalities = {}

        # img: Get first/last frame timestamps from VRS provider
        if self.with_img and len(self.stream_id) > 0:
            stream_id = self.stream_id[0]
            num_frames = self.provider.get_num_data(stream_id)
            if num_frames > 0:
                _, first_record = self.provider.get_image_data_by_index(stream_id, 0)
                _, last_record = self.provider.get_image_data_by_index(
                    stream_id, num_frames - 1
                )
                modalities["img"] = (
                    first_record.capture_timestamp_ns,
                    last_record.capture_timestamp_ns,
                    num_frames,
                )

        # traj: Use self.pose_ts min/max
        if self.with_traj and hasattr(self, "pose_ts") and len(self.pose_ts) > 0:
            modalities["traj"] = (
                self.pose_ts.min(),
                self.pose_ts.max(),
                len(self.pose_ts),
            )

        # sdp: Use self.sdp_times_combined min/max
        if (
            self.with_sdp
            and hasattr(self, "sdp_times_combined")
            and len(self.sdp_times_combined) > 0
        ):
            modalities["sdp"] = (
                self.sdp_times_combined.min(),
                self.sdp_times_combined.max(),
                len(self.sdp_times_combined),
            )

        # obb: Use self.obb_ts min/max
        if self.with_obb and hasattr(self, "obb_ts") and len(self.obb_ts) > 0:
            modalities["obb"] = (self.obb_ts.min(), self.obb_ts.max(), len(self.obb_ts))

        if not modalities:
            return

        # Find global min/max across all modalities
        global_min = min(start for start, _, _ in modalities.values())
        global_max = max(end for _, end, _ in modalities.values())
        total_duration_ns = global_max - global_min

        if total_duration_ns <= 0:
            return

        # Convert to seconds for display
        total_duration_s = total_duration_ns / 1e9

        # Print header
        print("\n==> Modality Time Ranges:")
        bar_width = 50

        # Build labels with Hz
        labels = {}
        for name, (start_ns, end_ns, count) in modalities.items():
            duration_s = (end_ns - start_ns) / 1e9
            if duration_s > 0 and count > 1:
                labels[name] = f"{name} [{count / duration_s:.0f}Hz]"
            else:
                labels[name] = name
        label_width = max(len(l) for l in labels.values()) + 1

        # Print time scale header
        print(f"{' ' * label_width} 0.0s{' ' * (bar_width - 8)}{total_duration_s:.1f}s")

        # Print each modality
        for name, (start_ns, end_ns, count) in modalities.items():
            # Calculate relative positions (0 to 1)
            rel_start = (start_ns - global_min) / total_duration_ns
            rel_end = (end_ns - global_min) / total_duration_ns

            # Convert to bar positions
            bar_start = int(rel_start * bar_width)
            bar_end = int(rel_end * bar_width)
            bar_length = max(1, bar_end - bar_start)

            # Build the bar string
            bar = (
                " " * bar_start
                + "=" * bar_length
                + " " * (bar_width - bar_start - bar_length)
            )

            # Convert times to relative seconds
            start_s = (start_ns - global_min) / 1e9
            end_s = (end_ns - global_min) / 1e9

            print(f"{labels[name]:<{label_width}}|{bar}|  {start_s:.1f} - {end_s:.1f}s")

        # Print restricted range marker if provided
        if restricted_range is not None:
            restrict_start_ns, restrict_end_ns = restricted_range
            # Calculate relative positions
            rel_start = (restrict_start_ns - global_min) / total_duration_ns
            rel_end = (restrict_end_ns - global_min) / total_duration_ns
            # Convert to bar positions
            marker_start = int(rel_start * bar_width)
            marker_end = int(rel_end * bar_width)
            # Clamp to valid range
            marker_start = max(0, min(marker_start, bar_width - 2))
            marker_end = max(marker_start + 1, min(marker_end, bar_width - 1))
            # Build marker string with [ and ]
            marker_bar = list(" " * bar_width)
            marker_bar[marker_start] = "["
            marker_bar[marker_end] = "]"
            # Fill between markers with -
            for i in range(marker_start + 1, marker_end):
                marker_bar[i] = "-"
            marker_str = "".join(marker_bar)
            # Convert times to relative seconds
            start_s = (restrict_start_ns - global_min) / 1e9
            end_s = (restrict_end_ns - global_min) / 1e9
            print(
                f"{'sampled':<{label_width}}|{marker_str}|  {start_s:.1f} - {end_s:.1f}s"
            )

        print()

    def _find_frame_by_timestamp(self, target_ns):
        """Binary search to find frame index closest to target timestamp."""
        stream_id = self.stream_id[0]
        lo, hi = 0, self.provider.get_num_data(stream_id) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            _, record = self.provider.get_image_data_by_index(stream_id, mid)
            if record.capture_timestamp_ns < target_ns:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def __len__(self):
        return self.iter_length

    def __iter__(self):
        return self

    def _single(self, idx, stream_id, timed_calibs):
        data, record = self.provider.get_image_data_by_index(stream_id, idx)

        if not data.is_valid():
            print("==> Warning: invalid image data")
            return False
        ts_ns = record.capture_timestamp_ns
        output = {}
        output["time_ns"] = ts_ns

        if not self.with_img:
            return output

        img = data.to_numpy_array()
        HH, WW = img.shape[0], img.shape[1]
        resize = self.resize
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Determine target size
        if resize is None:
            resizeH = HH
            resizeW = WW
        elif isinstance(resize, tuple):
            resizeH = resize[0]
            resizeW = resize[1]
        else:
            resizeH = resize
            resizeW = resize

        # Resize with cv2 on numpy (faster than torch interpolate on CPU)
        if (resizeH != HH or resizeW != WW) and not self.pinhole:
            img = cv2.resize(img, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)

        img_torch = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        img_torch = img_torch / 255.0  # to 0-1

        if self.is_nebula:
            rotated = torch.tensor([False])
        else:
            rotated = torch.tensor([True])

        calib_idx = find_nearest2(self.calib_ts, ts_ns)
        cam_fish = timed_calibs[calib_idx].float()
        cam = cam_fish.float()

        if self.pinhole:
            # Compute ratios from camera's intrinsic size, not raw image size
            cam_w = cam_fish.size[0].item()
            cam_h = cam_fish.size[1].item()
            w_ratio = resizeW / cam_w
            h_ratio = resizeH / cam_h

            cx = cam_fish.c[0] * w_ratio
            cy = cam_fish.c[1] * h_ratio
            if self.pinhole_fxy is not None:
                # Use directly specified focal length
                fxy = self.pinhole_fxy
            elif self.camera == "slaml" or self.camera == "slamr":
                fxy = cam_fish.f[0] * 0.8
            else:
                fxy = cam_fish.f[0] * 1.2
            fx = fxy * w_ratio
            fy = fxy * h_ratio
            intr_pin = [fx, fy, cx, cy]  # fx, fy, cx, cy
            cam_pin = CameraTW.from_surreal(
                height=resizeH,
                width=resizeW,
                type_str="Pinhole",
                params=intr_pin,
                T_camera_rig=cam_fish.T_camera_rig,
            ).float()
            xx, yy = torch.meshgrid(
                torch.arange(resizeW), torch.arange(resizeH), indexing="ij"
            )
            xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()  # (H*W, 2)
            rays, valid = cam_pin.unproject(xy[None])  # (H*W, 3), (H*W,)
            xy, valid2 = cam_fish.project(rays)  # (H*W, 2), (H*W,)
            xy = xy[0]
            valid = valid[0] & valid2[0]
            xy[~valid] = -1
            # normalize to [-1, 1]
            xy[:, 0] = (xy[:, 0] / (WW - 1)) * 2 - 1
            xy[:, 1] = (xy[:, 1] / (HH - 1)) * 2 - 1
            uv = (
                xy.view(1, resizeW, resizeH, 2).permute(0, 2, 1, 3).float()
            )  # (1, H, W, 2)
            img_torch = torch.nn.functional.grid_sample(
                img_torch,
                uv,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            cam = cam_pin.float()
            output["fisheye_cam"] = cam_fish  # Store fisheye cam for bb2 transformation
        else:
            # Log original calibration principal point (first frame only)
            if self.count == 0:
                calib_w, calib_h = cam.size[0].item(), cam.size[1].item()
                expected_cx_orig = (calib_w - 1) / 2.0
                expected_cy_orig = (calib_h - 1) / 2.0
                actual_cx_orig = cam.c[0].item()
                actual_cy_orig = cam.c[1].item()
            # First scale camera from calibration size to actual VRS image size
            cam = cam.scale_to_size((WW, HH))
            # Then scale to resize size if specified
            if resize is not None:
                cam = cam.scale_to_size((resizeW, resizeH))
            # Log principal point info (first frame only)
            if self.count == 0:
                expected_cx = (resizeW - 1) / 2.0
                expected_cy = (resizeH - 1) / 2.0
                actual_cx = cam.c[0].item()
                actual_cy = cam.c[1].item()
                pp_thresh = 0.02 * max(resizeW, resizeH)
                if (
                    abs(actual_cx - expected_cx) > pp_thresh
                    or abs(actual_cy - expected_cy) > pp_thresh
                ):
                    print(
                        f"==> Warning: Principal point is off-center. "
                        f"Before resize: expected=({expected_cx_orig:.1f}, {expected_cy_orig:.1f}), actual=({actual_cx_orig:.1f}, {actual_cy_orig:.1f}). "
                        f"After resize to {resizeW}x{resizeH}: expected=({expected_cx:.1f}, {expected_cy:.1f}), actual=({actual_cx:.1f}, {actual_cy:.1f})"
                    )
        cam = cam.float()
        if rotated and self.unrotate:
            output["pinhole_cam_prerot"] = cam.clone()
            cam = cam.rotate_90_cw()
            img_torch = torch.rot90(img_torch, k=3, dims=(2, 3))
            rotated = torch.tensor([False])
        output["img"] = img_torch.float()
        output["cam"] = cam
        output["rotated"] = rotated
        output["orig_size"] = (WW, HH)  # Original VRS image size for bb2 scaling

        return output

    def load(self, idx):
        output = {}

        for ni, (si, ca) in enumerate(zip(self.stream_id, self.calibs)):
            out = self._single(idx, si, ca)
            if out is False:
                return False
            for key in out:
                output[f"{key}{ni}"] = out[key]

        ts_ns = output[
            "time_ns0"
        ]  # use the first one as the timestamp for points and obb

        if self.with_traj:
            for ni in range(len(self.stream_id)):
                ts_ns = output[f"time_ns{ni}"]
                nearest_idx = find_nearest2(self.pose_ts, ts_ns)
                pose_ns = self.pose_ts[nearest_idx]
                delta_s = abs(pose_ns - ts_ns) / 1e9
                if delta_s > 0.02:
                    print(
                        f"==> Warning: large time diff between image and traj: {delta_s:.3f} sec"
                    )
                    return False
                T_world_rig = self.traj[nearest_idx]
                output[f"T_world_rig{ni}"] = T_world_rig.float()

        if self.with_sdp:
            # Use pre-computed timestamp arrays and combined dict
            if self.camera == "slaml":
                time_to_uids = self.time_to_uids_slaml
                sdp_times = self.sdp_times_slaml
            elif self.camera == "slamr":
                time_to_uids = self.time_to_uids_slamr
                sdp_times = self.sdp_times_slamr
            else:
                time_to_uids = self.time_to_uids_combined
                sdp_times = self.sdp_times_combined
            nearest_sdp_idx = find_nearest2(sdp_times, ts_ns)
            sdp_ns = sdp_times[nearest_sdp_idx]
            delta_s = abs(sdp_ns - ts_ns) / 1e9
            if delta_s > 0.02:
                print(
                    f"==> Warning: large time diff between image and sdp: {delta_s:.3f} sec"
                )
                return False
            uids = time_to_uids[sdp_ns]
            # Use pre-computed numpy array for fast lookup
            indices = [self.uid_to_idx[uid] for uid in uids]
            p3d = torch.from_numpy(self.p3_array[indices, :3])
            output["sdp_w"] = p3d

        if self.with_obb:
            ts_ns = output[f"time_ns{ni}"]
            nearest_idx = find_nearest2(self.obb_ts, ts_ns)
            obb_ns = self.obb_ts[nearest_idx]
            delta_s = abs(obb_ns - ts_ns) / 1e9
            if delta_s > 0.1:
                print(
                    f"==> Warning: large time diff between image and obb: {delta_s:.3f} sec"
                )
            obb = self.obbs[nearest_idx].float()

            # Filter OBBs by max depth from camera
            if self.obb_max_depth is not None and obb is not None and len(obb) > 0:
                T_world_rig = output["T_world_rig0"]
                T_world_cam = T_world_rig @ output["cam0"].T_camera_rig.inverse()
                cam_pos = T_world_cam.t  # Camera position in world coords (1, 3)
                obb_pos = obb.T_world_object.t  # OBB positions in world coords (N, 3)
                distances = torch.norm(obb_pos - cam_pos, dim=-1)  # (N,)
                keep_mask = distances <= self.obb_max_depth
                obb = obb[keep_mask]

            output["obbs"] = obb

            # Populate bb2d0 from OBB's pre-computed 2D bounding boxes
            if obb is not None and len(obb) > 0:
                # Select the appropriate bb2 based on camera
                if self.camera == "rgb":
                    bb2d_all = (
                        obb.bb2_rgb.clone()
                    )  # (N, 4) format: [xmin, xmax, ymin, ymax]
                elif self.camera == "slaml":
                    bb2d_all = obb.bb2_slaml.clone()
                elif self.camera == "slamr":
                    bb2d_all = obb.bb2_slamr.clone()
                else:
                    bb2d_all = obb.bb2_rgb.clone()  # Default to rgb for frameset

                # Get original image size from output (set in _single)
                orig_size = output.get("orig_size0", None)
                if orig_size is not None:
                    orig_w, orig_h = orig_size

                    # Apply resize scaling if resize is set (but not if pinhole - handled below)
                    if self.resize is not None and "fisheye_cam0" not in output:
                        if isinstance(self.resize, tuple):
                            target_h, target_w = self.resize
                        else:
                            target_h = target_w = self.resize
                        # Scale bb2 coordinates: [xmin, xmax, ymin, ymax]
                        # Check for valid entries (not -1) before scaling
                        valid = bb2d_all[:, 0] >= 0
                        if valid.any():
                            scale_x = target_w / orig_w
                            scale_y = target_h / orig_h
                            bb2d_all[valid, 0] = bb2d_all[valid, 0] * scale_x  # xmin
                            bb2d_all[valid, 1] = bb2d_all[valid, 1] * scale_x  # xmax
                            bb2d_all[valid, 2] = bb2d_all[valid, 2] * scale_y  # ymin
                            bb2d_all[valid, 3] = bb2d_all[valid, 3] * scale_y  # ymax

                # Transform bb2d from fisheye to pinhole coordinates if pinhole mode
                if "fisheye_cam0" in output:
                    fisheye_cam = output["fisheye_cam0"]
                    pinhole_cam = output.get("pinhole_cam_prerot0", output["cam0"])
                    valid = bb2d_all[:, 0] >= 0
                    if valid.any():
                        # Get 4 corners of each bbox: [xmin, xmax, ymin, ymax]
                        # Corners: TL(xmin,ymin), TR(xmax,ymin), BL(xmin,ymax), BR(xmax,ymax)
                        tl = bb2d_all[valid][:, [0, 2]]  # (N, 2) [xmin, ymin]
                        tr = bb2d_all[valid][:, [1, 2]]  # (N, 2) [xmax, ymin]
                        bl = bb2d_all[valid][:, [0, 3]]  # (N, 2) [xmin, ymax]
                        br = bb2d_all[valid][:, [1, 3]]  # (N, 2) [xmax, ymax]

                        # Unproject through fisheye to rays
                        tl_rays, _ = fisheye_cam.unproject(tl[None])
                        tr_rays, _ = fisheye_cam.unproject(tr[None])
                        bl_rays, _ = fisheye_cam.unproject(bl[None])
                        br_rays, _ = fisheye_cam.unproject(br[None])

                        # Project through pinhole
                        tl_pin, _ = pinhole_cam.project(tl_rays)
                        tr_pin, _ = pinhole_cam.project(tr_rays)
                        bl_pin, _ = pinhole_cam.project(bl_rays)
                        br_pin, _ = pinhole_cam.project(br_rays)

                        # Stack all corners and compute new bbox
                        all_corners = torch.stack(
                            [tl_pin[0], tr_pin[0], bl_pin[0], br_pin[0]], dim=1
                        )  # (N, 4, 2)
                        xmin_new = all_corners[:, :, 0].min(dim=1).values
                        xmax_new = all_corners[:, :, 0].max(dim=1).values
                        ymin_new = all_corners[:, :, 1].min(dim=1).values
                        ymax_new = all_corners[:, :, 1].max(dim=1).values

                        # Clamp to image bounds
                        img_w, img_h = (
                            pinhole_cam.size[0].item(),
                            pinhole_cam.size[1].item(),
                        )
                        xmin_new = xmin_new.clamp(0, img_w - 1)
                        xmax_new = xmax_new.clamp(0, img_w - 1)
                        ymin_new = ymin_new.clamp(0, img_h - 1)
                        ymax_new = ymax_new.clamp(0, img_h - 1)

                        # Update bb2d_all for valid entries
                        bb2d_all[valid, 0] = xmin_new
                        bb2d_all[valid, 1] = xmax_new
                        bb2d_all[valid, 2] = ymin_new
                        bb2d_all[valid, 3] = ymax_new

                # Rotate bb2d to match 90° CW image rotation when unrotate is active
                if not self.is_nebula and self.unrotate:
                    valid = bb2d_all[:, 0] >= 0
                    if valid.any():
                        # Height of the pre-rotation image
                        prerot_cam = output.get("pinhole_cam_prerot0", output["cam0"])
                        H_before = int(prerot_cam.size[1])
                        old = bb2d_all[valid].clone()
                        bb2d_all[valid, 0] = H_before - 1 - old[:, 3]
                        bb2d_all[valid, 1] = H_before - 1 - old[:, 2]
                        bb2d_all[valid, 2] = old[:, 0]
                        bb2d_all[valid, 3] = old[:, 1]

                output["bb2d0"] = bb2d_all
            else:
                # No OBBs, set empty bb2d0
                output["bb2d0"] = torch.empty(0, 4)

        return output

    def _start_prefetch(self):
        """Start prefetching the frame at self.index in a background thread."""
        if self.index > self.end_index or self.count >= self.max_n:
            return
        idx = self.index
        self._prefetch_result = None
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(idx,), daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self, idx):
        """Background worker that loads a single frame."""
        try:
            self._prefetch_result = self.load(idx=idx)
        except Exception as e:
            self._prefetch_result = e

    def __next__(self):
        if self.index > self.end_index or self.count >= self.max_n:
            raise StopIteration

        # Wait for prefetched result
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            out = self._prefetch_result
            self._prefetch_thread = None
            self._prefetch_result = None
        else:
            out = self.load(idx=self.index)

        # Re-raise exceptions from the worker thread
        if isinstance(out, Exception):
            raise out

        self.index += self.skip_n
        self.count += 1

        # Kick off prefetch for the next frame
        self._start_prefetch()

        return out
