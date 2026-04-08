# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import csv
import gzip
import io
import json
import os
import pickle
import time
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from utils.taxonomy import (
    SSI_COLORS,
    SSI_NAME2SEM,
    SSI_SEM2NAME,
)
from utils.tw.camera import CameraTW
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW, quat_to_rotmat, rotmat_to_quat
from utils.tw.tensor_utils import (
    find_nearest,
    pad_string,
    string2tensor,
)

# pyre-unsafe


def probe_gravity_direction(traj_path, num_samples=100):
    """
    Probe the trajectory file to detect the gravity direction without fully loading it.

    Args:
        traj_path: Path to the closed_loop_trajectory.csv file
        num_samples: Number of gravity samples to read (default: 100)

    Returns:
        str: "z" if gravity is in Z-direction, "y" if in Y-direction, "x" if in X-direction,
             or "unknown" if gravity data is not available or direction is unclear
    """
    with open(traj_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")

    # Find gravity column indices if they exist
    gravity_x_idx = None
    if "gravity_x_world" in header:
        gravity_x_idx = header.index("gravity_x_world")
    else:
        return "unknown"  # No gravity data available

    lines = lines[1:]  # Remove header

    # Sample gravity values
    gravity_samples = []
    step = max(1, len(lines) // num_samples)
    for ii in range(0, min(len(lines), num_samples * step), step):
        line = lines[ii].split(",")
        try:
            gx = float(line[gravity_x_idx])
            gy = float(line[gravity_x_idx + 1])
            gz = float(line[gravity_x_idx + 2])
            gravity_samples.append([gx, gy, gz])
        except (IndexError, ValueError):
            continue

    if len(gravity_samples) == 0:
        return "unknown"

    # Compute mean gravity vector
    gravity = np.array(gravity_samples)
    mean_gravity = np.mean(gravity, axis=0)
    gx, gy, gz = mean_gravity

    # Determine primary gravity direction
    abs_gx, abs_gy, abs_gz = abs(gx), abs(gy), abs(gz)

    if abs_gz >= abs_gx and abs_gz >= abs_gy:
        return "z"
    elif abs_gy >= abs_gx and abs_gy >= abs_gz:
        return "y"
    elif abs_gx >= abs_gy and abs_gx >= abs_gz:
        return "x"
    else:
        return "unknown"


def load_closed_loop_trajectory(traj_path, subsample=1, grav_y=False):
    with open(traj_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")

    # Find gravity column indices if they exist
    gravity_x_idx = None
    if "gravity_x_world" in header:
        gravity_x_idx = header.index("gravity_x_world")

    if len(header) == 8 and header[0] == "tracking_timestamp_us":  # dioptraSX
        ts_line = 0
        tx_line = 1
        qx_line = 5
        qw_line = 4
        ts_mult = 1000.0
    elif len(header) == 8:  # ase
        ts_line = 0
        tx_line = 1
        qx_line = 5
        qw_line = 4
        ts_mult = 1.0
    elif len(header) == 20:  # ase with gravity
        ts_line = 1
        tx_line = 3
        qx_line = 6
        qw_line = 9
        ts_mult = 1000.0
    elif len(header) == 28:  # location toolbox
        ts_line = 1
        tx_line = 3
        qx_line = 6
        qw_line = 9
        ts_mult = 1000.0
        assert header[ts_line] == "tracking_timestamp_us"
        assert header[tx_line] == "tx_world_device"
        assert header[qx_line] == "qx_world_device"
        assert header[qw_line] == "qw_world_device"
    elif len(header) == 29:  # older location toolbox
        ts_line = 2
        tx_line = 4
        qx_line = 7
        qw_line = 10
        ts_mult = 1000.0
        assert header[ts_line] == "tracking_timestamp_us"
        assert header[tx_line] == "tx_world_device"
        assert header[qx_line] == "qx_world_device"
        assert header[qw_line] == "qw_world_device"
    else:
        raise IOError(f"Unable to read trajectory file, got {len(header)} columns")

    lines = lines[1:]  # Remove header.

    # get framerate
    init_ts = []
    for ii, line in enumerate(lines):
        line = line.split(",")
        init_ts.append(int(line[ts_line]))
        if ii > 100:
            break
    diffs = np.array(init_ts[1:]) - np.array(init_ts[:-1])
    med_diff_us = np.median(diffs)
    est_hz = int(1.0 / (med_diff_us / (1000.0 * 1000.0)))
    print("trajectory estimated hz is %d" % est_hz)

    timestamps_ns, Rs, ts = [], [], []
    gravity_samples = []
    for ii in range(0, len(lines), subsample):
        line = lines[ii]
        line = line.split(",")
        timestamp = int(line[ts_line])
        tx = float(line[tx_line])
        ty = float(line[tx_line + 1])
        tz = float(line[tx_line + 2])
        qx = float(line[qx_line])
        qy = float(line[qx_line + 1])
        qz = float(line[qx_line + 2])
        qw = float(line[qw_line])
        Rs.append(quat_to_rotmat(qw, qx, qy, qz))
        ts.append([tx, ty, tz])
        timestamps_ns.append(ts_mult * timestamp)
        # Read gravity if available
        if gravity_x_idx is not None:
            gx = float(line[gravity_x_idx])
            gy = float(line[gravity_x_idx + 1])
            gz = float(line[gravity_x_idx + 2])
            gravity_samples.append([gx, gy, gz])
    Rs = torch.tensor(np.array(Rs))
    ts = torch.tensor(np.array(ts))
    T_world_rigs = PoseTW.from_Rt(Rs, ts)
    timestamps_ns = torch.tensor(timestamps_ns, dtype=torch.long)

    # Check if gravity is in the z-direction
    if gravity_samples:
        gravity = np.array(gravity_samples)
        mean_gravity = np.mean(gravity, axis=0)
        gx, gy, gz = mean_gravity
        gravity_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        print(
            f"==> Gravity vector: ({gx:.2f}, {gy:.2f}, {gz:.2f}), magnitude: {gravity_mag:.2f}"
        )

        # Check if gravity is primarily in z-direction
        if abs(gz) < max(abs(gx), abs(gy)):
            if abs(gy) > abs(gx) and abs(gy) > abs(gz):
                gravity_axis = "Y"
            elif abs(gx) > abs(gy) and abs(gx) > abs(gz):
                gravity_axis = "X"
            else:
                gravity_axis = "unknown"

            error_msg = (
                f"Gravity is NOT in the z-direction!\n"
                f"  gravity = ({gx:.2f}, {gy:.2f}, {gz:.2f})\n"
                f"  Gravity appears to be in the {gravity_axis}-direction\n"
                f"  Expected gravity to be primarily in the Z-direction"
            )

            if gravity_axis == "Y" and not grav_y:
                raise ValueError(
                    error_msg + "\n"
                    "  To use Y-up coordinate system, set is_adt=True in SSTLoader"
                )
            elif gravity_axis == "Y" and grav_y:
                pass  # Y-up with grav_y=True is expected, no warning needed
            else:
                print("=" * 60)
                print(f"WARNING: {error_msg}")
                print("=" * 60)

    return T_world_rigs, timestamps_ns


def load_online_calib(calib_path, vrs_path=None):
    with open(calib_path, "r") as f:
        lines = f.readlines()

    first_calib = json.loads(lines[0])
    intr_type = first_calib["CameraCalibrations"][0]["Projection"]["Name"]
    assert intr_type == "FisheyeRadTanThinPrism"

    assert first_calib["CameraCalibrations"][0]["Label"] in [
        "camera-slam-left",
        "0",
        "slam-front-left",
    ]
    assert first_calib["CameraCalibrations"][1]["Label"] in [
        "camera-slam-right",
        "1",
        "slam-front-right",
    ]
    slaml_idx = 0
    slamr_idx = 1
    if len(first_calib["CameraCalibrations"]) == 3:
        assert first_calib["CameraCalibrations"][2]["Label"] in ["camera-rgb", "4"]
        has_rgb = True
        num_calib = 3
        rgb_idx = 2
    elif len(first_calib["CameraCalibrations"]) == 5:
        assert first_calib["CameraCalibrations"][4]["Label"] in ["camera-rgb", "4"]
        has_rgb = True
        num_calib = 3
        rgb_idx = 4
    else:
        has_rgb = False
        num_calib = 2

    slaml_calibs = []
    slamr_calibs = []
    rgb_calibs = []
    timestamps_ns = []
    for line in lines:
        line = json.loads(line)

        param = []
        T_cam_rig = []
        for i in [slaml_idx, slamr_idx, rgb_idx]:
            calib = line["CameraCalibrations"][i]
            assert calib["Calibrated"] == True
            param.append(calib["Projection"]["Params"])
            tx, ty, tz = calib["T_Device_Camera"]["Translation"]
            qw, (qx, qy, qz) = calib["T_Device_Camera"]["UnitQuaternion"]
            rot_mat = quat_to_rotmat(qw, qx, qy, qz)
            trans = torch.tensor([tx, ty, tz])
            T_rig_cam = PoseTW.from_Rt(rot_mat, trans).float()
            T_cam_rig.append(T_rig_cam.inverse())

        # Try to guess SLAM HxW.
        if "front" in first_calib["CameraCalibrations"][0]["Label"]:
            # Nebula
            slam_hw = (512, 512)
        else:
            # Aria V1
            mean_fxy_slam = (param[0][0] + param[1][0]) / 2.0
            if mean_fxy_slam < 300:
                slam_hw = (480, 640)
            elif mean_fxy_slam < 800:
                slam_hw = (1024, 1280)
            else:
                raise IOError("Unknown camera resolution, focal lengths too large")

        vr_slam = torch.sqrt(torch.tensor(slam_hw[0] ** 2 + slam_hw[1] ** 2)) / 2
        vr_slam *= 1.2

        slaml_calibs.append(
            CameraTW.from_surreal(
                width=slam_hw[1],
                height=slam_hw[0],
                type_str="Fisheye624",
                params=param[0],
                T_camera_rig=T_cam_rig[0],
                valid_radius=vr_slam.reshape(1),
            )
        )
        slamr_calibs.append(
            CameraTW.from_surreal(
                width=slam_hw[1],
                height=slam_hw[0],
                type_str="Fisheye624",
                params=param[1],
                T_camera_rig=T_cam_rig[1],
                valid_radius=vr_slam.reshape(1),
            )
        )

        if has_rgb:
            # Try to guess RGB HxW based on cx and cy.
            cx_rgb = param[2][1]  # principal point x
            cy_rgb = param[2][2]  # principal point y
            mean_cxy = (cx_rgb + cy_rgb) / 2.0  # mean of cx and cy
            approx_hw = (
                mean_cxy * 2.0
            )  # multiple center position by 2 to get approx dimension

            if calib["Label"] == "4":
                guess_h, guess_w = 1748, 2328
            elif rgb_idx == 4:  # nebula
                # Nebula can have different resolutions - infer from principal point
                # 2016x1512: cx ≈ 1008, cy ≈ 756
                # 2560x1920: cx ≈ 1280, cy ≈ 960
                if cx_rgb > 1150 or cy_rgb > 850:
                    # Principal point suggests higher resolution (2560x1920)
                    guess_h, guess_w = 1920, 2560
                else:
                    guess_h, guess_w = 1512, 2016
            elif abs((approx_hw - 704.0) / 704.0) < 0.1:
                guess_h, guess_w = 704, 704
            elif abs((approx_hw - 1408.0) / 1408.0) < 0.1:
                guess_h, guess_w = 1408, 1408
            elif abs((approx_hw - 2880) / 2880) < 0.1:
                guess_h, guess_w = 2880, 2880
            else:
                raise IOError("Unknown RGB size")
            vr_rgb = torch.sqrt(torch.tensor(guess_h**2 + guess_w**2)) / 2

            rgb_calibs.append(
                CameraTW.from_surreal(
                    width=guess_w,
                    height=guess_h,
                    type_str="Fisheye624",
                    params=param[2],
                    T_camera_rig=T_cam_rig[2],
                    valid_radius=vr_rgb.reshape(1),
                )
            )
        timestamps_ns.append(1000 * int(line["tracking_timestamp_us"]))

    slaml_calibs = torch.stack(slaml_calibs)
    slamr_calibs = torch.stack(slamr_calibs)
    if has_rgb:
        rgb_calibs = torch.stack(rgb_calibs)
    else:
        rgb_calibs = None
    timestamps_ns = torch.tensor(timestamps_ns, dtype=torch.long)
    return slaml_calibs, slamr_calibs, rgb_calibs, timestamps_ns


def load_semidense(
    global_path,
    obs_path,
    calib_path,
    max_depth_std=0.05,
    max_inv_depth_std=0.005,
    force_reload=False,
):
    if os.path.exists(calib_path):
        serial2label = {}
        with open(calib_path, "r") as f:
            lines = f.readlines()
            line = lines[0]
            line = json.loads(line)
            calib = line["CameraCalibrations"]
            for j in range(len(calib)):
                serial = calib[j]["SerialNumber"]
                label = calib[j]["Label"]
                serial2label[serial] = label
            assert len(serial2label)
    else:
        # ase_v1
        serial2label = {
            "16": "camera-slam-left",
            "17": "camera-slam-right",
            "18": "camera-rgb",
        }

    print("Loading semi-dense point global properties")
    point_start = time.time()
    cache_dir = os.path.dirname(global_path)
    point_cache = os.path.join(
        cache_dir,
        f"cache_global_ds{max_depth_std:.04f}_ids{max_inv_depth_std:.05f}.pkl",
    )
    if os.path.exists(point_cache) and not force_reload:
        # load from the cached file
        with gzip.open(point_cache, "rb") as f:
            uid_to_p3 = pickle.load(f)
    else:
        with gzip.open(global_path, "rt") as f:
            header = f.readline().strip().split(",")
            col_uid = header.index("uid")
            col_px = header.index("px_world")
            col_py = header.index("py_world")
            col_pz = header.index("pz_world")
            col_ids = header.index("inv_dist_std")
            col_ds = header.index("dist_std")
            cols = [col_uid, col_px, col_py, col_pz, col_ids, col_ds]
            raw = f.read()
        data = np.loadtxt(io.StringIO(raw), delimiter=",", usecols=cols)
        mask = (data[:, 4] <= max_inv_depth_std) & (data[:, 5] <= max_depth_std)
        data = data[mask]
        uid_to_p3 = {int(r[0]): (r[1], r[2], r[3], r[4], r[5]) for r in data}
        print(f"==> Writing to {point_cache}")
        with gzip.open(point_cache, "wb") as f:
            pickle.dump(uid_to_p3, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Done loading global, took {time.time() - point_start:.2f} secs")

    print("Loading observations")
    obs_start = time.time()
    cache_dir = os.path.dirname(obs_path)
    obs_cache = os.path.join(
        cache_dir, f"cache_obs_ds{max_depth_std:.04f}_ids{max_inv_depth_std:.05f}.pkl"
    )
    if os.path.exists(obs_cache) and not force_reload:
        # load from the cached file
        with gzip.open(obs_cache, "rb") as f:
            time_to_uids_slaml = pickle.load(f)
            time_to_uids_slamr = pickle.load(f)
    else:
        with gzip.open(obs_path, "rt") as f:
            header = f.readline().strip().split(",")
            col_uid = header.index("uid")
            col_ts = header.index("frame_tracking_timestamp_us")
            col_serial = header.index("camera_serial")
            raw = f.read()
        time_to_uids_slaml = defaultdict(list)
        time_to_uids_slamr = defaultdict(list)
        valid_uids = set(uid_to_p3.keys())
        slam_left = {"camera-slam-left", "0", "slam-front-left"}
        slam_right = {"camera-slam-right", "1", "slam-front-right"}
        for line in raw.splitlines():
            parts = line.split(",")
            uid = int(parts[col_uid])
            if uid not in valid_uids:
                continue
            time_ns = int(parts[col_ts]) * 1000
            label = serial2label[parts[col_serial]]
            if label in slam_left:
                time_to_uids_slaml[time_ns].append(uid)
            elif label in slam_right:
                time_to_uids_slamr[time_ns].append(uid)
        print(f"==> Writing to {obs_cache}")
        with gzip.open(obs_cache, "wb") as f:
            pickle.dump(time_to_uids_slaml, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(time_to_uids_slamr, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Done loading obs, took {time.time() - obs_start:.2f} secs")

    return time_to_uids_slaml, time_to_uids_slamr, uid_to_p3


## ----- ADT I/O STUFF BELOW -----


def dump_obbs_adt(
    root_path, timed_obbs, sem2name=SSI_SEM2NAME, dont_write=False, do_zip=False
):
    # Write 2d_bounding_boxes.
    text = "stream_id,object_uid,timestamp[ns],x_min[pixel],x_max[pixel],y_min[pixel],y_max[pixel],visibility_ratio[%]\n"

    # Convert all obbs to double for highest precision to keep Rs as valid rotations.
    timed_obbs = {key: val.double() for key, val in timed_obbs.items()}

    def _dump_view(stream_id, inst_ids, time_ns, bb2s):
        lines = ""
        for bb2, i_id in zip(bb2s, inst_ids):
            if i_id == -1:
                continue
            vis_ratio = 0.0  # Hardcode to 0.0 since we don't have that
            lines += "%s, %d, %d, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (
                stream_id,
                i_id,
                time_ns,
                bb2[0],
                bb2[1],
                bb2[2],
                bb2[3],
                vis_ratio,
            )
        return lines

    for time_ns in timed_obbs:
        obbs = timed_obbs[time_ns]
        text += _dump_view("214-1", obbs.inst_id, time_ns, obbs.bb2_rgb)
        text += _dump_view("1201-1", obbs.inst_id, time_ns, obbs.bb2_slaml)
        text += _dump_view("1201-2", obbs.inst_id, time_ns, obbs.bb2_slamr)
    bb2d_path = os.path.join(root_path, "2d_bounding_box.csv")
    if not dont_write:
        with open(bb2d_path, "w") as f:
            f.write(text)
    print("Wrote %d lines to %s" % (len(text.split("\n")) - 2, bb2d_path))

    all_obbs = {}
    for time_ns in timed_obbs:
        obbs = timed_obbs[time_ns]
        for obb in obbs:
            inst_id = int(obb.inst_id)
            if inst_id == -1:
                continue
            if inst_id not in all_obbs:
                all_obbs[inst_id] = obb.clone()
                # Observations are not valid any more.
                all_obbs[inst_id].set_bb2(cam_id=0, bb2d=-1)
                all_obbs[inst_id].set_bb2(cam_id=1, bb2d=-1)
                all_obbs[inst_id].set_bb2(cam_id=2, bb2d=-1)
    all_obbs = torch.stack([val for val in all_obbs.values()])

    # Write T_world_object
    # TODO: support dynamic objects.
    text = "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\n"
    for obb in all_obbs:
        inst_id = obb.inst_id
        if inst_id == -1:
            continue
        T_wo = obb.T_world_object
        tx, ty, tz = T_wo.t
        tx, ty, tz = tx.item(), ty.item(), tz.item()
        R = T_wo.fit_to_SO3().R
        qw, qx, qy, qz = rotmat_to_quat(R.numpy())
        ts = -1
        text += "%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (
            inst_id,
            ts,
            tx,
            ty,
            tz,
            qw,
            qx,
            qy,
            qz,
        )
    scene_objects_path = os.path.join(root_path, "scene_objects.csv")
    if not dont_write:
        with open(scene_objects_path, "w") as f:
            f.write(text)
    print("Wrote %d lines to %s" % (len(text.split("\n")) - 2, scene_objects_path))

    # Write extents.
    text = "object_uid,timestamp[ns],p_local_obj_xmin[m],p_local_obj_xmax[m],p_local_obj_ymin[m],p_local_obj_ymax[m],p_local_obj_zmin[m],p_local_obj_zmax[m]\n"
    for obb in all_obbs:
        inst_id = obb.inst_id
        if inst_id == -1:
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = obb.bb3_object
        ts = -1
        text += "%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (
            inst_id,
            ts,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        )
    bb3d_path = os.path.join(root_path, "3d_bounding_box.csv")
    if not dont_write:
        with open(bb3d_path, "w") as f:
            f.write(text)
    print("Wrote %d lines to %s" % (len(text.split("\n")) - 2, bb3d_path))

    # Write instances.json
    instances = {}
    for obb in all_obbs:
        inst_id = int(obb.inst_id)
        sem_id = int(obb.sem_id)
        if inst_id not in instances:
            if sem2name is None:
                sem_name = "Anything"
            elif sem_id not in sem2name:
                sem_name = "Unknown"
            else:
                sem_name = str(sem2name[sem_id])
            instances[inst_id] = {}
            instances[inst_id]["category"] = sem_name
            if torch.isnan(obb.prob):
                prob = -1
            else:
                prob = float(obb.prob)
            instances[inst_id]["prob"] = prob
    inst_path = os.path.join(root_path, "instances.json")
    if not dont_write:
        with open(inst_path, "w") as f:
            json.dump(instances, f)
    print("Wrote %d lines to %s" % (len(instances), inst_path))

    if do_zip:
        # zip all four files into a .zip file
        zip_path = os.path.join(root_path, "adt.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in [
                bb2d_path,
                scene_objects_path,
                bb3d_path,
                inst_path,
            ]:
                zipf.write(file, os.path.basename(file))
        print(f"Wrote zip to {zip_path}")

    return


def load_obbs_adt(
    root_path,
    only_open=False,
    visibility_thresh=0.0,
    use_description=False,
    return_sem2id=False,
    force_reload=False,
    view_filter="all",
    only_3d=False,
):
    # Validate and normalize view_filter
    valid_views = {"all", "rgb", "slaml", "slamr"}
    if isinstance(view_filter, str):
        view_filter = view_filter.lower()
        if view_filter not in valid_views:
            raise ValueError(
                f"Invalid view_filter '{view_filter}'. Must be one of {valid_views}"
            )
        views_to_load = (
            {"rgb", "slaml", "slamr"} if view_filter == "all" else {view_filter}
        )
    elif isinstance(view_filter, (list, tuple, set)):
        views_to_load = {v.lower() for v in view_filter}
        invalid = views_to_load - {"rgb", "slaml", "slamr"}
        if invalid:
            raise ValueError(
                f"Invalid view(s) in view_filter: {invalid}. Must be 'rgb', 'slaml', or 'slamr'"
            )
    else:
        raise ValueError(
            f"view_filter must be a string or list, got {type(view_filter)}"
        )

    # Create view filter string for cache key
    view_key = "_".join(sorted(views_to_load)) if view_filter != "all" else "all"

    # Format visibility threshold with underscore instead of dot for filename
    vis_str = f"{visibility_thresh:.02f}".replace(".", "_")

    # Check for cached file
    cache_path = os.path.join(
        root_path,
        f"cache_obbs_adt_vis{vis_str}_desc{use_description}_view{view_key}.pt",
    )
    if os.path.exists(cache_path) and not force_reload and not only_3d:
        print(f"==> Loading cached OBBs from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        final_obbs = cached_data["final_obbs"]
        sem_name_to_id = cached_data["sem_name_to_id"]
        print(f"==> Loaded {len(final_obbs)} timestamps from cache")
        result = (final_obbs,)
        if return_sem2id:
            result += (sem_name_to_id,)
        if len(result) == 1:
            return result[0]
        return result

    bb2d_path = os.path.join(root_path, "2d_bounding_box.csv")
    if not only_3d:
        bb2_rgb, bb2_slaml, bb2_slamr = load_2d_bounding_boxes_adt(
            bb2d_path, visibility_thresh
        )

        print(f"got {len(bb2_rgb)} rgb bb2 observations")
        print(f"got {len(bb2_slaml)} slaml bb2 observations")
        print(f"got {len(bb2_slamr)} slamr bb2 observations")
    # N = len(bb2_rgb)
    # assert (
    #    len(bb2_slaml) == N
    # ), "visibility not supported yet, only same size 2D bounding box supported with ADT loader"
    # assert (
    #    len(bb2_slamr) == N
    # ), "visibility not supported yet, only same size 2D bounding box supported with ADT loader"

    bb3d_path = os.path.join(root_path, "scene_objects.csv")
    T_wo_all = load_3d_bounding_box_transforms(bb3d_path)
    T_wo_times = [key for key in T_wo_all]

    # Load instances early so we can show human-readable names in warnings
    instance_path = os.path.join(root_path, "instances.json")
    if use_description:
        instances, instance_descs = load_instances_adt(
            instance_path, return_descriptions=True
        )
    else:
        instances = load_instances_adt(instance_path)
        instance_descs = None

    # Check for dynamic objects and determine which ones actually move
    dynamic_times = [t for t in T_wo_times if t != -1]
    static_T_wo = T_wo_all.get(-1, {})

    # Track which objects are truly dynamic (poses change) vs pseudo-dynamic (poses don't change)
    truly_dynamic_ids = set()
    pseudo_static_ids = {}  # object_id -> pose to use (from first timestamp)

    if dynamic_times:
        # Collect all poses for each dynamic object
        dynamic_object_poses = {}  # object_id -> list of (timestamp, PoseTW)
        for t in dynamic_times:
            for obj_id, pose in T_wo_all[t].items():
                if obj_id not in dynamic_object_poses:
                    dynamic_object_poses[obj_id] = []
                dynamic_object_poses[obj_id].append((t, pose))

        # Check if each object's pose actually changes
        TRANSLATION_THRESH = 0.01  # 1cm threshold for considering movement
        ROTATION_THRESH = 0.01  # ~0.5 degree threshold for rotation difference

        for obj_id, pose_list in dynamic_object_poses.items():
            # Get reference pose (static if available, otherwise first dynamic)
            if obj_id in static_T_wo:
                ref_pose = static_T_wo[obj_id]
            else:
                ref_pose = pose_list[0][1]

            # Check if any pose differs significantly from reference
            moves = False
            ref_t = ref_pose.t.squeeze()
            ref_R = ref_pose.R.squeeze()

            for _, pose in pose_list:
                t_diff = (pose.t.squeeze() - ref_t).norm().item()
                # Rotation difference: Frobenius norm of R1 - R2
                R_diff = (pose.R.squeeze() - ref_R).norm().item()

                if t_diff > TRANSLATION_THRESH or R_diff > ROTATION_THRESH:
                    moves = True
                    break

            if moves:
                truly_dynamic_ids.add(obj_id)
            else:
                # Object doesn't actually move - treat as static
                # Use static pose if available, otherwise use first dynamic pose
                if obj_id in static_T_wo:
                    pseudo_static_ids[obj_id] = static_T_wo[obj_id]
                else:
                    pseudo_static_ids[obj_id] = pose_list[0][1]

    if -1 not in T_wo_all and not pseudo_static_ids:
        raise IOError(
            "No static objects found (timestamp=-1). Only static objects are supported."
        )

    # Merge static objects with pseudo-static ones
    T_wo = dict(static_T_wo)  # Start with truly static objects
    T_wo.update(pseudo_static_ids)  # Add pseudo-static objects

    # Report object classification
    static_names = [
        instances.get(obj_id, f"unknown_id_{obj_id}")
        for obj_id in sorted(static_T_wo.keys())
    ]
    pseudo_static_names = [
        instances.get(obj_id, f"unknown_id_{obj_id}")
        for obj_id in sorted(pseudo_static_ids.keys())
    ]
    truly_dynamic_names = [
        instances.get(obj_id, f"unknown_id_{obj_id}")
        for obj_id in sorted(truly_dynamic_ids)
    ]

    extent_path = os.path.join(root_path, "3d_bounding_box.csv")
    bb3 = load_3d_bounding_box_local_extents(extent_path)

    M = len(T_wo)

    name2sem = dict(SSI_NAME2SEM)  # mutable copy so we can extend it
    name2color = SSI_COLORS
    name2desc = {key: key for key in SSI_NAME2SEM}

    # Create case-insensitive lookup: lowercase -> original case name
    name2sem_lower = {key.lower(): key for key in name2sem}

    # Pre-compute ANYTHING count (instances with name "Anything")
    anything_count = 0
    for inst in instances:
        if inst not in bb3 or inst not in T_wo:
            continue
        inst_name = instances[inst]
        if inst_name.lower() == "anything":
            anything_count += 1

    print("\n" + "=" * 80)
    print("OBJECT CLASSIFICATION REPORT:")
    print(
        f"  STATIC ({len(static_T_wo)} objects): {static_names[:10]}{'...' if len(static_names) > 10 else ''}"
    )
    print(
        f"  PSEUDO-STATIC ({len(pseudo_static_ids)} objects, marked dynamic but don't move): {pseudo_static_names[:10]}{'...' if len(pseudo_static_names) > 10 else ''}"
    )
    print(
        f"  TRULY DYNAMIC ({len(truly_dynamic_ids)} objects, will be SKIPPED): {truly_dynamic_names[:10]}{'...' if len(truly_dynamic_names) > 10 else ''}"
    )
    print(f"  ANYTHING: {anything_count} objects")
    total_objects = len(static_T_wo) + len(pseudo_static_ids) + len(truly_dynamic_ids)
    print(f"  TOTAL USABLE: {len(T_wo)} objects out of {total_objects} total objects")
    print("=" * 80 + "\n")

    # bb3 may contain dynamic objects too, so don't assert exact match
    # assert len(bb3) == M
    # assert len(instances) == M

    class_counts = {}
    sem_name_to_id = {}
    all_obbs = {}
    not_found_counts = {}  # Track unknown instance types

    # Create mapping from original large instance IDs to smaller IDs
    # Only remap IDs > 10000 to avoid int32 overflow
    orig_to_new_id = {}
    # Start remapped IDs from 10001 to avoid collision with small IDs
    remap_counter = 10001

    next_sem_id = max(name2sem.values()) + 1  # starts at 33

    # for inst in T_wo:
    for inst in instances:
        if inst not in bb3:
            print(f"WARNING: instance {inst} not found in bb3, skipping")
            continue
        if inst not in T_wo:
            # Dynamic object, already warned above
            continue

        # Only remap large IDs to avoid int32 overflow
        if inst > 10000:
            orig_to_new_id[inst] = remap_counter
            remap_counter += 1
        else:
            orig_to_new_id[inst] = inst

        b3 = torch.tensor(bb3[inst])
        t_wo = T_wo[inst]
        inst_name = instances[inst]
        inst_id = torch.tensor(orig_to_new_id[inst]).reshape(1)
        sem_id = torch.tensor([-1])
        color = torch.tensor([1.0, 1.0, 1.0])

        # Try exact match first, then case-insensitive match
        matched_name = None
        if inst_name in name2sem:
            matched_name = inst_name
        elif inst_name.lower() in name2sem_lower:
            matched_name = name2sem_lower[inst_name.lower()]

        if matched_name is None:
            not_found_counts[inst_name] = not_found_counts.get(inst_name, 0) + 1
            # Auto-assign a unique sem_id for this category
            if inst_name not in name2sem:
                name2sem[inst_name] = next_sem_id
                next_sem_id += 1
            sem_id = torch.tensor(int(name2sem[inst_name])).reshape(1)
            color = torch.tensor(name2color.get(inst_name, name2color["Anything"]))
            if use_description and instance_descs is not None:
                text = instance_descs[inst]
            else:
                text = inst_name  # Use original name as text
        else:
            sem_id = torch.tensor(int(name2sem[matched_name])).reshape(1)
            color = torch.tensor(name2color.get(matched_name, name2color["Anything"]))
            # Use descriptive instance name when use_description=True, otherwise use category name
            if use_description and instance_descs is not None:
                text = instance_descs[inst]
            else:
                text = name2desc.get(matched_name, matched_name)
        if text not in sem_name_to_id:
            sem_name_to_id[text] = sem_id.item()
        text = string2tensor(pad_string(text, max_len=128, silent=True))

        obb = ObbTW.from_lmc(
            b3, None, None, None, t_wo, sem_id, inst_id, None, None, color, text
        )
        all_obbs[int(inst_id)] = obb

    # Print summary of unknown instance types (mapped to "Anything")
    if not_found_counts:
        total_mapped = sum(not_found_counts.values())
        print(
            f"INFO: Mapped {total_mapped} objects with {len(not_found_counts)} unique categories with auto-assigned sem_ids: {dict(not_found_counts)}"
        )

    # If only_3d, skip 2D loading and return all 3D OBBs under timestamp -1
    if only_3d:
        obb_list = list(all_obbs.values())
        if not obb_list:
            raise IOError("No valid 3D objects found.")
        stacked = torch.stack(obb_list)
        final_obbs = {-1: stacked}
        print(f"==> only_3d mode: {len(obb_list)} objects at timestamp -1")
        if return_sem2id:
            return final_obbs, sem_name_to_id
        return final_obbs

    # print("==> Loaded the following of each class (showing every 25th frame):")
    # for ii, cn in enumerate(sorted(class_counts)):
    #     if ii % 50 == 0:
    #         print(f"2d obs from frameset {ii} got {class_counts[cn]} {cn}(s)")

    timed_obbs = {}

    # Create reverse mapping for bb2 lookup
    new_to_orig_id = {v: k for k, v in orig_to_new_id.items()}

    def _update_timed_obbs(
        timed_obbs, all_obbs, bb2s, bb2_idx, orig_to_new_id, new_to_orig_id
    ):
        obb_times = [tt for tt in timed_obbs]
        # Iterate all bb2 observations, and add new ObbTW or update any existing.
        for ts in bb2s:
            visible_bb3_ids = [val[0] for val in bb2s[ts]]
            visible_bb3 = []
            for orig_inst_id in visible_bb3_ids:
                # Convert original ID to new ID for all_obbs lookup
                if orig_inst_id in orig_to_new_id:
                    new_inst_id = orig_to_new_id[orig_inst_id]
                    if new_inst_id in all_obbs:
                        visible_bb3.append(all_obbs[new_inst_id])
            # id2bb2 still uses original IDs from bb2 data
            id2bb2 = {int(val[0]): val[1:] for val in bb2s[ts]}

            # Allow 1ms tolerance for matching across SLAM and RGB views.
            found = False
            if len(obb_times) > 0:
                found_ts = find_nearest(obb_times, ts)
                diff = abs(ts - found_ts)
                if diff < 1e6:  # 1ms
                    found = True
                    ts = found_ts

            # if ts not in timed_obbs:
            if not found:
                timed_obbs[ts] = {}
            for obb in visible_bb3:
                new_inst_id = int(obb.inst_id)
                # Convert back to original ID for id2bb2 lookup
                orig_inst_id = new_to_orig_id[new_inst_id]
                if new_inst_id in timed_obbs[ts]:
                    existing_obb = timed_obbs[ts][new_inst_id]
                    cur_obb = existing_obb
                else:
                    cur_obb = obb.clone()
                cur_obb.set_bb2(bb2_idx, torch.tensor(id2bb2[orig_inst_id]))
                timed_obbs[ts][new_inst_id] = cur_obb.clone()
        return timed_obbs

    # Only load 2D bounding boxes for the requested views
    if "rgb" in views_to_load:
        timed_obbs = _update_timed_obbs(
            timed_obbs, all_obbs, bb2_rgb, 0, orig_to_new_id, new_to_orig_id
        )
    if "slaml" in views_to_load:
        timed_obbs = _update_timed_obbs(
            timed_obbs, all_obbs, bb2_slaml, 1, orig_to_new_id, new_to_orig_id
        )
    if "slamr" in views_to_load:
        timed_obbs = _update_timed_obbs(
            timed_obbs, all_obbs, bb2_slamr, 2, orig_to_new_id, new_to_orig_id
        )

    final_obbs = {}
    for ts in timed_obbs:
        sort_idx = sorted(timed_obbs[ts])
        obb_list = [timed_obbs[ts][inst_id] for inst_id in sort_idx]
        if len(obb_list) == 0:
            # No valid objects at this timestamp, skip
            continue
        stacked_obbs = torch.stack(obb_list)
        final_obbs[ts] = stacked_obbs.clone()

    if not final_obbs:
        raise IOError(
            "No valid objects found after filtering. Check that instance types match name2sem."
        )

    counts_per_ts = []
    count_rgb = 0
    count_slaml = 0
    count_slamr = 0
    # Show 10 linearly spaced samples for the counts.
    sample_inds = np.linspace(0, len(final_obbs), 10).astype(int)
    for ii, ts in enumerate(final_obbs):
        num_rgb = (final_obbs[ts].bb2_rgb[:, 0] != -1).sum()
        num_slaml = (final_obbs[ts].bb2_slaml[:, 0] != -1).sum()
        num_slamr = (final_obbs[ts].bb2_slamr[:, 0] != -1).sum()
        count_rgb += num_rgb
        count_slaml += num_slaml
        count_slamr += num_slamr
        counts_per_ts.append(num_rgb + num_slaml + num_slamr)
        if ii in sample_inds:
            print(f"==> {ts} rgb {num_rgb} slaml {num_slaml} slamr {num_slamr}")
    print("==> Final 2DBB counts for cameras:")
    print(f"rgb {count_rgb}")
    print(f"slaml {count_slaml}")
    print(f"slamr {count_slamr}")

    # Save to cache
    print(f"==> Saving OBBs cache to {cache_path}")
    torch.save(
        {"final_obbs": final_obbs, "sem_name_to_id": sem_name_to_id},
        cache_path,
    )

    result = (final_obbs,)
    if return_sem2id:
        result += (sem_name_to_id,)
    if len(result) == 1:
        return result[0]
    return result


def load_2d_bounding_boxes_adt(bb2d_path, visibility_thresh=0.0):
    bb2ds_rgb = {}
    bb2ds_slaml = {}
    bb2ds_slamr = {}

    with open(bb2d_path) as f:
        lines = f.readlines()

    count = 0
    for ii, line in enumerate(lines):
        if ii == 0:
            continue  # skip header
        line = line.decode("utf-8").rstrip().split(",")
        if len(line) == 6:
            device_id = "unknown"
            offset = -1
        else:
            # expected header:
            # stream_id,object_uid,timestamp[ns],x_min[pixel],x_max[pixel],y_min[pixel],y_max[pixel],visibility_ratio[%]\n'
            device_id = str(line[0])
            offset = 0
        object_id = int(line[1 + offset])
        timestamp = int(line[2 + offset])  # ns
        x_min = max(0, float(line[3 + offset]))
        x_max = max(0, float(line[4 + offset]))
        y_min = max(0, float(line[5 + offset]))
        y_max = max(0, float(line[6 + offset]))
        visibility = float(line[7 + offset])  # visibility ratio
        if visibility < visibility_thresh:
            continue

        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [x_min, x_max, y_min, y_max]):
            continue

        if device_id in ["214-1", "340-5", "unknown"]:
            if timestamp not in bb2ds_rgb:
                bb2ds_rgb[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_rgb[timestamp].append((object_id, x_min, x_max, y_min, y_max))
        if device_id in ["1201-1", "340-1", "unknown"]:
            if timestamp not in bb2ds_slaml:
                bb2ds_slaml[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_slaml[timestamp].append((object_id, x_min, x_max, y_min, y_max))
        if device_id in ["1201-2", "340-2", "unknown"]:
            if timestamp not in bb2ds_slamr:
                bb2ds_slamr[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_slamr[timestamp].append((object_id, x_min, x_max, y_min, y_max))
        if device_id not in [
            "214-1",
            "1201-1",
            "1201-2",
            "1201-3",  # nebula side left.
            "1201-4",  # nebula side right.
            "unknown",
            "340-1",
            "340-2",
            "340-5",
        ]:
            raise IOError(f"unexpected device id {device_id} in 2d observations")

        count += 1
    print(
        f"loaded {count} 2d bbs for {len(bb2ds_rgb)}[rgb] {len(bb2ds_slaml)}[slaml] {len(bb2ds_slamr)}[slamr] timestamps from {bb2d_path}"
    )
    return bb2ds_rgb, bb2ds_slaml, bb2ds_slamr


def load_3d_bounding_box_transforms(scene_path, time_in_secs=False, load_torch=False):
    T_world_object = {}

    with open(scene_path) as f:
        lines = np.genfromtxt(
            f,
            dtype=[int] * 2 + [float] * 7,
            names=True,
            delimiter=",",
            usecols=range(9),
        )
        if lines.size == 1:
            lines = lines[np.newaxis]

    for line in lines:
        object_id = line[0]
        timestamp_ns = line[1]
        if time_in_secs and timestamp_ns != -1:
            timestamp = timestamp_ns / 1e9
        else:
            timestamp = timestamp_ns
        tx = line[2]
        ty = line[3]
        tz = line[4]
        qw = line[5]
        qx = line[6]
        qy = line[7]
        qz = line[8]
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [tx, ty, tz, qw, qx, qy, qz]):
            continue
        rot_mat = quat_to_rotmat(qw, qx, qy, qz)
        trans = torch.tensor([tx, ty, tz])
        T_wo = PoseTW.from_Rt(rot_mat, trans)
        if timestamp not in T_world_object:
            T_world_object[timestamp] = {}
        T_world_object[timestamp][object_id] = T_wo
    return T_world_object


def load_3d_bounding_box_local_extents(bb3d_path, load_torch=False):
    bb3ds_local = {}
    with open(bb3d_path) as f:
        # Object UID, Timestamp ( ns ), p_local_obj.xmin, p_local_obj.xmax, p_local_obj.ymin, p_local_obj.ymax, p_local_obj.zmin, p_local_obj.zmax
        lines = np.genfromtxt(
            f,
            dtype=[int] * 2 + [float] * 6,
            names=True,
            delimiter=",",
            usecols=range(8),
        )
        if lines.size == 1:
            lines = lines[np.newaxis]
    for line in lines:
        object_id = line[0]
        xmin = line[2]
        xmax = line[3]
        ymin = line[4]
        ymax = line[5]
        zmin = line[6]
        zmax = line[7]
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [xmin, xmax, ymin, ymax, zmin, zmax]):
            continue
        local = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        if load_torch:
            local = torch.from_numpy(local)
        bb3ds_local[object_id] = local
    return bb3ds_local


def load_instances_adt(instances_path, return_descriptions=False):
    instance2proto = {}
    instance2desc = {}
    with open(instances_path) as f:
        content = json.load(f)
    for inst_id in content:
        instance2proto[int(inst_id)] = content[inst_id]["category"]
        # Try to get a descriptive name - check various fields
        inst_data = content[inst_id]
        # Priority: instance_name > prototype_name > category
        desc = inst_data.get(
            "instance_name", inst_data.get("prototype_name", inst_data["category"])
        )
        instance2desc[int(inst_id)] = desc

    # lot of other info available, for example:
    #  {'instance_id': 5691266090916432, 'instance_name': 'Hook_4',
    #  'prototype_name': 'Hook', 'category': 'hook', 'category_uid': 643,
    #  'motion_type': 'static', 'instance_type': 'object', 'rigidity': 'rigid',
    #  'rotational_symmetry': {'is_annotated': False},
    #  'canonical_pose': {'up_vector': [0, 1, 0], 'front_vector': [0, 0, 1]}}

    result = (instance2proto,)
    if return_descriptions:
        result += (instance2desc,)
    if len(result) == 1:
        return result[0]
    return result


def read_obb_csv(csv_path, force_anything=False):
    with open(csv_path, "r") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return {}

    # Parse all data into lists first
    ts_list = []
    tx_list, ty_list, tz_list = [], [], []
    qw_list, qx_list, qy_list, qz_list = [], [], [], []
    sx_list, sy_list, sz_list = [], [], []
    inst_list, sem_list, prob_list = [], [], []
    names = []

    for line in lines[1:]:
        elts = line.split(",")
        ts_list.append(int(elts[0]))
        tx_list.append(float(elts[1]))
        ty_list.append(float(elts[2]))
        tz_list.append(float(elts[3]))
        qw_list.append(float(elts[4]))
        qx_list.append(float(elts[5]))
        qy_list.append(float(elts[6]))
        qz_list.append(float(elts[7]))
        sx_list.append(float(elts[8]))
        sy_list.append(float(elts[9]))
        sz_list.append(float(elts[10]))
        names.append(elts[11])
        inst_list.append(int(elts[12]))
        sem_list.append(int(elts[13]))
        prob_list.append(float(elts[14]))

    # Convert to torch tensors
    ts_t = torch.tensor(ts_list, dtype=torch.long)
    sx_t = torch.tensor(sx_list, dtype=torch.float32)
    sy_t = torch.tensor(sy_list, dtype=torch.float32)
    sz_t = torch.tensor(sz_list, dtype=torch.float32)
    inst_t = torch.tensor(inst_list, dtype=torch.long)
    sem_t = torch.tensor(sem_list, dtype=torch.long)
    prob_t = torch.tensor(prob_list, dtype=torch.float32)

    # Build quaternion (wxyz) and translation tensors
    quat_wxyz = torch.tensor(
        list(zip(qw_list, qx_list, qy_list, qz_list)), dtype=torch.float32
    )
    trans = torch.tensor(list(zip(tx_list, ty_list, tz_list)), dtype=torch.float32)

    # Vectorized bounding box computation
    bb3s_t = torch.stack(
        [-sx_t / 2.0, sx_t / 2.0, -sy_t / 2.0, sy_t / 2.0, -sz_t / 2.0, sz_t / 2.0],
        dim=1,
    )

    if force_anything:
        sem_t = torch.full_like(sem_t, 32)

    # Precompute all text tensors
    text_tensors = []
    for name in names:
        if len(name) > 128:
            print(f"WARNING: name {name} is too long, truncating")
        text_tensors.append(string2tensor(pad_string(name, max_len=128, silent=True)))
    text_t = torch.stack(text_tensors)

    # Build all poses at once using PoseTW.from_qt
    T_wo_poses = PoseTW.from_qt(quat_wxyz, trans)

    # Build all ObbTW objects at once using batched constructor
    all_obbs = ObbTW.from_lmc(
        bb3_object=bb3s_t,
        T_world_object=T_wo_poses._data,
        sem_id=sem_t.reshape(-1, 1),
        inst_id=inst_t.reshape(-1, 1),
        prob=prob_t.reshape(-1, 1),
        text=text_t,
    ).float()

    # Group by timestamp
    timed_obbs = {}
    unique_ts = torch.unique(ts_t)
    for ts in unique_ts:
        mask = ts_t == ts
        timed_obbs[int(ts)] = all_obbs[mask]

    return timed_obbs


def save_bb2d_csv(
    csv_path: str,
    frame_id: int,
    bb2d: Union[torch.Tensor, np.ndarray],
    scores: Union[torch.Tensor, np.ndarray, list],
    labels: List[str],
    sem_name_to_id: Optional[Dict] = None,
    append: bool = True,
    time_ns: int = -1,
    img_width: int = -1,
    img_height: int = -1,
    sensor: str = "unknown",
    device: str = "unknown",
) -> None:
    """
    Save 2D bounding box detections to CSV.

    Args:
        csv_path: Path to the output CSV file
        frame_id: Frame index
        bb2d: Bounding boxes (N, 4) in x1, y1, x2, y2 format
        scores: Confidence scores (N,)
        labels: Semantic labels for each detection
        sem_name_to_id: Optional mapping from semantic name to ID
        append: Whether to append to existing file or overwrite
        time_ns: Timestamp in nanoseconds
        img_width: Image width in pixels
        img_height: Image height in pixels
        sensor: Sensor name (rgb, slaml, slamr)
        device: Device name (Aria Gen 1, Aria Gen 2, iPad, etc.)
    """
    if sem_name_to_id is None:
        sem_name_to_id = {}

    # Convert tensors to numpy if needed
    if isinstance(bb2d, torch.Tensor):
        bb2d = bb2d.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(scores, list):
        scores = np.array(scores)

    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(csv_path)

    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "time_ns",
                    "frame_id",
                    "sensor",
                    "device",
                    "img_width",
                    "img_height",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "name",
                    "instance",
                    "sem_id",
                    "prob",
                ]
            )

        for i in range(len(bb2d)):
            x1, y1, x2, y2 = bb2d[i]
            label = labels[i] if i < len(labels) else "unknown"
            sem_id = sem_name_to_id.get(label, -1)
            prob = float(scores[i]) if i < len(scores) else 0.0

            writer.writerow(
                [
                    time_ns,
                    frame_id,
                    sensor,
                    device,
                    img_width,
                    img_height,
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    label,
                    -1,  # instance ID (not tracked)
                    sem_id,
                    f"{prob:.6f}",
                ]
            )


def load_bb2d_csv(csv_path: str) -> dict[int, dict]:
    """Load 2D bounding box CSV and group detections by time_ns.

    Returns:
        Dict mapping time_ns -> {"bb2d": (N,4) ndarray in x1,y1,x2,y2,
                                  "scores": (N,) ndarray, "labels": list[str],
                                  "sem_ids": list[int],
                                  "img_width": int, "img_height": int}
    """
    groups: dict[int, dict] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            time_ns = int(row[0])
            img_width = int(row[4])
            img_height = int(row[5])
            x1, y1, x2, y2 = float(row[6]), float(row[7]), float(row[8]), float(row[9])
            name = row[10]
            sem_id = int(row[12]) if len(row) > 12 else -1
            prob = float(row[13])
            if time_ns not in groups:
                groups[time_ns] = {
                    "bb2d": [],
                    "scores": [],
                    "labels": [],
                    "sem_ids": [],
                    "img_width": img_width,
                    "img_height": img_height,
                }
            groups[time_ns]["bb2d"].append([x1, y1, x2, y2])
            groups[time_ns]["scores"].append(prob)
            groups[time_ns]["labels"].append(name)
            groups[time_ns]["sem_ids"].append(sem_id)

    # Convert lists to numpy arrays
    for time_ns in groups:
        groups[time_ns]["bb2d"] = np.array(groups[time_ns]["bb2d"], dtype=np.float32)
        groups[time_ns]["scores"] = np.array(
            groups[time_ns]["scores"], dtype=np.float32
        )

    return groups


class ObbCsvWriter2:
    def __init__(self, file_name="", append=False, verbose=False):
        if not file_name:
            file_name = "/tmp/obbs.csv"

        self.file_name = file_name
        self.verbose = verbose
        headers_str = "time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,name,instance,sem_id,prob"
        headers = headers_str.split(",")
        self.num_cols = len(headers)

        if append and os.path.exists(file_name):
            if self.verbose:
                print(f"appending to existing obb file: {file_name}")
            self.file_writer = open(self.file_name, "a")
        else:
            if self.verbose:
                print(f"starting obb writer to {file_name}")
            self.file_writer = open(self.file_name, "w")
            header_row = ",".join(headers)
            self.file_writer.write(header_row + "\n")

    def write(
        self,
        obb_padded: ObbTW,
        timestamps_ns: int = -1,
        sem_id_to_name: Optional[Dict[int, str]] = None,
    ):
        # pyre-fixme[16]: `list` has no attribute `clone`.
        obb = obb_padded.remove_padding().clone().cpu()
        time_ns = str(int(timestamps_ns))

        N = obb.shape[0]
        if N == 0:
            if self.verbose:
                print(f"WARNING: no obbs at time {time_ns}")
            return

        # Batch-extract all fields as numpy to avoid per-element tensor ops
        obbs_poses = obb.T_world_object
        all_q = (
            obbs_poses.q.numpy()
        )  # (N, 4) - vectorized rotation_matrix_to_quaternion
        all_t = obbs_poses.t.numpy()  # (N, 3)
        obbs_dims = obb.bb3_diagonal.numpy()  # (N, 3)
        obb_sems = obb.sem_id.squeeze(-1).numpy()
        obb_inst = obb.inst_id.squeeze(-1).numpy()
        obb_prob = obb.prob.squeeze(-1).numpy()

        # Batch-extract text labels
        text_bytes = obb.text.byte().numpy()  # (N, 128)
        names = []
        for i in range(N):
            # Decode bytes to string, strip null padding and whitespace
            raw = (
                bytes(text_bytes[i])
                .rstrip(b"\x00")
                .decode("ascii", errors="replace")
                .strip()
            )
            if raw == "":
                if sem_id_to_name and obb_sems[i] in sem_id_to_name:
                    raw = sem_id_to_name[obb_sems[i]]
                else:
                    raw = str(int(obb_sems[i]))
            names.append(raw.replace(",", " ").lower())

        # Build all rows at once
        lines = []
        for i in range(N):
            txyz = f"{all_t[i, 0]},{all_t[i, 1]},{all_t[i, 2]}"
            qwxyz = f"{all_q[i, 0]},{all_q[i, 1]},{all_q[i, 2]},{all_q[i, 3]}"
            sxyz = f"{obbs_dims[i, 0]},{obbs_dims[i, 1]},{obbs_dims[i, 2]}"
            lines.append(
                f"{time_ns},{txyz},{qwxyz},{sxyz},{names[i]},{obb_inst[i]},{obb_sems[i]},{obb_prob[i]}\n"
            )
        self.file_writer.write("".join(lines))
        self.file_writer.flush()
        if self.verbose:
            print(f"===> ObbCsvWriter: wrote {N} rows to {self.file_name}")

    def close(self):
        self.file_writer.close()

    def __del__(self):
        if hasattr(self, "file_writer"):
            self.file_writer.close()
