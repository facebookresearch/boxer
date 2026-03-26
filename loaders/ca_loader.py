# pyre-ignore-all-errors
import io
import json
import os
import tarfile

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from tw.camera import CameraTW, get_base_aria_rgb_camera
from tw.obb import ObbTW
from tw.pose import PoseTW
from tw.tensor_utils import pad_string, string2tensor


def _fishify(image, cam_pin, cam_fish):
    """Convert pinhole image to fisheye image."""
    image_torch = torch.from_numpy(image.astype(np.float32))
    if torch.cuda.is_available():
        image_torch = image_torch.cuda()
        cam_fish = cam_fish.cuda()
        cam_pin = cam_pin.cuda()
    device = image_torch.device
    H, W = image.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    target = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).reshape(-1, 1, 2)
    rays, _ = cam_fish.unproject(target)
    source, _ = cam_pin.project(rays)
    source = source.reshape(H, W, 2)
    source[:, :, 0] = 2 * (source[:, :, 0] / W) - 1
    source[:, :, 1] = 2 * (source[:, :, 1] / H) - 1
    if image_torch.ndim == 3:
        image_torch = image_torch.permute(2, 0, 1)
    else:
        image_torch = image_torch.unsqueeze(0)
    image_torch = image_torch.float()
    image_torch2 = torch.nn.functional.grid_sample(
        image_torch[None], source[None], "bicubic", align_corners=False
    )
    image_fish = image_torch2[0]
    if image_fish.shape[0] == 1:
        image_fish = image_fish.squeeze(0)
    else:
        image_fish = image_fish.permute(1, 2, 0)
    image_fish = image_fish.cpu().numpy()
    return image_fish


def _load_tar(
    tar_path,
    start_frame=0,
    skip_frames=1,
    max_frames=1000,
    fisheye=False,
    verbose=False,
    use_canny=True,
    num_samples=10000,
    frame_gt=False,
    load_images=True,
):
    """
    Load a single CA1M tar file into memory

    Args:
        tar_path: Path to the input tar file
        start_frame: Number of frames to skip at the beginning
        skip_frames: Number of frames to skip between each snippet
        max_frames: Maximum number of frames to load
        use_canny: If True, sample points using Canny edges; if False, sample randomly
        num_samples: Number of points to sample per frame
        frame_gt: If True, use per-frame 3D boxes from [timestamp].wide/instances.json
                  instead of world boxes from world.gt/instances.json. Per-frame boxes
                  are rendered independently and not cropped/transformed.
        load_images: If True (default), load RGB images, depth images, and sample
                     point clouds. If False, only load metadata (cameras, poses, OBBs,
                     timestamps, bb2ds) to reduce memory usage.
    """
    count = 0
    T_wcs = []
    cams = []
    depth_cams = []
    obbs = []
    rgb_images = []
    timestamps_ns = []
    depth_images = []
    sdp_ws = []
    bb2ds = []  # 2D bounding boxes in xxyy format
    bb2d_inst_ids = []  # Instance IDs for each 2D bounding box
    id2inst = {}
    seq_name = os.path.basename(tar_path).split(".")[0]

    with tarfile.open(tar_path, "r") as tar:
        # Read world into memory.
        world_name = None
        for member in tar.getmembers():
            if member.isfile() and "world.gt/instances.json" in member.name:
                world_name = member.name
                break
        if world_name is None:
            raise ValueError("No world found in tar file.")

        member = tar.getmember(world_name)
        f = tar.extractfile(member)
        bb3 = json.loads(f.read().decode("utf-8"))
        N = len(bb3)
        if verbose:
            print("==> Found %d 3DBBs in world" % N)
        bb3_R = torch.stack([torch.tensor(bb["R"]) for bb in bb3])
        bb3_t = torch.stack([torch.tensor(bb["position"]) for bb in bb3])
        T_wo = PoseTW.from_Rt(bb3_R, bb3_t)
        bb3_sc = torch.stack([torch.tensor(bb["scale"]) for bb in bb3])
        xmin = -bb3_sc[:, 0] / 2
        xmax = bb3_sc[:, 0] / 2
        ymin = -bb3_sc[:, 1] / 2
        ymax = bb3_sc[:, 1] / 2
        zmin = -bb3_sc[:, 2] / 2
        zmax = bb3_sc[:, 2] / 2
        sz = torch.stack([xmin, xmax, ymin, ymax, zmin, zmax], dim=-1)

        ids = [bb["id"] for bb in bb3]
        for i, id_ in enumerate(ids):
            id2inst[id_] = i
        inst_ids = torch.tensor([id2inst[id_] for id_ in ids], dtype=torch.int64)
        # Build mapping from unique category name to unique semantic id
        category_names = [bb["category"] for bb in bb3]
        unique_categories = sorted(set(category_names))
        sem_id_to_name = {i: cat for i, cat in enumerate(unique_categories)}
        # Map instance id to semantic id (category id)
        category_to_sem_id = {cat: i for i, cat in enumerate(unique_categories)}
        sem_ids = [category_to_sem_id[bb["category"]] for bb in bb3]
        sem_ids = torch.tensor(sem_ids, dtype=torch.int64)
        # Create a mapping from inst_id to "caption" field
        inst_id_to_caption = {}
        for bb in bb3:
            inst_id_to_caption[id2inst[bb["id"]]] = bb.get("caption", "")
        # Store the semantic name in the obb.text field.
        text = [string2tensor(pad_string(xx, max_len=128)) for xx in category_names]
        text = torch.stack(text)
        all_obbs = ObbTW.from_lmc(
            bb3_object=sz,
            T_world_object=T_wo,
            sem_id=sem_ids,
            inst_id=inst_ids,
            text=text,
        )

        # Get image tags.
        image_tags = set()
        for member in tar.getmembers():
            if "world" in member.name:
                continue
            image_tag = member.name.split(".")[0]
            if image_tag not in image_tags:
                image_tags.add(image_tag)
        if verbose:
            print("==> Found sequence of length %d" % len(image_tags))

        image_tags = sorted(image_tags)
        image_tags = image_tags[start_frame:]  # Skip first start_frame frames
        image_tags = image_tags[::skip_frames]

        H, W = None, None
        Hd, Wd = None, None

        for image_tag in tqdm(image_tags):
            # Get timestamp (from image tag).
            timestamp_ns = int(image_tag.split("/")[-1])

            # Load rgb image (skip after first frame when load_images=False).
            if load_images or H is None:
                name = image_tag + ".wide/image.png"
                f = tar.extractfile(tar.getmember(name))
                image = np.array(Image.open(io.BytesIO(f.read())))
                H, W = image.shape[:2]

            # Load camera extrinsics.
            name = image_tag + ".gt/RT.json"
            f = tar.extractfile(tar.getmember(name))
            RT = torch.tensor(json.loads(f.read().decode("utf-8")))
            camR = RT[:3, :3]
            camt = RT[:3, 3]
            T_wc = PoseTW.from_Rt(camR, camt)

            # Load camera intrinsics.
            name = image_tag + ".wide/image/K.json"
            f = tar.extractfile(tar.getmember(name))
            K = torch.tensor(json.loads(f.read().decode("utf-8")))
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            params = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
            cam = CameraTW.from_surreal(
                width=W, height=H, type_str="pinhole", params=params
            )

            # Load depth image (skip after first frame when load_images=False).
            if load_images or Hd is None:
                name = image_tag + ".wide/depth.png"
                f = tar.extractfile(tar.getmember(name))
                depth = np.array(Image.open(io.BytesIO(f.read())))
                Hd, Wd = depth.shape[:2]
            name = image_tag + ".wide/depth/K.json"
            f = tar.extractfile(tar.getmember(name))
            Kd = torch.tensor(json.loads(f.read().decode("utf-8")))
            fxd = Kd[0, 0]
            fyd = Kd[1, 1]
            cxd = Kd[0, 2]
            cyd = Kd[1, 2]
            paramsd = torch.tensor([fxd, fyd, cxd, cyd], dtype=torch.float32)
            depth_cam = CameraTW.from_surreal(
                width=Wd, height=Hd, type_str="pinhole", params=paramsd
            )

            if load_images:
                depth_resize = cv2.resize(
                    depth, (W, H), interpolation=cv2.INTER_NEAREST
                )
                if use_canny:
                    try:
                        weights = cv2.Canny(image, 30, 60)
                        weights_flat = weights.ravel().astype(np.float64)
                        weights_flat /= weights_flat.sum()  # normalize to sum=1
                        idx = np.random.choice(
                            H * W, size=num_samples, replace=True, p=weights_flat
                        )
                        ys, xs = np.unravel_index(idx, (H, W))
                    except ValueError:
                        # Fallback to random sampling if Canny fails
                        xs = np.random.randint(0, W, size=(num_samples))
                        ys = np.random.randint(0, H, size=(num_samples))
                else:
                    # Sample points randomly from image
                    xs = np.random.randint(0, W, size=(num_samples))
                    ys = np.random.randint(0, H, size=(num_samples))
                points = np.stack([xs, ys], axis=-1)
                points3, valid = cam.unproject(points[None])
                points3 = points3[0]
                valid = valid[0]
                points3 = points3[valid]
                hh = points[valid, 1].astype(int)
                ww = points[valid, 0].astype(int)
                zz = (
                    torch.tensor(depth_resize[hh, ww].astype(np.float32)).float()
                    / 1000.0
                )
                sdp_c = points3.reshape(-1, 3)
                sdp_c = sdp_c * zz.reshape(-1, 1)
                sdp_w = T_wc * sdp_c
                if sdp_w.shape[0] < num_samples:
                    num_pad = num_samples - sdp_w.shape[0]
                    pad_vals = float("nan") * np.ones((num_pad, 3))
                    sdp_w = np.concatenate([sdp_w, pad_vals], axis=0)
                sdp_w = torch.as_tensor(sdp_w).detach().clone()

            # Load 3DBBs.
            name = image_tag + ".wide/instances.json"
            f = tar.extractfile(tar.getmember(name))
            bb3 = json.loads(f.read().decode("utf-8"))
            visible_obbs = []
            frame_bb2ds = []  # 2D bounding boxes for this frame
            frame_bb2d_inst_ids = []  # Instance IDs for each 2D bbox
            for i_bb, bb in enumerate(bb3):
                id_ = bb["id"]
                if id_ not in id2inst:
                    continue
                inst_id = id2inst[id_]
                if frame_gt:
                    # Use per-frame 3D box data directly (rendered per-frame, pose-independent)
                    bb_R = torch.tensor(bb["R"])
                    bb_t = torch.tensor(bb["position"])
                    T_co = PoseTW.from_Rt(bb_R[None], bb_t[None])
                    T_wo_frame = T_wc @ T_co
                    bb_sc = torch.tensor(bb["scale"])
                    xmin = -bb_sc[0] / 2
                    xmax = bb_sc[0] / 2
                    ymin = -bb_sc[1] / 2
                    ymax = bb_sc[1] / 2
                    zmin = -bb_sc[2] / 2
                    zmax = bb_sc[2] / 2
                    sz_frame = torch.stack([xmin, xmax, ymin, ymax, zmin, zmax])[None]
                    world_obb = all_obbs[inst_id]
                    frame_obb = ObbTW.from_lmc(
                        bb3_object=sz_frame,
                        T_world_object=T_wo_frame,
                        sem_id=world_obb.sem_id[None],
                        inst_id=world_obb.inst_id[None],
                        text=world_obb.text[None],
                    )
                    visible_obbs.append(frame_obb[0])
                else:
                    visible_obbs.append(all_obbs[inst_id].clone())
                # Load 2D bounding box if available
                if "box_2d_rend" in bb:
                    bbox = bb["box_2d_rend"]
                    x1, y1, x2, y2 = bbox
                    frame_bb2ds.append([x1, x2, y1, y2])
                elif "bbox" in bb:
                    bbox = bb["bbox"]
                    x_min, y_min, width, height = bbox
                    x1 = x_min
                    x2 = x_min + width
                    y1 = y_min
                    y2 = y_min + height
                    frame_bb2ds.append([x1, x2, y1, y2])
                else:
                    frame_bb2ds.append([float("nan")] * 4)
                frame_bb2d_inst_ids.append(inst_id)
            visible_obbs = ObbTW.stack(visible_obbs).add_padding(512)
            # Convert frame 2D BBs to tensor and pad to match obbs
            if len(frame_bb2ds) > 0:
                frame_bb2ds = torch.tensor(frame_bb2ds, dtype=torch.float32)
                frame_bb2d_inst_ids = torch.tensor(
                    frame_bb2d_inst_ids, dtype=torch.int64
                )
                num_pad = 512 - frame_bb2ds.shape[0]
                if num_pad > 0:
                    pad_vals = torch.full((num_pad, 4), float("nan"))
                    frame_bb2ds = torch.cat([frame_bb2ds, pad_vals], dim=0)
                    pad_inst_ids = torch.full((num_pad,), -1, dtype=torch.int64)
                    frame_bb2d_inst_ids = torch.cat(
                        [frame_bb2d_inst_ids, pad_inst_ids], dim=0
                    )
            else:
                frame_bb2ds = torch.full((512, 4), float("nan"))
                frame_bb2d_inst_ids = torch.full((512,), -1, dtype=torch.int64)

            if fisheye:
                pinhole_cam = cam
                cam_fish = get_base_aria_rgb_camera()
                cam_fish._data[0] = W
                cam_fish._data[1] = H
                cam_fish._data[2] = fx * 1.15
                cam_fish._data[3] = fy * 1.15
                cam_fish._data[4] = cx
                cam_fish._data[5] = cy
                if load_images:
                    image_fish = _fishify(image, cam, cam_fish)
                    image_fish = np.clip(image_fish, 0, 255)
                    image_fish = image_fish.astype(np.uint8)
                    depth_fish = _fishify(depth, depth_cam, cam_fish)
                    depth_fish = depth_fish.astype(np.int32)

                # Transform bb2d from pinhole to fisheye coordinates
                valid = ~torch.isnan(frame_bb2ds[:, 0])
                if valid.any():
                    tl = frame_bb2ds[valid][:, [0, 2]]
                    tr = frame_bb2ds[valid][:, [1, 2]]
                    bl = frame_bb2ds[valid][:, [0, 3]]
                    br = frame_bb2ds[valid][:, [1, 3]]

                    tl_rays, _ = pinhole_cam.unproject(tl[None])
                    tr_rays, _ = pinhole_cam.unproject(tr[None])
                    bl_rays, _ = pinhole_cam.unproject(bl[None])
                    br_rays, _ = pinhole_cam.unproject(br[None])

                    tl_fish, _ = cam_fish.project(tl_rays)
                    tr_fish, _ = cam_fish.project(tr_rays)
                    bl_fish, _ = cam_fish.project(bl_rays)
                    br_fish, _ = cam_fish.project(br_rays)

                    all_corners = torch.stack(
                        [tl_fish[0], tr_fish[0], bl_fish[0], br_fish[0]], dim=1
                    )
                    x1_new = all_corners[:, :, 0].min(dim=1).values
                    x2_new = all_corners[:, :, 0].max(dim=1).values
                    y1_new = all_corners[:, :, 1].min(dim=1).values
                    y2_new = all_corners[:, :, 1].max(dim=1).values

                    img_w, img_h = cam_fish.size[0].item(), cam_fish.size[1].item()
                    x1_new = x1_new.clamp(0, img_w - 1)
                    x2_new = x2_new.clamp(0, img_w - 1)
                    y1_new = y1_new.clamp(0, img_h - 1)
                    y2_new = y2_new.clamp(0, img_h - 1)

                    frame_bb2ds[valid, 0] = x1_new
                    frame_bb2ds[valid, 1] = x2_new
                    frame_bb2ds[valid, 2] = y1_new
                    frame_bb2ds[valid, 3] = y2_new

                cam = cam_fish
                depth_cam = cam_fish
                if load_images:
                    image = image_fish
                    depth = depth_fish

            if load_images:
                rgb_images.append(image)
                depth_images.append(depth)
                sdp_ws.append(sdp_w.clone())
            cams.append(cam)
            depth_cams.append(depth_cam)
            timestamps_ns.append(timestamp_ns)
            T_wcs.append(T_wc.clone())
            obbs.append(visible_obbs.clone())
            bb2ds.append(frame_bb2ds.clone())
            bb2d_inst_ids.append(frame_bb2d_inst_ids.clone())

            count += 1
            if count >= max_frames:
                if verbose:
                    print("Reached max count, exiting.")
                break

        T_wcs = torch.stack(T_wcs)
        cams = torch.stack(cams)
        obbs = torch.stack(obbs)
        timestamp_ns = torch.tensor(timestamps_ns)
        if load_images:
            sdp_ws = torch.stack(sdp_ws)
        bb2ds = torch.stack(bb2ds)
        bb2d_inst_ids = torch.stack(bb2d_inst_ids)

    N = len(T_wcs)
    assert N == len(cams)
    assert N == len(obbs)
    assert N == len(timestamp_ns)
    assert N == len(depth_cams)
    assert N == len(bb2ds)
    assert N == len(bb2d_inst_ids)
    if load_images:
        assert N == len(rgb_images)
        assert N == len(depth_images)
        assert N == len(sdp_ws)

    # Detect frame rate, should be 10 Hz.
    if len(timestamps_ns) > 1:
        hz = round(float(1.0 / (np.diff(np.array(timestamps_ns)) / 1e9).mean()))
    else:
        hz = 10  # default when only 1 frame
    if verbose:
        print(f"got rgb image {hz} Hz")

    out = {}
    out["rgb_images"] = rgb_images
    out["depth_images"] = depth_images
    out["cams"] = cams
    out["depth_cams"] = depth_cams
    out["T_wcs"] = T_wcs
    out["obbs"] = obbs
    out["timestamp_ns"] = timestamp_ns
    out["sdp_ws"] = sdp_ws
    out["bb2d"] = bb2ds
    out["bb2d_inst_ids"] = bb2d_inst_ids
    out["sem_id_to_name"] = sem_id_to_name
    out["inst_id_to_caption"] = inst_id_to_caption
    out["seq_name"] = seq_name
    out["hz"] = hz
    out["image_tags"] = image_tags
    return out


from loaders.base_loader import BaseLoader


class CALoader(BaseLoader):
    def __init__(
        self,
        seq_name,
        start_frame=0,
        skip_frames=1,
        max_frames=10,
        pinhole=False,
        resize=None,
        remove_structure=True,
        remove_large=True,
        min_dim=0.05,
        use_canny=True,
        num_samples=10000,
        filter_border_bbs=False,
        border_valid_ratio=0.95,
        frame_gt=False,
        bb2d_use_pseudo=True,  # Compute 2D BBs from 3D OBB projection
    ):
        path = "https://ml-site.cdn-apple.com/datasets/ca1m/val/" + seq_name + ".tar"
        out_dir = os.path.expanduser(f"~/data/ca1m/{seq_name}")
        tar_path = os.path.expanduser(f"~/data/ca1m/{seq_name}/{seq_name}.tar")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            if not os.path.exists(tar_path):
                print(f"Downloading {path}")
                cmd = f"wget {path} -O {tar_path}"
                os.system(cmd)

        fisheye = not pinhole
        out = _load_tar(
            tar_path,
            start_frame=start_frame,
            skip_frames=skip_frames,
            max_frames=max_frames,
            fisheye=fisheye,
            use_canny=use_canny,
            num_samples=num_samples,
            frame_gt=frame_gt,
        )
        self.rgb_images = out["rgb_images"]
        self.depth_images = out["depth_images"]
        self.cams = out["cams"]
        self.depth_cams = out["depth_cams"]
        self.Ts_wc = out["T_wcs"]
        self.obbs = out["obbs"]
        self.timestamp_ns = out["timestamp_ns"]
        self.sdp_ws = out["sdp_ws"]
        self.bb2ds = out.get(
            "bb2d", None
        )  # 2D bounding boxes in xxyy format (may not exist)
        self.bb2d_inst_ids = out.get(
            "bb2d_inst_ids", None
        )  # Instance IDs for each 2D bbox
        self.sem_id_to_name = out["sem_id_to_name"]
        sem_name_to_id = {v: k for k, v in self.sem_id_to_name.items()}
        self.sem_name_to_id = sem_name_to_id
        self.inst_id_to_caption = out["inst_id_to_caption"]
        self.seq_name = out["seq_name"]
        self.length = len(self.rgb_images)
        self.index = 0  # start_frame is now handled by _load_tar
        self.camera = "rgb"
        self.device_name = "ipad"
        self.pinhole = pinhole
        self.resize = resize
        print(f"Loaded {self.length} frames from {tar_path}")
        self.tar_path = tar_path
        self.local_root = out_dir
        self.remove_structure = remove_structure
        self.remove_large = remove_large
        self.max_dimension = 3.0  # meters
        self.min_dim = min_dim  # minimum dimension in meters (default 4cm)
        self.filter_border_bbs = filter_border_bbs
        self.border_valid_ratio = border_valid_ratio
        self.bb2d_use_pseudo = bb2d_use_pseudo
        self.num_border_bbs_filtered = (
            0  # Track number of 3DBBs filtered by border check
        )

        # Find semantic IDs for floor and wall classes to filter out
        self.structure_sem_ids = []
        if remove_structure:
            for sem_id, name in self.sem_id_to_name.items():
                if "floor" == name.lower() or "wall" == name.lower():
                    self.structure_sem_ids.append(sem_id)
                    print(f"==> filtering out: {name}: {sem_id}")

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        datum = {}
        img = self.rgb_images[self.index]
        img_torch = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        img_torch = img_torch[None]

        HH = img_torch.shape[2]
        WW = img_torch.shape[3]
        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)):
                resizeH, resizeW = self.resize
            else:
                resizeH = self.resize
                resizeW = self.resize
            img_torch = torch.nn.functional.interpolate(
                img_torch,
                size=(resizeH, resizeW),
                mode="bilinear",
                align_corners=True,
            )
        datum["img0"] = img_torch.float()

        cam = self.cams[self.index]
        if self.resize is not None:
            w_ratio = resizeW / WW
            h_ratio = resizeH / HH
            cam = cam.scale((w_ratio, h_ratio))
        datum["cam0"] = cam.float()

        # load depth
        depth = self.depth_images[self.index]
        depth_torch = torch.from_numpy(depth.astype(np.float32))
        depth_torch = depth_torch[None, None]
        HHd, WWd = depth_torch.shape[2:]
        if self.resize is not None:
            depth_torch = torch.nn.functional.interpolate(
                depth_torch,
                size=(resizeH, resizeW),
                mode="bilinear",
                align_corners=True,
            )
        datum["depth0"] = depth_torch

        depth_cam = self.depth_cams[self.index]
        if self.resize is not None:
            w_ratio = resizeW / WWd
            h_ratio = resizeH / HHd
            depth_cam = depth_cam.scale((w_ratio, h_ratio))
        datum["depth_cam0"] = depth_cam.float()

        T_wc = self.Ts_wc[self.index]
        T_wr = T_wc @ self.cams[self.index].T_camera_rig
        datum["T_world_rig0"] = T_wr.float()

        obbs = self.obbs[self.index].float().remove_padding()

        # Filter out floor/wall instances if remove_structure is True
        if len(self.structure_sem_ids) > 0:
            keep_mask = ~torch.isin(
                obbs.sem_id.squeeze(-1),
                torch.tensor(self.structure_sem_ids),
            )
            obbs = obbs[keep_mask]

        # Filter out large objects if remove_large is True
        if self.remove_large and len(obbs) > 0:
            # Get object dimensions from bb3_object (xmin, xmax, ymin, ymax, zmin, zmax)
            bb3 = obbs.bb3_object
            dims_x = bb3[:, 1] - bb3[:, 0]  # xmax - xmin
            dims_y = bb3[:, 3] - bb3[:, 2]  # ymax - ymin
            dims_z = bb3[:, 5] - bb3[:, 4]  # zmax - zmin
            max_dims = torch.max(torch.stack([dims_x, dims_y, dims_z], dim=1), dim=1)[0]
            keep_mask = max_dims <= self.max_dimension
            obbs = obbs[keep_mask]

        # Expand minimum object dimensions to min_dim
        if self.min_dim > 0 and len(obbs) > 0:
            bb3 = obbs.bb3_object.clone()
            dims_x = bb3[:, 1] - bb3[:, 0]  # xmax - xmin
            dims_y = bb3[:, 3] - bb3[:, 2]  # ymax - ymin
            dims_z = bb3[:, 5] - bb3[:, 4]  # zmax - zmin

            # Expand dimensions that are smaller than min_dim
            expand_x = (self.min_dim - dims_x).clamp(min=0) / 2
            expand_y = (self.min_dim - dims_y).clamp(min=0) / 2
            expand_z = (self.min_dim - dims_z).clamp(min=0) / 2

            bb3[:, 0] -= expand_x  # xmin
            bb3[:, 1] += expand_x  # xmax
            bb3[:, 2] -= expand_y  # ymin
            bb3[:, 3] += expand_y  # ymax
            bb3[:, 4] -= expand_z  # zmin
            bb3[:, 5] += expand_z  # zmax

            obbs.set_bb3_object(bb3)

        # Filter out OBBs that project to image border or invalid region
        if self.filter_border_bbs and len(obbs) > 0:
            num_before = len(obbs)
            cam = datum["cam0"]
            T_wr = datum["T_world_rig0"]
            # Add batch dimension for get_pseudo_bb2
            obbs_batched = obbs[None]  # (1, N, 165)
            _, bb2_valid = obbs_batched.get_pseudo_bb2(
                cam[None].unsqueeze(1),  # (1, 1, C)
                T_wr[None].unsqueeze(1),  # (1, 1, 12)
                num_samples_per_edge=50,
                valid_ratio=self.border_valid_ratio,
            )
            bb2_valid = bb2_valid.squeeze(0).squeeze(0)  # (N,)
            obbs = obbs[bb2_valid]
            self.num_border_bbs_filtered += num_before - len(obbs)

        datum["obbs"] = obbs
        datum["sdp_w"] = self.sdp_ws[self.index].float()
        datum["time_ns0"] = self.timestamp_ns[self.index]
        datum["rotated0"] = torch.tensor(False).reshape(1)
        datum["num_img"] = torch.tensor(1).reshape(1)

        # Compute or load 2D bounding boxes
        if self.bb2d_use_pseudo and len(obbs) > 0:
            # Compute pseudo 2D BBs by projecting 3D OBBs
            cam = datum["cam0"]
            T_wr = datum["T_world_rig0"]
            obbs_batched = obbs[None]  # (1, N, 165)
            bb2d, bb2d_valid = obbs_batched.get_pseudo_bb2(
                cam[None].unsqueeze(1),  # (1, 1, C)
                T_wr[None].unsqueeze(1),  # (1, 1, 12)
                num_samples_per_edge=10,
                valid_ratio=0.1667,
            )
            bb2d = bb2d.squeeze(0).squeeze(0)  # (N, 4) in xxyy format
            bb2d_valid = bb2d_valid.squeeze(0).squeeze(0)  # (N,)
            # Set invalid projections to NaN (keep aligned with obbs)
            bb2d[~bb2d_valid] = float("nan")
            datum["bb2d0"] = bb2d
        elif (
            self.bb2ds is not None and self.bb2d_inst_ids is not None and len(obbs) > 0
        ):
            # Use pre-computed 2D BBs from dataset (filtered by instance ID)
            obb_inst_ids = obbs.inst_id.squeeze(-1)  # (N,)
            # Get bb2ds and inst_ids for this frame (remove padding)
            frame_bb2ds = self.bb2ds[self.index]  # (512, 4)
            frame_bb2d_inst_ids = self.bb2d_inst_ids[self.index]  # (512,)
            # Filter out padded entries (inst_id == -1)
            valid_mask = frame_bb2d_inst_ids >= 0
            frame_bb2ds = frame_bb2ds[valid_mask]
            frame_bb2d_inst_ids = frame_bb2d_inst_ids[valid_mask]
            # Filter bb2ds to only include those with inst_ids in filtered obbs
            keep_mask = torch.isin(frame_bb2d_inst_ids, obb_inst_ids)
            filtered_bb2ds = frame_bb2ds[keep_mask]  # (M, 4) tensor
            datum["bb2d0"] = filtered_bb2ds  # Tensor of [x1, x2, y1, y2] per object

        self.index += 1
        return datum
