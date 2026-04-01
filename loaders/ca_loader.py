# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
import glob
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from utils.tw.camera import CameraTW
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor

from loaders.base_loader import BaseLoader


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_world_obbs(data_dir):
    """Load world-level OBBs and metadata from an extracted CA-1M directory."""
    world_path = os.path.join(data_dir, "world.gt", "instances.json")
    if not os.path.exists(world_path):
        raise ValueError(f"No world.gt/instances.json found in {data_dir}")

    bb3 = _read_json(world_path)
    id2inst = {}
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

    category_names = [bb["category"] for bb in bb3]
    unique_categories = sorted(set(category_names))
    sem_id_to_name = {i: cat for i, cat in enumerate(unique_categories)}
    category_to_sem_id = {cat: i for i, cat in enumerate(unique_categories)}
    sem_ids = torch.tensor(
        [category_to_sem_id[bb["category"]] for bb in bb3], dtype=torch.int64
    )

    text = [string2tensor(pad_string(xx, max_len=128)) for xx in category_names]
    text = torch.stack(text)
    all_obbs = ObbTW.from_lmc(
        bb3_object=sz,
        T_world_object=T_wo,
        sem_id=sem_ids,
        inst_id=inst_ids,
        text=text,
    )

    return all_obbs, id2inst, sem_id_to_name


class CALoader(BaseLoader):
    def __init__(
        self,
        seq_name,
        start_frame=0,
        skip_frames=1,
        max_frames=10,
        resize=None,
        remove_structure=True,
        remove_large=True,
        min_dim=0.05,
        num_samples=10000,
    ):
        seq_name = seq_name.strip("/")
        # Find the extracted data directory.
        from utils.demo_utils import SAMPLE_DATA_PATH

        sample = os.path.join(SAMPLE_DATA_PATH, seq_name)
        if os.path.exists(sample):
            out_dir = sample
        else:
            out_dir = os.path.join(SAMPLE_DATA_PATH, seq_name)
        world_files = glob.glob(
            os.path.join(out_dir, "**/world.gt/instances.json"), recursive=True
        )
        if not world_files:
            raise FileNotFoundError(
                f"No extracted CA-1M data found in {out_dir}. "
                f"Run: python scripts/download_ca1m_sample.py --video-id {seq_name.split('-')[-1]}"
            )
        data_dir = os.path.dirname(os.path.dirname(world_files[0]))

        # Load world OBBs (lightweight, just JSON parsing).
        all_obbs, id2inst, sem_id_to_name = _load_world_obbs(data_dir)
        self.all_obbs = all_obbs
        self.id2inst = id2inst
        self.sem_id_to_name = sem_id_to_name
        self.sem_name_to_id = {v: k for k, v in sem_id_to_name.items()}

        # Discover frame timestamps from .wide directories.
        wide_dirs = glob.glob(os.path.join(data_dir, "*.wide"))
        image_tags = sorted({os.path.basename(d).split(".")[0] for d in wide_dirs})
        image_tags = image_tags[start_frame:]
        image_tags = image_tags[::skip_frames]
        image_tags = image_tags[:max_frames]
        self.image_tags = image_tags

        self.data_dir = data_dir
        self.seq_name = seq_name
        self.length = len(image_tags)
        self.index = 0
        self.camera = "rgb"
        self.device_name = "ipad"
        self.resize = resize
        self.num_samples = num_samples
        self.remove_structure = remove_structure
        self.remove_large = remove_large
        self.max_dimension = 3.0
        self.min_dim = min_dim

        # Find semantic IDs for floor and wall classes to filter out
        self.structure_sem_ids = (
            self.find_structure_sem_ids(self.sem_id_to_name) if remove_structure else []
        )

        # Timestamps derived from image tags (always available without loading images).
        self.timestamp_ns = torch.tensor(
            [int(tag) for tag in image_tags], dtype=torch.int64
        )

        # Cached bulk data (populated by load_all() for viewer use).
        self.rgb_images = None
        self.Ts_wc = None
        self.cams = None
        self.sdp_ws = None

        print(f"Found {self.length} frames in {data_dir}")

        self._init_prefetch()

    def load_all(self):
        """Pre-load all frames into memory (for viewer random access).

        Populates self.rgb_images, self.Ts_wc, self.cams, self.sdp_ws.
        """
        from tqdm import tqdm

        rgb_images = []
        Ts_wc = []
        cams = []
        sdp_ws = []
        for tag in tqdm(self.image_tags, desc="Loading frames"):
            frame = self._load_frame(tag)
            rgb_images.append(frame["image"])
            Ts_wc.append(frame["T_wc"].clone())
            cams.append(frame["cam"])
            sdp_ws.append(frame["sdp_w"].clone())
        self.rgb_images = rgb_images
        self.Ts_wc = torch.stack(Ts_wc)
        self.cams = torch.stack(cams)
        self.sdp_ws = torch.stack(sdp_ws)

    def load_metadata(self, sdp_fps=1.0):
        """Load only poses, cameras, and a global SDP cloud (no images/OBBs).

        Much faster than load_all() — suitable for interactive viewers that
        load images on demand.

        Args:
            sdp_fps: Target FPS for SDP sampling (default 1.0).
        """
        from tqdm import tqdm

        Ts_wc = []
        cams = []
        for tag in tqdm(self.image_tags, desc="Loading metadata"):
            data_dir = self.data_dir
            # Load camera extrinsics
            RT = torch.tensor(
                _read_json(os.path.join(data_dir, tag + ".gt", "RT.json"))
            )
            T_wc = PoseTW.from_Rt(RT[:3, :3], RT[:3, 3])
            Ts_wc.append(T_wc)
            # Load camera intrinsics
            K = torch.tensor(
                _read_json(os.path.join(data_dir, tag + ".wide", "image", "K.json"))
            )
            # Need image size for camera — read just the header
            img_path = os.path.join(data_dir, tag + ".wide", "image.png")
            with Image.open(img_path) as im:
                W, H = im.size
            params = torch.tensor(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32
            )
            cam = CameraTW.from_surreal(
                width=W, height=H, type_str="pinhole", params=params
            )
            cams.append(cam)

        self.Ts_wc = torch.stack(Ts_wc)
        self.cams = torch.stack(cams)

        # Build global SDP from a subset of frames at ~sdp_fps
        timestamps_s = self.timestamp_ns.float() / 1e9
        dt = 1.0 / sdp_fps
        sdp_chunks = []
        last_sdp_time = -float("inf")
        for i, tag in enumerate(self.image_tags):
            t = timestamps_s[i].item()
            if t - last_sdp_time < dt:
                continue
            last_sdp_time = t
            data_dir = self.data_dir
            RT = torch.tensor(
                _read_json(os.path.join(data_dir, tag + ".gt", "RT.json"))
            )
            K = torch.tensor(
                _read_json(os.path.join(data_dir, tag + ".wide", "image", "K.json"))
            )
            depth_raw = np.array(
                Image.open(os.path.join(data_dir, tag + ".wide", "depth.png"))
            )
            img_path = os.path.join(data_dir, tag + ".wide", "image.png")
            with Image.open(img_path) as im:
                W, H = im.size
            depth_resize = cv2.resize(
                depth_raw, (W, H), interpolation=cv2.INTER_NEAREST
            )
            depth_m = depth_resize.astype(np.float32) / 1000.0
            R_wc = RT[:3, :3].numpy().astype(np.float32)
            t_wc = RT[:3, 3].numpy().astype(np.float32)
            sdp_w = self.sdp_from_depth(
                depth_m,
                K[0, 0].item(),
                K[1, 1].item(),
                K[0, 2].item(),
                K[1, 2].item(),
                R_wc,
                t_wc,
                self.num_samples,
            )
            if len(sdp_w) > 0:
                sdp_chunks.append(sdp_w)
        if sdp_chunks:
            self.sdp_global = torch.cat(sdp_chunks, dim=0)
        else:
            self.sdp_global = torch.zeros(0, 3)
        print(
            f"Built global SDP: {len(self.sdp_global)} points from {len(sdp_chunks)} frames"
        )

    def _load_frame(self, image_tag):
        """Load all data for a single frame from disk."""
        data_dir = self.data_dir

        # Load RGB image.
        image = np.array(
            Image.open(os.path.join(data_dir, image_tag + ".wide", "image.png"))
        )
        H, W = image.shape[:2]

        # Load camera extrinsics.
        RT = torch.tensor(
            _read_json(os.path.join(data_dir, image_tag + ".gt", "RT.json"))
        )
        T_wc = PoseTW.from_Rt(RT[:3, :3], RT[:3, 3])

        # Load camera intrinsics.
        K = torch.tensor(
            _read_json(os.path.join(data_dir, image_tag + ".wide", "image", "K.json"))
        )
        params = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
        cam = CameraTW.from_surreal(
            width=W, height=H, type_str="pinhole", params=params
        )

        # Load depth and sample semi-dense points (uniform grid subsampling).
        depth_raw = np.array(
            Image.open(os.path.join(data_dir, image_tag + ".wide", "depth.png"))
        )
        depth_resize = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_m = depth_resize.astype(np.float32) / 1000.0
        R_wc = RT[:3, :3].numpy().astype(np.float32)
        t_wc = RT[:3, 3].numpy().astype(np.float32)
        sdp_w = self.sdp_from_depth(
            depth_m,
            params[0].item(),
            params[1].item(),
            params[2].item(),
            params[3].item(),
            R_wc,
            t_wc,
            self.num_samples,
        )

        # Load per-frame 3D bounding boxes.
        bb3 = _read_json(os.path.join(data_dir, image_tag + ".wide", "instances.json"))
        visible_obbs = []
        for bb in bb3:
            id_ = bb["id"]
            if id_ not in self.id2inst:
                continue
            inst_id = self.id2inst[id_]
            visible_obbs.append(self.all_obbs[inst_id].clone())
        if len(visible_obbs) > 0:
            visible_obbs = ObbTW.stack(visible_obbs).add_padding(512)
        else:
            visible_obbs = ObbTW(torch.zeros(0, 165))

        return {
            "image": image,
            "cam": cam,
            "T_wc": T_wc,
            "obbs": visible_obbs,
            "sdp_w": sdp_w,
            "timestamp_ns": int(image_tag),
        }

    def load(self, idx):
        """Load a single frame by index and return a datum dict."""
        image_tag = self.image_tags[idx]
        frame = self._load_frame(image_tag)

        datum = {}
        img = frame["image"]
        img_torch = self.img_to_tensor(img)

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

        cam = frame["cam"]
        if self.resize is not None:
            cam = cam.scale((resizeW / WW, resizeH / HH))
        datum["cam0"] = cam.float()

        T_wc = frame["T_wc"]
        T_wr = T_wc @ frame["cam"].T_camera_rig
        datum["T_world_rig0"] = T_wr.float()

        obbs = frame["obbs"].float().remove_padding()

        # Filter out floor/wall instances
        obbs = self.filter_obbs_by_sem_id(obbs, self.structure_sem_ids)

        # Filter out large objects and expand minimum dimensions
        if self.remove_large:
            obbs = self.filter_obbs_large(obbs, self.max_dimension)
        obbs = self.expand_obbs_min_dim(obbs, self.min_dim)

        datum["obbs"] = obbs
        datum["sdp_w"] = frame["sdp_w"].float()
        datum["time_ns0"] = frame["timestamp_ns"]
        datum["rotated0"] = torch.tensor(False).reshape(1)

        # Compute pseudo 2D bounding boxes from OBBs
        if len(obbs) > 0:
            obbs_batched = obbs[None]
            bb2d, bb2d_valid = obbs_batched.get_pseudo_bb2(
                datum["cam0"][None].unsqueeze(1),
                datum["T_world_rig0"][None].unsqueeze(1),
                num_samples_per_edge=10,
                valid_ratio=0.1667,
            )
            bb2d = bb2d.squeeze(0).squeeze(0)
            bb2d_valid = bb2d_valid.squeeze(0).squeeze(0)
            bb2d[~bb2d_valid] = float("nan")
            datum["bb2d0"] = bb2d

        return datum
