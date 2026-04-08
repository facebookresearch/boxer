# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import json
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from utils.demo_utils import SAMPLE_DATA_PATH
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor

# Supported Omni3D datasets
OMNI3D_DATASETS = [
    "SUNRGBD",
    "ARKitScenes",
    "KITTI",
    "Hypersim",
    "Objectron",
    "nuScenes",
]


def load_sunrgbd_extrinsics(data_root: str, file_path: str) -> Optional[np.ndarray]:
    """Load SUNRGBD extrinsics (camera-to-world rotation) from the extrinsics folder.

    Args:
        data_root: Root directory of Omni3D data (e.g., sample_data/Omni3D)
        file_path: Image file path from Omni3D JSON (e.g., SUNRGBD/kv2/.../image/xxx.jpg)

    Returns:
        3x3 rotation matrix if found, None otherwise
    """
    # Navigate from image path to extrinsics folder
    # file_path format: SUNRGBD/kv2/.../image/xxx.jpg
    # extrinsics is at: SUNRGBD/kv2/.../extrinsics/
    image_dir = os.path.dirname(file_path)  # .../image
    base_dir = os.path.dirname(image_dir)  # .../
    extrinsics_dir = os.path.join(data_root, base_dir, "extrinsics")

    if not os.path.exists(extrinsics_dir):
        return None

    # Find the extrinsics file (usually one .txt file)
    txt_files = [f for f in os.listdir(extrinsics_dir) if f.endswith(".txt")]
    if len(txt_files) == 0:
        return None

    extrinsics_path = os.path.join(extrinsics_dir, txt_files[0])
    try:
        with open(extrinsics_path, "r") as f:
            content = f.read()
        lines = content.strip().split("\n")
        matrix = np.array([[float(x) for x in line.split()] for line in lines])
        # Extract 3x3 rotation from 3x4 matrix
        if matrix.shape == (3, 4):
            return matrix[:3, :3]
        elif matrix.shape == (4, 4):
            return matrix[:3, :3]
        elif matrix.shape == (3, 3):
            return matrix
    except Exception:
        pass
    return None


def corners_to_obb(
    corners_3d: np.ndarray, category_id: int, category_name: str, prob: float = 1.0
) -> ObbTW:
    """
    Create an ObbTW object from 8 3D corners.

    For Omni3D, annotations are in camera coordinates, so we treat camera
    frame as world frame (T_world_rig = identity).

    Args:
        corners_3d: (8, 3) array of 3D corners
        category_id: Category ID (semantic ID)
        category_name: Category name string (for text_string() support)
        prob: Confidence/probability score

    Returns:
        ObbTW object
    """
    corners_tensor = torch.tensor(corners_3d, dtype=torch.float32)
    # Call from_corners with just the corners (due to @autocast decorator)
    obb = ObbTW.from_corners(corners_tensor)
    # Set sem_id, prob, and text using setter methods
    obb.set_sem_id(torch.tensor([category_id], dtype=torch.float32))
    obb.set_prob(torch.tensor([prob], dtype=torch.float32))
    # Set text field for text_string() support (used by demo_boxer.py --gt2d)
    text_tensor = string2tensor(pad_string(category_name, max_len=128))
    obb.set_text(text_tensor)
    return obb


from loaders.base_loader import BaseLoader


class OmniLoader(BaseLoader):
    """
    Data loader for Omni3D dataset images and 3D bounding boxes.
    Returns datum in the same format as CALoader for use with demo_boxer.py.

    Omni3D provides unified 3D annotations in camera coordinate system with:
    +x right, +y down, +z toward screen.
    """

    def __init__(
        self,
        dataset_name: str = "SUNRGBD",
        split: str = "val",
        data_root: Optional[str] = None,
        start_images: Optional[int] = None,
        max_images: Optional[int] = None,
        skip_images: int = 1,
        category_filter: Optional[List[str]] = None,
        remove_structure: bool = True,
        remove_large: bool = True,
        min_dim: float = 0.05,
        max_dimension: float = 3.0,
        debug: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        remove_no_3d: bool = True,
    ):
        if data_root is None:
            data_root = os.path.join(SAMPLE_DATA_PATH, "Omni3D")

        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.resize = None  # Will be set by BoxerNetWrapper
        self.camera = "omni3d"
        self.device_name = dataset_name
        self.category_filter = category_filter
        self.remove_structure = remove_structure
        self.remove_large = remove_large
        self.min_dim = min_dim
        self.max_dimension = max_dimension
        self.debug = debug
        self.remove_no_3d = remove_no_3d

        # Load JSON annotation file
        json_filename = f"{dataset_name}_{split}.json"
        json_path = os.path.join(data_root, json_filename)

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Omni3D JSON file not found: {json_path}\n"
                f"Please download it from the Omni3D dataset."
            )

        print(f"Loading {json_path}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        self.categories = data.get("categories", [])
        self.images = data.get("images", [])
        annotations = data.get("annotations", [])

        # Build lookup dictionaries
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self.sem_id_to_name = self.cat_id_to_name
        self.sem_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}

        # Build annotation lookup by image_id
        self.img_id_to_anns = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # Apply skip and max filters
        self.images = self.images[::skip_images]

        # Shuffle images if requested
        if shuffle:
            import random

            random.seed(seed)
            random.shuffle(self.images)
            print(f"==> Shuffled images with seed {seed}")

        if start_images is not None:
            self.images = self.images[start_images:]
            print(f"==> Starting from image {start_images}")

        if max_images is not None:
            self.images = self.images[:max_images]

        self.length = len(self.images)
        self.index = 0

        # Find semantic IDs for floor and wall classes to filter out
        self.structure_sem_ids = (
            self.find_structure_sem_ids(self.sem_id_to_name) if remove_structure else []
        )

        num_annotations = sum(len(v) for v in self.img_id_to_anns.values())
        print(
            f"Loaded {self.length} images from {dataset_name} {split} "
            f"({num_annotations} total annotations)"
        )

        self._init_prefetch()

    def load(self, idx):
        img_info = self.images[idx]
        datum = {}

        # Load image
        img_path = os.path.join(self.data_root, img_info["file_path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        HH, WW = img_np.shape[:2]

        # Scale intrinsics from K matrix
        K = img_info["K"]

        # Resize if requested
        if self.resize is not None:
            resizeH = resizeW = self.resize
            scale_x = resizeW / WW
            scale_y = resizeH / HH
        else:
            resizeH, resizeW = HH, WW
            scale_x = scale_y = 1.0

        fx = K[0][0] * scale_x
        fy = K[1][1] * scale_y
        cx = K[0][2] * scale_x
        cy = K[1][2] * scale_y

        # Create pinhole camera from K matrix
        cam_pin = self.pinhole_from_K(resizeW, resizeH, fx, fy, cx, cy)

        # Resize image if needed
        if self.resize is not None:
            img_pil = img.resize((resizeW, resizeH), Image.BILINEAR)
            img_np = np.array(img_pil)

        img_torch = self.img_to_tensor(img_np)

        datum["img0"] = img_torch.float()
        datum["cam0"] = cam_pin.float()

        # Load depth for semi-dense points (SUNRGBD only)
        depth_np_pinhole = None

        if self.dataset_name == "SUNRGBD":
            depth_path = img_info["file_path"].replace("/image/", "/depth/")
            depth_path = depth_path.replace(".jpg", ".png")
            depth_full_path = os.path.join(self.data_root, depth_path)

            # Some SUNRGBD subsets have depth files with different names than
            # the image. Fall back to the single .png in the depth directory.
            if not os.path.exists(depth_full_path):
                depth_dir = os.path.dirname(depth_full_path)
                if os.path.isdir(depth_dir):
                    pngs = [f for f in os.listdir(depth_dir) if f.endswith(".png")]
                    if len(pngs) == 1:
                        depth_full_path = os.path.join(depth_dir, pngs[0])

            if os.path.exists(depth_full_path):
                depth_img = Image.open(depth_full_path)
                depth_np = np.array(depth_img, dtype=np.float32) / 8000.0

                if self.resize is not None:
                    depth_pil = Image.fromarray(depth_np)
                    depth_pil = depth_pil.resize((resizeW, resizeH), Image.NEAREST)
                    depth_np = np.array(depth_pil)

                depth_np_pinhole = depth_np

        # Compose transformations for world pose:
        # 1. Load SUNRGBD extrinsics if available (camera-to-SUNRGBD-world rotation)
        # 2. Apply Y-Z swap to convert from Y-down to Z-down convention
        #
        # PoseTW format: [R_flat (9 elements), t (3 elements)] = 12 elements
        # R_flat is row-major: [R00, R01, R02, R10, R11, R12, R20, R21, R22]

        if self.dataset_name == "SUNRGBD":
            # Try to load SUNRGBD extrinsics (camera tilt relative to gravity)
            R_extrinsics = load_sunrgbd_extrinsics(
                self.data_root, img_info["file_path"]
            )
            if R_extrinsics is not None:
                # SUNRGBD extrinsics: camera-to-world rotation (already handles gravity tilt)
                # Need to compose with Y-Z swap rotation
                # R_yz_swap: Y-down to Z-down [[1,0,0],[0,0,1],[0,-1,0]]
                R_yz_swap = np.array(
                    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
                )
                # Compose: R_total = R_yz_swap @ R_extrinsics
                R_total = R_yz_swap @ R_extrinsics
                # Flatten row-major for PoseTW: [R_flat (9), t (3)]
                R_flat = R_total.flatten()
                T_wr_data = torch.tensor(
                    [
                        R_flat[0],
                        R_flat[1],
                        R_flat[2],
                        R_flat[3],
                        R_flat[4],
                        R_flat[5],
                        R_flat[6],
                        R_flat[7],
                        R_flat[8],
                        0,
                        0,
                        0,
                    ],
                    dtype=torch.float32,
                )
            else:
                # No extrinsics found, fall back to just Y-Z swap
                # R_yz_swap = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
                T_wr_data = torch.tensor(
                    [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], dtype=torch.float32
                )
        else:
            # For other datasets, just apply Y-Z swap
            # R_yz_swap = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
            T_wr_data = torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], dtype=torch.float32
            )

        datum["T_world_rig0"] = PoseTW(T_wr_data)

        # Create semi-dense points from depth (uniform grid subsampling)
        if depth_np_pinhole is not None:
            R_wc = T_wr_data[:9].reshape(3, 3).numpy().astype(np.float32)
            t_wc = T_wr_data[9:].numpy().astype(np.float32)
            datum["sdp_w"] = self.sdp_from_depth(
                depth_np_pinhole, fx, fy, cx, cy, R_wc, t_wc
            )
        else:
            datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)

        # Use image id as timestamp
        datum["time_ns0"] = int(img_info["id"])

        # No rotation
        datum["rotated0"] = torch.tensor(False).reshape(1)

        # Get GT 2D bounding boxes for this image
        img_id = img_info["id"]
        anns = self.img_id_to_anns.get(img_id, [])

        # Helper function to check if annotation has valid 3D box
        def has_valid_3d(ann):
            if not ann.get("valid3D", True):
                return False
            bbox3d_cam = ann.get("bbox3D_cam")
            if bbox3d_cam is None:
                return False
            corners_3d = np.array(bbox3d_cam)
            if corners_3d.shape != (8, 3):
                return False
            center_cam = ann.get("center_cam", [0, 0, 0])
            if center_cam == [-1, -1, -1]:
                return False
            if np.any(corners_3d[:, 2] <= 0):
                return False
            return True

        if len(anns) > 0:
            bb2d_list = []
            labels_list = []
            for ann in anns:
                # Skip annotations without valid 3D box if remove_no_3d is True
                if self.remove_no_3d and not has_valid_3d(ann):
                    continue
                # Omni3D uses bbox2D_tight in [x1, y1, x2, y2] format
                if "bbox2D_tight" not in ann:
                    continue
                bbox = ann["bbox2D_tight"]
                x1, y1, x2, y2 = bbox
                # Scale to resized image and convert to xxyy format for render_bb2
                xmin = x1 * scale_x
                xmax = x2 * scale_x
                ymin = y1 * scale_y
                ymax = y2 * scale_y
                bb2d_list.append([xmin, xmax, ymin, ymax])
                cat_id = ann["category_id"]
                labels_list.append(self.cat_id_to_name.get(cat_id, "unknown"))
            if len(bb2d_list) > 0:
                datum["bb2d0"] = torch.tensor(bb2d_list, dtype=torch.float32)
                datum["gt_labels"] = labels_list
            else:
                datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
                datum["gt_labels"] = []
        else:
            datum["bb2d0"] = torch.zeros(0, 4, dtype=torch.float32)
            datum["gt_labels"] = []

        # Get GT 3D bounding boxes (ObbTW) for this image
        obb_list = []
        if self.debug:
            debug_stats = {
                "total": 0,
                "invalid_valid3D": 0,
                "no_bbox3D_cam": 0,
                "bad_shape": 0,
                "invalid_center": 0,
                "behind_camera": 0,
                "category_filtered": 0,
                "kept": 0,
            }
        for ann in anns:
            if self.debug:
                debug_stats["total"] += 1

            # Skip invalid 3D annotations
            if not ann.get("valid3D", True):
                if self.debug:
                    debug_stats["invalid_valid3D"] += 1
                continue

            # Get 3D box corners
            bbox3d_cam = ann.get("bbox3D_cam")
            if bbox3d_cam is None:
                if self.debug:
                    debug_stats["no_bbox3D_cam"] += 1
                continue

            corners_3d = np.array(bbox3d_cam)

            # Check for invalid corner data
            if corners_3d.shape != (8, 3):
                if self.debug:
                    debug_stats["bad_shape"] += 1
                continue

            # Skip boxes with invalid center (marked as [-1, -1, -1])
            center_cam = ann.get("center_cam", [0, 0, 0])
            if center_cam == [-1, -1, -1]:
                if self.debug:
                    debug_stats["invalid_center"] += 1
                continue

            # Skip boxes with any corner behind camera (z <= 0)
            # Note: In Omni3D camera coords, +z is toward scene
            if np.any(corners_3d[:, 2] <= 0):
                if self.debug:
                    debug_stats["behind_camera"] += 1
                continue

            cat_id = ann["category_id"]
            cat_name = self.cat_id_to_name.get(cat_id, "unknown")

            # Apply category filter if specified
            if self.category_filter is not None:
                if cat_name not in self.category_filter:
                    if self.debug:
                        debug_stats["category_filtered"] += 1
                    continue

            # Create ObbTW from corners
            obb = corners_to_obb(corners_3d, cat_id, cat_name)
            obb_list.append(obb)
            if self.debug:
                debug_stats["kept"] += 1

        if self.debug and debug_stats["total"] > 0:
            print(f"  [DEBUG] Frame {idx}: {debug_stats}")

        # Stack OBBs into single ObbTW tensor
        if len(obb_list) > 0:
            obbs = ObbTW(torch.stack([obb._data for obb in obb_list]))
        else:
            obbs = ObbTW(torch.zeros(0, 165))

        # Filter out floor/wall instances if remove_structure is True
        obbs = self.filter_obbs_by_sem_id(obbs, self.structure_sem_ids)

        # Filter out large objects and expand minimum dimensions
        if self.remove_large:
            obbs = self.filter_obbs_large(obbs, self.max_dimension)
        obbs = self.expand_obbs_min_dim(obbs, self.min_dim)

        datum["obbs"] = obbs

        return datum
