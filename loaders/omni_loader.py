# pyre-unsafe
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from utils.demo_utils import EVAL_PATH
from tw.camera import CameraTW, get_base_aria_rgb_camera
from tw.obb import ObbTW
from tw.pose import PoseTW
from tw.tensor_utils import pad_string, string2tensor

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
        data_root: Root directory of Omni3D data (e.g., ~/data/Omni3D)
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


def fishify(image, cam_pin, cam_fish):
    """Convert pinhole image to fisheye image.

    Args:
        image: Input image as numpy array (H, W, C) in uint8 [0, 255] or float [0, 1]
        cam_pin: Pinhole camera
        cam_fish: Fisheye camera

    Returns:
        Warped fisheye image as numpy array in same format as input
    """
    # Convert to float32 and normalize to [0, 1] for proper interpolation
    input_dtype = image.dtype
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)

    image_torch = torch.from_numpy(image_float)
    if torch.cuda.is_available():
        image_torch = image_torch.cuda()
        cam_fish = cam_fish.cuda()
        cam_pin = cam_pin.cuda()
    device = image_torch.device
    H, W = image_float.shape[:2]
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

    # Use padding_mode='zeros' for out-of-bounds pixels (black borders)
    image_torch2 = torch.nn.functional.grid_sample(
        image_torch[None],
        source[None],
        mode="bicubic",
        align_corners=False,
        padding_mode="zeros",
    )
    image_fish = image_torch2[0]

    # Clamp to [0, 1] to avoid artifacts from bicubic interpolation
    image_fish = image_fish.clamp(0.0, 1.0)

    if image_fish.shape[0] == 1:
        image_fish = image_fish.squeeze(0)
    else:
        image_fish = image_fish.permute(1, 2, 0)
    image_fish = image_fish.cpu().numpy()

    # Convert back to original dtype
    if input_dtype == np.uint8:
        image_fish = (image_fish * 255.0).astype(np.uint8)

    return image_fish


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
        fisheye: bool = False,
        remove_structure: bool = True,
        remove_large: bool = True,
        min_dim: float = 0.05,
        max_dimension: float = 3.0,
        debug: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        remove_no_3d: bool = True,
        with_da3_depth: bool = False,
    ):
        if data_root is None:
            data_root = os.path.expanduser("~/data/Omni3D")

        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.resize = None  # Will be set by BoxerNetWrapper
        self.camera = "omni3d"
        self.device_name = dataset_name
        self.category_filter = category_filter
        self.fisheye = fisheye
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

        self.info = data.get("info", {})
        self.categories = data.get("categories", [])
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])

        # Build lookup dictionaries
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self.cat_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}
        self.sem_id_to_name = self.cat_id_to_name
        self.sem_name_to_id = self.cat_name_to_id

        # Build annotation lookup by image_id
        self.img_id_to_anns = {}
        for ann in self.annotations:
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
        self.structure_sem_ids = []
        if remove_structure:
            for sem_id, name in self.sem_id_to_name.items():
                if "floor" == name.lower() or "wall" == name.lower():
                    self.structure_sem_ids.append(sem_id)
                    print(f"==> filtering out: {name}: {sem_id}")

        # DA3 depth setup
        self.with_da3_depth = with_da3_depth
        self.da3_dir = None
        if with_da3_depth:
            da3_dir = os.path.join(EVAL_PATH, "boxer", dataset_name, "da3")
            if os.path.isdir(da3_dir):
                self.da3_dir = da3_dir
                print(f"==> DA3 depth directory: {da3_dir}")
            else:
                print(
                    f"==> Warning: DA3 depth dir not found: {da3_dir}, falling back to native depth"
                )

        print(
            f"Loaded {self.length} images from {dataset_name} {split} "
            f"({len(self.annotations)} total annotations)"
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        img_info = self.images[self.index]
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
        # Note: dist_params must be empty (0 elements) for pinhole camera
        # If dist_params has 4 elements, it will be detected as KB4 fisheye
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_pin_data = torch.tensor(
            [resizeW, resizeH, fx, fy, cx, cy, -1, 1e-3, 99999.0, 99999.0],
            dtype=torch.float32,
        )
        cam_pin_data = torch.cat(
            [cam_pin_data, T_cam_rig]
        )  # No dist_params for pinhole
        cam_pin = CameraTW(cam_pin_data)

        # Resize image if needed
        if self.resize is not None:
            img_pil = img.resize((resizeW, resizeH), Image.BILINEAR)
            img_np = np.array(img_pil)

        if self.fisheye:
            # Create fisheye camera (BoxerNet was trained on fisheye images)
            cam_fish = get_base_aria_rgb_camera()
            cam_fish._data[0] = resizeW
            cam_fish._data[1] = resizeH
            cam_fish._data[2] = fx * 1.15  # mitigate black borders
            cam_fish._data[3] = fy * 1.15
            cam_fish._data[4] = cx
            cam_fish._data[5] = cy

            # Convert pinhole image to fisheye
            img_out = fishify(img_np, cam_pin, cam_fish)
            # Ensure camera is on CPU with contiguous data (fishify may move to GPU)
            cam_out = CameraTW(cam_fish._data.detach().cpu().clone())
        else:
            # Use pinhole camera directly
            img_out = img_np
            cam_out = cam_pin

        # Convert to torch tensor [1, 3, H, W] normalized to [0, 1]
        img_torch = torch.from_numpy(img_out).permute(2, 0, 1).float() / 255.0
        img_torch = img_torch[None]

        datum["img0"] = img_torch.float()
        datum["cam0"] = cam_out.float()

        # Load depth image
        # Store pinhole depth for sdp_w calculation (before any fisheye transformation)
        depth_np_pinhole = None

        # Try DA3 depth first (if enabled)
        da3_depth_loaded = False
        if self.da3_dir is not None:
            img_id = img_info["id"]
            da3_path = os.path.join(self.da3_dir, f"{img_id}.png")
            if os.path.exists(da3_path):
                depth_img = cv2.imread(da3_path, cv2.IMREAD_UNCHANGED)
                depth_np = depth_img.astype(np.float32) / 1000.0  # mm -> meters

                # Resize to match image if needed
                if depth_np.shape[:2] != (resizeH, resizeW):
                    depth_np = cv2.resize(
                        depth_np, (resizeW, resizeH), interpolation=cv2.INTER_NEAREST
                    )

                depth_np_pinhole = depth_np.copy()

                if self.fisheye:
                    depth_out = fishify(depth_np, cam_pin, cam_fish)
                else:
                    depth_out = depth_np

                depth_torch = torch.from_numpy(depth_out).float()[None, None]
                datum["depth0"] = depth_torch
                datum["depth_cam0"] = cam_out.float()
                da3_depth_loaded = True

        if not da3_depth_loaded:
            # Native depth loading (SUNRGBD only, other datasets get zero depth)
            if self.dataset_name == "SUNRGBD":
                # Construct depth path from image path
                # Image: SUNRGBD/kv2/.../image/xxx.jpg
                # Depth: SUNRGBD/kv2/.../depth/xxx.png (raw depth, needs >>3 for meters)
                depth_path = img_info["file_path"].replace("/image/", "/depth/")
                depth_path = depth_path.replace(".jpg", ".png")
                depth_full_path = os.path.join(self.data_root, depth_path)

                # Some SUNRGBD subsets (b3dodata, sun3ddata) have depth files with
                # different names than the image. Fall back to the single .png in
                # the depth directory when the exact name doesn't match.
                if not os.path.exists(depth_full_path):
                    depth_dir = os.path.dirname(depth_full_path)
                    if os.path.isdir(depth_dir):
                        pngs = [f for f in os.listdir(depth_dir) if f.endswith(".png")]
                        if len(pngs) == 1:
                            depth_full_path = os.path.join(depth_dir, pngs[0])

                if os.path.exists(depth_full_path):
                    # Load depth as uint16 and convert to meters
                    depth_img = Image.open(depth_full_path)
                    depth_np = np.array(depth_img, dtype=np.float32)
                    # SUNRGBD depth encoding: (raw >> 3) / 1000 = raw / 8000 to get meters
                    depth_np = depth_np / 8000.0

                    # Resize depth if needed (matching image resize)
                    if self.resize is not None:
                        from PIL import Image as PILImage

                        depth_pil = PILImage.fromarray(depth_np)
                        depth_pil = depth_pil.resize(
                            (resizeW, resizeH), PILImage.NEAREST
                        )
                        depth_np = np.array(depth_pil)

                    # Store pinhole depth for sdp_w calculation
                    depth_np_pinhole = depth_np.copy()

                    # Apply fisheye transformation if needed (for visualization only)
                    if self.fisheye:
                        # Fishify depth image (similar to RGB)
                        depth_fish = fishify(depth_np, cam_pin, cam_fish)
                        depth_out = depth_fish
                    else:
                        depth_out = depth_np

                    # Convert to torch tensor [1, 1, H, W]
                    depth_torch = torch.from_numpy(depth_out).float()
                    depth_torch = depth_torch[None, None]  # Add batch and channel dims

                    datum["depth0"] = depth_torch
                    datum["depth_cam0"] = cam_out.float()  # Same camera as RGB
                else:
                    # No depth available, provide zeros
                    depth_torch = torch.zeros(
                        1, 1, resizeH, resizeW, dtype=torch.float32
                    )
                    datum["depth0"] = depth_torch
                    datum["depth_cam0"] = cam_out.float()
            else:
                # For non-SUNRGBD datasets, provide zero depth
                depth_torch = torch.zeros(1, 1, resizeH, resizeW, dtype=torch.float32)
                datum["depth0"] = depth_torch
                datum["depth_cam0"] = cam_out.float()

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

        # Create semi-dense points from depth using Canny edge sampling
        # (similar to ca_loader.py _load_tar())
        # Use pinhole depth and camera for correct 3D unprojection
        num_samples = 10000
        if depth_np_pinhole is not None:
            depth_for_sdp = depth_np_pinhole  # Use pinhole depth (before fisheye)
            H_sdp, W_sdp = depth_for_sdp.shape

            # Check if we have valid depth
            valid_depth_mask = depth_for_sdp > 0
            if valid_depth_mask.sum() > 100:
                # Use Canny edges on RGB image to sample points
                # Use original pinhole image for Canny (before fisheye)
                if img_np.dtype != np.uint8:
                    img_for_canny = (img_np * 255).astype(np.uint8)
                else:
                    img_for_canny = img_np
                # Resize to match depth size if needed
                if img_for_canny.shape[:2] != (H_sdp, W_sdp):
                    img_for_canny = cv2.resize(img_for_canny, (W_sdp, H_sdp))
                # Convert to grayscale for Canny
                if img_for_canny.ndim == 3:
                    img_gray = cv2.cvtColor(img_for_canny, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_for_canny

                try:
                    edges = cv2.Canny(img_gray, 30, 60)
                    # Mask out edges where depth is invalid
                    edges = edges.astype(np.float64) * valid_depth_mask.astype(
                        np.float64
                    )
                    weights_flat = edges.ravel()
                    if weights_flat.sum() > 0:
                        weights_flat /= weights_flat.sum()
                        idx = np.random.choice(
                            H_sdp * W_sdp,
                            size=num_samples,
                            replace=True,
                            p=weights_flat,
                        )
                        ys, xs = np.unravel_index(idx, (H_sdp, W_sdp))
                    else:
                        # Fallback to random sampling on valid depth pixels
                        valid_ys, valid_xs = np.where(valid_depth_mask)
                        if len(valid_ys) > 0:
                            idx = np.random.choice(
                                len(valid_ys),
                                size=min(num_samples, len(valid_ys)),
                                replace=True,
                            )
                            ys, xs = valid_ys[idx], valid_xs[idx]
                        else:
                            xs = np.array([])
                            ys = np.array([])
                except Exception:
                    # Fallback to random sampling on valid depth pixels
                    valid_ys, valid_xs = np.where(valid_depth_mask)
                    if len(valid_ys) > 0:
                        idx = np.random.choice(
                            len(valid_ys),
                            size=min(num_samples, len(valid_ys)),
                            replace=True,
                        )
                        ys, xs = valid_ys[idx], valid_xs[idx]
                    else:
                        xs = np.array([])
                        ys = np.array([])

                if len(xs) > 0:
                    # Unproject points to 3D using PINHOLE camera (not fisheye)
                    points = np.stack([xs, ys], axis=-1).astype(np.float32)
                    points3, valid = cam_pin.unproject(torch.from_numpy(points)[None])
                    points3 = points3[0].numpy()
                    valid = valid[0].numpy()

                    # Get depth values at sampled points
                    zz = depth_for_sdp[ys.astype(int), xs.astype(int)]

                    # Scale rays by depth to get 3D points in camera coords
                    sdp_c = points3 * zz.reshape(-1, 1)

                    # Filter out invalid points (zero depth or invalid projection)
                    valid_pts = valid & (zz > 0)
                    sdp_c = sdp_c[valid_pts]

                    # Transform to world coordinates
                    sdp_c_torch = torch.from_numpy(sdp_c.astype(np.float32))
                    T_wr = datum["T_world_rig0"]
                    sdp_w = T_wr * sdp_c_torch

                    # Pad to num_samples if needed
                    if sdp_w.shape[0] < num_samples:
                        num_pad = num_samples - sdp_w.shape[0]
                        pad_vals = torch.full(
                            (num_pad, 3), float("nan"), dtype=torch.float32
                        )
                        sdp_w = torch.cat([sdp_w, pad_vals], dim=0)

                    datum["sdp_w"] = sdp_w.float()
                else:
                    datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)
            else:
                # Not enough valid depth pixels
                datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)
        else:
            # No depth available
            datum["sdp_w"] = torch.zeros(0, 3, dtype=torch.float32)

        # Use image id as timestamp
        datum["time_ns0"] = int(img_info["id"])

        # No rotation
        datum["rotated0"] = torch.tensor(False).reshape(1)

        # Number of images (always 1 for Omni3D)
        datum["num_img"] = torch.tensor(1).reshape(1)

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
            debug_stats["total"] += 1

            # Skip invalid 3D annotations
            if not ann.get("valid3D", True):
                debug_stats["invalid_valid3D"] += 1
                continue

            # Get 3D box corners
            bbox3d_cam = ann.get("bbox3D_cam")
            if bbox3d_cam is None:
                debug_stats["no_bbox3D_cam"] += 1
                continue

            corners_3d = np.array(bbox3d_cam)

            # Check for invalid corner data
            if corners_3d.shape != (8, 3):
                debug_stats["bad_shape"] += 1
                continue

            # Skip boxes with invalid center (marked as [-1, -1, -1])
            center_cam = ann.get("center_cam", [0, 0, 0])
            if center_cam == [-1, -1, -1]:
                debug_stats["invalid_center"] += 1
                continue

            # Skip boxes with any corner behind camera (z <= 0)
            # Note: In Omni3D camera coords, +z is toward scene
            if np.any(corners_3d[:, 2] <= 0):
                debug_stats["behind_camera"] += 1
                continue

            cat_id = ann["category_id"]
            cat_name = self.cat_id_to_name.get(cat_id, "unknown")

            # Apply category filter if specified
            if self.category_filter is not None:
                if cat_name not in self.category_filter:
                    debug_stats["category_filtered"] += 1
                    continue

            # Create ObbTW from corners
            obb = corners_to_obb(corners_3d, cat_id, cat_name)
            obb_list.append(obb)
            debug_stats["kept"] += 1

        if self.debug and debug_stats["total"] > 0:
            print(f"  [DEBUG] Frame {self.index}: {debug_stats}")

        # Stack OBBs into single ObbTW tensor
        if len(obb_list) > 0:
            obbs = ObbTW(torch.stack([obb._data for obb in obb_list]))
        else:
            obbs = ObbTW(torch.zeros(0, 165))

        # Apply filters (similar to CALoader)
        # Filter out floor/wall instances if remove_structure is True
        if len(self.structure_sem_ids) > 0 and len(obbs) > 0:
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

        datum["obbs"] = obbs

        self.index += 1
        return datum
