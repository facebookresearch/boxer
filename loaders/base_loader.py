# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import threading
from typing import Optional

import numpy as np
import torch

# Structure classes filtered out by default across all loaders
STRUCTURE_CLASSES = {"floor", "wall", "shelter"}


class BaseLoader:
    """Base class for all data loaders.

    All loaders yield datum dicts with at minimum:
        img0:         (1, C, H, W) torch.Tensor, float32 in [0, 1]
        cam0:         CameraTW intrinsics/extrinsics
        T_world_rig0: PoseTW world-from-rig pose
        sdp_w:        (N, 3) torch.Tensor semi-dense points in world
        time_ns0:     int timestamp in nanoseconds
        rotated0:     bool whether image was rotated 90 CW
        bb2d0:        (M, 4) torch.Tensor 2D bounding boxes
        obbs:         ObbTW ground-truth 3D bounding boxes

    Subclasses implement load(idx) and call _init_prefetch() at the end of
    __init__ to enable background prefetching. The base __next__/__iter__
    handle the iterator protocol, index management, and prefetch coordination.
    """

    camera: str = "unknown"
    device_name: str = "unknown"
    resize: Optional[tuple] = None

    # --- Iterator protocol with prefetch ---

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def load(self, idx) -> dict:
        """Load a single frame by index. Subclasses must implement this."""
        raise NotImplementedError

    def _init_prefetch(self):
        """Initialize and start prefetching. Call at end of subclass __init__."""
        self._prefetch_result = None
        self._prefetch_thread = None
        self._start_prefetch()

    def _start_prefetch(self):
        """Start prefetching the frame at self.index in a background thread."""
        if self.index >= self.length:
            return
        self._prefetch_result = None
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(self.index,), daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self, idx):
        """Background worker that loads a single frame."""
        try:
            self._prefetch_result = self.load(idx=idx)
        except Exception as e:
            self._prefetch_result = e

    def __next__(self) -> dict:
        if self.index >= self.length:
            raise StopIteration

        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            out = self._prefetch_result
            self._prefetch_thread = None
            self._prefetch_result = None
        else:
            out = self.load(idx=self.index)

        if isinstance(out, Exception):
            raise out

        self.index += 1
        self._start_prefetch()
        return out

    # --- Shared utility methods ---

    @staticmethod
    def img_to_tensor(img_np):
        """Convert HWC uint8 numpy image to (1, C, H, W) float32 tensor in [0, 1]."""
        return torch.from_numpy(img_np).permute(2, 0, 1).float()[None] / 255.0

    @staticmethod
    def pinhole_from_K(w, h, fx, fy, cx, cy, valid_radius=None):
        """Create a pinhole CameraTW from intrinsics with identity T_cam_rig."""
        from utils.tw.camera import CameraTW

        if valid_radius is None:
            vr_w, vr_h = 99999.0, 99999.0
        else:
            vr_w, vr_h = valid_radius
        T_cam_rig = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
        )
        cam_data = torch.tensor(
            [w, h, fx, fy, cx, cy, -1, 1e-3, vr_w, vr_h],
            dtype=torch.float32,
        )
        cam_data = torch.cat([cam_data, T_cam_rig])
        return CameraTW(cam_data)

    @staticmethod
    def sdp_from_depth(depth_np, fx, fy, cx, cy, R_wc, t_wc, num_samples=10000):
        """Sample semi-dense points from depth via uniform grid subsampling.

        Uses fast numpy pinhole unprojection (no CameraTW overhead).

        Args:
            depth_np: (H, W) float32 depth in meters (0 = invalid)
            fx, fy, cx, cy: pinhole intrinsics (float)
            R_wc: (3, 3) float32 world-from-camera rotation
            t_wc: (3,) float32 world-from-camera translation
            num_samples: target number of output points

        Returns:
            (num_samples, 3) torch.Tensor float32, NaN-padded if fewer valid points
        """
        if depth_np is None:
            return torch.zeros(0, 3, dtype=torch.float32)

        dh, dw = depth_np.shape
        step = max(1, int(np.sqrt(dh * dw / (num_samples * 2))))
        ys_grid, xs_grid = np.mgrid[0:dh:step, 0:dw:step]
        ys_flat = ys_grid.ravel()
        xs_flat = xs_grid.ravel()
        zz = depth_np[ys_flat, xs_flat]
        valid_mask = zz > 0
        ys_v = ys_flat[valid_mask]
        xs_v = xs_flat[valid_mask]
        zz_v = zz[valid_mask]

        if len(ys_v) > num_samples:
            idx = np.random.choice(len(ys_v), size=num_samples, replace=False)
            ys_v, xs_v, zz_v = ys_v[idx], xs_v[idx], zz_v[idx]

        if len(ys_v) == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        x3d = (xs_v.astype(np.float32) - cx) / fx * zz_v
        y3d = (ys_v.astype(np.float32) - cy) / fy * zz_v
        sdp_c = np.stack([x3d, y3d, zz_v], axis=-1)

        sdp_w_np = (sdp_c @ R_wc.T) + t_wc
        sdp_w = torch.from_numpy(sdp_w_np)

        if sdp_w.shape[0] < num_samples:
            num_pad = num_samples - sdp_w.shape[0]
            pad_vals = torch.full((num_pad, 3), float("nan"), dtype=torch.float32)
            sdp_w = torch.cat([sdp_w, pad_vals], dim=0)

        return sdp_w.float()

    @staticmethod
    def find_structure_sem_ids(sem_id_to_name, structure_classes=None):
        """Return list of sem_ids whose name matches a structure class."""
        if structure_classes is None:
            structure_classes = STRUCTURE_CLASSES
        ids = []
        for sem_id, name in sem_id_to_name.items():
            if name.lower() in structure_classes:
                ids.append(sem_id)
                print(f"==> filtering out: {name}: {sem_id}")
        return ids

    @staticmethod
    def filter_obbs_by_sem_id(obbs, structure_sem_ids):
        """Remove OBBs whose sem_id is in structure_sem_ids."""
        if len(structure_sem_ids) == 0 or len(obbs) == 0:
            return obbs
        keep_mask = ~torch.isin(
            obbs.sem_id.squeeze(-1),
            torch.tensor(structure_sem_ids),
        )
        return obbs[keep_mask]

    @staticmethod
    def filter_obbs_by_name(obbs, structure_classes=None):
        """Remove OBBs whose text label matches a structure class name."""
        if structure_classes is None:
            structure_classes = STRUCTURE_CLASSES
        if len(obbs) == 0:
            return obbs
        text_labels = obbs.text_string()
        keep_mask = torch.tensor(
            [t.lower() not in structure_classes for t in text_labels],
            dtype=torch.bool,
        )
        return obbs[keep_mask]

    @staticmethod
    def filter_obbs_large(obbs, max_dimension=3.0):
        """Remove OBBs whose largest dimension exceeds max_dimension."""
        if len(obbs) == 0:
            return obbs
        bb3 = obbs.bb3_object
        dims_x = bb3[:, 1] - bb3[:, 0]
        dims_y = bb3[:, 3] - bb3[:, 2]
        dims_z = bb3[:, 5] - bb3[:, 4]
        max_dims = torch.max(torch.stack([dims_x, dims_y, dims_z], dim=1), dim=1)[0]
        return obbs[max_dims <= max_dimension]

    @staticmethod
    def expand_obbs_min_dim(obbs, min_dim=0.05):
        """Expand OBB dimensions that are smaller than min_dim."""
        if min_dim <= 0 or len(obbs) == 0:
            return obbs
        bb3 = obbs.bb3_object.clone()
        dims_x = bb3[:, 1] - bb3[:, 0]
        dims_y = bb3[:, 3] - bb3[:, 2]
        dims_z = bb3[:, 5] - bb3[:, 4]
        expand_x = (min_dim - dims_x).clamp(min=0) / 2
        expand_y = (min_dim - dims_y).clamp(min=0) / 2
        expand_z = (min_dim - dims_z).clamp(min=0) / 2
        bb3[:, 0] -= expand_x
        bb3[:, 1] += expand_x
        bb3[:, 2] -= expand_y
        bb3[:, 3] += expand_y
        bb3[:, 4] -= expand_z
        bb3[:, 5] += expand_z
        obbs.set_bb3_object(bb3)
        return obbs
