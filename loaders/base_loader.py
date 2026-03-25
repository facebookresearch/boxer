# pyre-unsafe
from typing import Optional


class BaseLoader:
    """Base class for all data loaders.

    All loaders yield datum dicts with at minimum:
        img0:         (1, C, H, W) torch.Tensor, float32 in [0, 1]
        cam0:         CameraTW intrinsics/extrinsics
        T_world_rig0: PoseTW world-from-rig pose
        sdp_w:        (N, 3) torch.Tensor semi-dense points in world
        time_ns0:     int timestamp in nanoseconds
        rotated0:     bool whether image was rotated 90 CW
        num_img:      int number of images (usually 1)
        bb2d0:        (M, 4) torch.Tensor 2D bounding boxes
        obbs:         ObbTW ground-truth 3D bounding boxes
    """

    camera: str = "unknown"
    device_name: str = "unknown"
    resize: Optional[tuple] = None

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        raise NotImplementedError
