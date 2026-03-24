# pyre-unsafe
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torchvision.io import read_image

from .pose import get_T_rot_z, IdentityPose, PoseTW
from .tensor_wrapper import autocast, autoinit, smart_cat, TensorWrapper


# Sample some random pixels in a circle around the camera center.
def random_fisheye_pixels(B, N, radius, cam_center):
    dists = radius * np.sqrt(np.random.rand(B, N)).astype(np.float32)
    angle = 2 * np.pi * np.random.rand(B, N).astype(np.float32)
    ptx = dists * np.cos(angle)
    pty = dists * np.sin(angle)
    pts = np.concatenate((ptx.reshape(B, N, 1), pty.reshape(B, N, 1)), axis=2)
    pts = pts + cam_center[None, None, :]
    return pts


def random_rect_pixels(B, N, H, W):
    pts = np.random.rand(B, N, 2).astype(np.float32)
    pts[..., 0] *= W - 1.0
    pts[..., 1] *= H - 1.0
    return pts


def is_fisheye624(inp):
    names = [
        "Fisheye624",
        "f624",
        "FisheyeRadTanThinPrism:f,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4",
        "FisheyeRadTanThinPrism:fu,fv,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4",
        "FisheyeRadTanThinPrism",
    ]
    names += [name.lower() for name in names]
    return inp in names


def is_kb4(inp):
    names = [
        "KB:fu,fv,u0,v0,k0,k1,k2,k3",
        "KannalaBrandtK4",
        "KB4",
    ]
    names += [name.lower() for name in names]
    return inp in names


def is_pinhole(inp):
    names = ["Pinhole", "Linear"]
    names += [name.lower() for name in names]
    return inp in names


class DefaultCameraTWData(TensorWrapper):
    """Allows multiple input sizes."""

    def __init__(self):
        self._data = -1 * torch.ones(33)

    @property
    def shape(self):
        return (torch.Size([34]), torch.Size([26]), torch.Size([22]))


class DefaultCameraTWParam(TensorWrapper):
    """Allows multiple input sizes."""

    def __init__(self):
        self._data = -1 * torch.ones(15)

    @property
    def shape(self):
        return (torch.Size([16]), torch.Size([15]), torch.Size([8]), torch.Size([4]))


class DefaultCameraTWDistParam(TensorWrapper):
    """Allows multiple input sizes."""

    def __init__(self):
        self._data = -1 * torch.ones(12)

    @property
    def shape(self):
        return (torch.Size([12]), torch.Size([4]), torch.Size([0]))


DEFAULT_CAM_DATA = DefaultCameraTWData()
DEFAULT_CAM_PARAM = DefaultCameraTWParam()
DEFAULT_CAM_DIST_PARAM = DefaultCameraTWDistParam()
DEFAULT_CAM_DATA_SIZE = 34

RGB_PARAMS = np.float32(
    # pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str,
    #  SupportsFloat, SupportsIndex]` but got `List[float]`.
    [2 * 600.0, 2 * 352.0, 2 * 352.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)
# pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str, SupportsFloat,
#  SupportsIndex]` but got `List[float]`.
SLAM_PARAMS = np.float32([500.0, 320.0, 240.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

FISHEYE624_TYPE_STR = (
    "FisheyeRadTanThinPrism:f,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4"
)
FISHEYE624_DF_TYPE_STR = (
    "FisheyeRadTanThinPrism:fu,fv,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4"
)
PINHOLE_TYPE_STR = "Pinhole"
KB4_TYPE_STR = "KB4"


def get_aria_camera(params=SLAM_PARAMS, width=640, height=480, valid_radius=None, B=1):
    type_str = FISHEYE624_TYPE_STR if params.shape[-1] == 15 else FISHEYE624_DF_TYPE_STR
    if valid_radius is None:
        cam = CameraTW.from_surreal(width, height, type_str, params)
    else:
        cam = CameraTW.from_surreal(
            width,
            height,
            type_str,
            params,
            valid_radius=valid_radius,
        )
    if B > 1:
        cam = cam.unsqueeze(0).repeat(B, 1)
    return cam


def get_pinhole_camera(params, width=640, height=480, valid_radius=None, B=1):
    type_str = PINHOLE_TYPE_STR
    if valid_radius is None:
        cam = CameraTW.from_surreal(width, height, type_str, params)
    else:
        cam = CameraTW.from_surreal(
            width,
            height,
            type_str,
            params,
            valid_radius=valid_radius,
        )
    if B > 1:
        cam = cam.unsqueeze(0).repeat(B, 1)
    return cam


def get_kb4_camera(B=1):
    HH = 640
    WW = 480
    fxy = 500.0
    cx = 320.0
    cy = 240.0
    dist = [1.0, 0.2615, 0.0096, -0.1034]
    valid_radius = WW
    params = [fxy, fxy, cx, cy] + dist
    type_str = KB4_TYPE_STR
    cam = CameraTW.from_surreal(
        WW,
        HH,
        type_str,
        params,
        valid_radius=valid_radius,
    )
    cam = cam.unsqueeze(0).repeat(B, 1)
    return cam


def get_base_aria_rgb_camera_full_res():
    params = RGB_PARAMS * 2
    params[1:3] += 32
    return get_aria_camera(params, 2880, 2880)


def get_base_aria_rgb_camera():
    return get_aria_camera(RGB_PARAMS, 1408, 1408)


def get_base_aria_slam_camera():
    return get_aria_camera(SLAM_PARAMS, 640, 480)


class CameraTW(TensorWrapper):
    """
    Class to represent a batch of camera calibrations of the same camera type.
    """

    SIZE_IND = slice(0, 2)
    F_IND = slice(2, 4)
    C_IND = slice(4, 6)
    GAIN_IND = 6
    EXPOSURE_S_IND = 7
    VALID_RADIUS_IND = slice(8, 10)
    T_CAM_RIG_IND = slice(10, 22)
    DIST_IND = slice(22, None)

    @autocast
    @autoinit
    def __init__(
        self, data: Union[torch.Tensor, DefaultCameraTWData] = DEFAULT_CAM_DATA
    ):
        assert isinstance(data, torch.Tensor)
        assert data.shape[-1] in [22, 26, 34]
        super().__init__(data)

    @classmethod
    @autoinit
    def from_parameters(
        cls,
        width: torch.Tensor = -1 * torch.ones(1),
        height: torch.Tensor = -1 * torch.ones(1),
        fx: torch.Tensor = -1 * torch.ones(1),
        fy: torch.Tensor = -1 * torch.ones(1),
        cx: torch.Tensor = -1 * torch.ones(1),
        cy: torch.Tensor = -1 * torch.ones(1),
        gain: torch.Tensor = -1 * torch.ones(1),
        exposure_s: torch.Tensor = 1e-3 * torch.ones(1),
        valid_radiusx: torch.Tensor = 99999.0 * torch.ones(1),
        valid_radiusy: torch.Tensor = 99999.0 * torch.ones(1),
        T_camera_rig: Union[torch.Tensor, PoseTW] = IdentityPose,  # 1x12.
        dist_params: Union[
            torch.Tensor, DefaultCameraTWDistParam
        ] = DEFAULT_CAM_DIST_PARAM,
    ):
        # Concatenate into one big data tensor, handles TensorWrapper objects.
        data = smart_cat(
            [
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                gain,
                exposure_s,
                valid_radiusx,
                valid_radiusy,
                T_camera_rig,
                dist_params,
            ],
            dim=-1,
        )
        return cls(data)

    @classmethod
    @autoinit
    def from_surreal(
        cls,
        width: torch.Tensor = -1 * torch.ones(1),
        height: torch.Tensor = -1 * torch.ones(1),
        type_str: str = "Fisheye624",
        params: Union[torch.Tensor, DefaultCameraTWParam] = DEFAULT_CAM_PARAM,
        gain: torch.Tensor = 1 * torch.ones(1),
        exposure_s: torch.Tensor = 1e-3 * torch.ones(1),
        valid_radius: torch.Tensor = 99999.0 * torch.ones(1),
        T_camera_rig: Union[torch.Tensor, PoseTW] = IdentityPose,  # 1x12.
    ):
        # Try to auto-determine the camera model.
        if (
            is_fisheye624(type_str) and params.shape[-1] == 16
        ):  # Fisheye624 double focals
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        elif (
            is_fisheye624(type_str) and params.shape[-1] == 15
        ):  # Fisheye624 single focal
            f = params[..., 0].unsqueeze(-1)
            cx = params[..., 1].unsqueeze(-1)
            cy = params[..., 2].unsqueeze(-1)
            dist_params = params[..., 3:]
            fx = fy = f
        elif is_kb4(type_str) and params.shape[-1] == 8:  # KB4.
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        elif is_pinhole(type_str) and params.shape[-1] == 4:  # Pinhole.
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        else:
            raise NotImplementedError(
                "Unknown number of params entered for camera model"
            )

        if torch.any(torch.logical_or(valid_radius > height, valid_radius > width)):
            if not is_pinhole(type_str):
                # Try to auto-determine the valid radius for fisheye cameras.
                default_radius = 99999.0
                hw_ratio = height / width
                eyevideo_camera_hw_ratio = torch.tensor(240.0 / 640.0).to(hw_ratio)
                slam_camera_hw_ratio = torch.tensor(480.0 / 640.0).to(hw_ratio)
                rgb_camera_hw_ratio = torch.tensor(2880.0 / 2880.0).to(hw_ratio)
                guess_rgb = hw_ratio == rgb_camera_hw_ratio
                guess_slam = hw_ratio == slam_camera_hw_ratio
                guess_eyevideo = hw_ratio == eyevideo_camera_hw_ratio
                valid_radius = default_radius * torch.ones_like(hw_ratio)
                # Assume "Rogallo"/"IMX577" aka Aria RGB Camera.
                valid_radius = torch.where(
                    guess_rgb, 1415 * (height / 2880), valid_radius
                )
                # Assume "Canyon"/"OV7251" aka Aria SLAM camera.
                valid_radius = torch.where(
                    guess_slam, 330 * (height / 480), valid_radius
                )
                # This is for Eye Video Camera
                valid_radius = torch.where(
                    guess_eyevideo, 330 * (height / 480), valid_radius
                )
                if torch.any(valid_radius == default_radius):
                    raise ValueError(
                        f"Failed to auto-determine valid radius based on aspect ratios (valid_radius {valid_radius}, width {width}, height {height})"
                    )
            else:
                # Note that the valid_radius for pinhole camera is not well-defined.
                # We heuristically set the valid radius to be the half of the image diagonal.
                # Add one pixel to be sure that all pixels in the image are valid.
                valid_radius = (
                    # pyre-fixme[58]: `**` is not supported for operand types
                    #  `Tensor` and `int`.
                    torch.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2) + 1.0
                )

        return cls.from_parameters(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            gain=gain,
            exposure_s=exposure_s,
            valid_radiusx=valid_radius,
            valid_radiusy=valid_radius,
            T_camera_rig=T_camera_rig,
            dist_params=dist_params,
        )

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., self.SIZE_IND]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., self.F_IND]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., self.C_IND]

    @property
    def K(self) -> torch.Tensor:
        """Intrinsic matrix with shape (..., 3, 3)"""
        K = torch.eye(3, device=self.device, dtype=self.dtype)
        # Make proper size of K to take care of B and T dims.
        K_view = [1] * (self.f.ndim - 1) + [3, 3]
        K_repeat = list(self.f.shape[:-1]) + [1, 1]
        K = K.view(K_view)
        K = K.repeat(K_repeat)
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]

        return K

    @property
    def gain(self) -> torch.Tensor:
        """Gain of the camera, with shape (..., 1)."""
        return self._data[..., self.GAIN_IND].unsqueeze(-1)

    @property
    def exposure_s(self) -> torch.Tensor:
        """Exposure of the camera in seconds, with shape (..., 1)."""
        return self._data[..., self.EXPOSURE_S_IND].unsqueeze(-1)

    @property
    def valid_radius(self) -> torch.Tensor:
        """Radius from camera center for valid projections, with shape (..., 1)."""
        return self._data[..., self.VALID_RADIUS_IND]

    @property
    def T_camera_rig(self) -> torch.Tensor:
        """Pose of camera, shape (..., 12)."""
        return PoseTW(self._data[..., self.T_CAM_RIG_IND])

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, D}), where D is number of distortion params."""
        return self._data[..., self.DIST_IND]

    @property
    def params(self) -> torch.Tensor:
        """Get the camera "params", which are defined as fx,fy,cx,cy,dist"""
        return torch.cat([self.f, self.c, self.dist], dim=-1)

    @property
    def is_fisheye624(self):
        return self.dist.shape[-1] == 12

    @property
    def is_kb4(self):
        return self.dist.shape[-1] == 4

    @property
    def is_linear(self):
        return self.dist.shape[-1] == 0

    def type_string(self):
        """Return a string describing the camera type."""
        if self.is_fisheye624:
            return "Fisheye624"
        elif self.is_kb4:
            return "KB4"
        elif self.is_linear:
            return "Pinhole"
        else:
            return f"Unknown(dist_dim={self.dist.shape[-1]})"

    def set_f(self, f: torch.Tensor):
        self._data[..., self.F_IND] = f

    def set_c(self, c: torch.Tensor):
        self._data[..., self.C_IND] = c

    def set_valid_radius(self, valid_radius: torch.Tensor):
        self._data[..., self.VALID_RADIUS_IND] = valid_radius

    def set_T_camera_rig(self, T_camera_rig: PoseTW):
        self._data[..., self.T_CAM_RIG_IND] = T_camera_rig._data.clone()

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the camera parameters after resizing an image."""
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat(
            [
                self.size * s,
                self.f * s,
                (self.c + 0.5) * s - 0.5,
                self.gain,
                self.exposure_s,
                self.valid_radius * s,
                # pyre-fixme[16]: `Tensor` has no attribute `_data`.
                self.T_camera_rig._data,
                self.dist,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def scale_to_size(self, size_wh: Union[int, Tuple[int]]):
        """Scale the camera parameters to a given image size"""
        if torch.unique(self.size).numel() > 2:
            raise ValueError(f"cannot handle multiple sizes {self.size}")
        if isinstance(size_wh, int):
            size_wh = (size_wh, size_wh)
        i0w = tuple([0] * self.ndim)
        i0h = tuple([0] * (self.ndim - 1) + [1])
        scale = (
            float(size_wh[0]) / float(self.size[i0w]),
            float(size_wh[1]) / float(self.size[i0h]),
        )
        # pyre-fixme[6]: For 1st argument expected `Union[Tuple[Union[float, int]],
        #  float, int]` but got `Tuple[float, float]`.
        return self.scale(scale)

    def scale_to(self, im: torch.Tensor):
        """
        Scale the camera parameters to match the size of the given image assumes
        ...xHxW image tensor convention of pytorch
        """
        H, W = im.shape[-2:]
        # pyre-fixme[6]: For 1st argument expected `Union[Tuple[int], int]` but got
        #  `Tuple[int, int]`.
        return self.scale_to_size((W, H))

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)

        # Expand the dimension if self._data is a tensor of CameraTW
        if len(self._data.shape) > 1:
            expand_dim = list(self._data.shape[:-1]) + [1]
            size = size.repeat(expand_dim)
            left_top = left_top.repeat(expand_dim)

        data = torch.cat(
            [
                size,
                self.f,
                self.c - left_top,
                self.gain,
                self.exposure_s,
                self.valid_radius,
                # pyre-fixme[16]: `Tensor` has no attribute `_data`.
                self.T_camera_rig._data,
                self.dist,
            ],
            dim=-1,
        )
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2, f"p2d shape needs to be 2d {p2d.shape}"
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), dim=-1)
        return valid

    @autocast
    def in_radius(self, p2d: torch.Tensor):
        """Check if 2D points are within the valid fisheye radius region."""
        assert p2d.shape[-1] == 2, f"p2d shape needs to be 2d {p2d.shape}"
        dists = torch.linalg.norm(
            (p2d - self.c.unsqueeze(-2)) / self.valid_radius.unsqueeze(-2),
            dim=-1,
            ord=2,
        )
        valid = dists < 1.0
        return valid

    @autocast
    def in_radius_mask(self):
        """
        Return a mask that is True where 2D points are within the valid fisheye
        radius region.  Returned mask is of shape ... x 1 x H x W, where ... is
        the shape of the camera (BxT or B for example).
        """
        s = self.shape[:-1]
        C = self.shape[-1]
        px = pixel_grid(self.view(-1, C)[0])
        H, W, _ = px.shape
        valids = self.in_radius(px.view(-1, 2))
        s = s + (1, H, W)
        valids = valids.view(s)
        return valids

    @autocast
    def in_fov(self, p3d: torch.Tensor, fov_deg: float):
        """Check if 3D points are within the valid FOV of the fisheye camera."""
        assert p3d.shape[-1] == 3, f"p3d shape needs to be 3d {p3d.shape}"
        # unproject the principal point to get the principal axis
        cam_center = self.c
        if self.c.ndim == 1:
            cam_center = self.c.unsqueeze(0)  # [34] -> [1, 34]
        # convert self.c from (..., 2) to (..., 1, 2)
        principal, _ = self.unproject(cam_center.unsqueeze(-2))
        principal = principal / torch.norm(principal, dim=-1, keepdim=True)
        # dot(principal, p3d) / (norm(principal) * norm(p3d))
        cos_angle = torch.sum(principal * p3d, dim=-1) / torch.norm(p3d, dim=-1)
        rad = torch.acos(cos_angle)
        fov_rad = np.deg2rad(fov_deg)
        valid = torch.isfinite(rad) & (rad < (fov_rad / 2.0)) & (cos_angle >= 0)
        return valid

    @autocast
    def project(
        self, p3d: torch.Tensor, fov_deg: float = 120.0, suppress_warning: bool = False
    ) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""

        # Try to auto-determine the camera model.
        if self.is_fisheye624:  # Fisheye624.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p3d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            p2d = fisheye624_project(p3d, params, suppress_warning=suppress_warning)
        elif self.is_kb4:  # KB4.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p3d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            p2d = kb4_project(p3d, params)
        elif self.is_linear:  # Pinhole.
            params = self.params
            if params.ndim == 1:
                B = p3d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            p2d = pinhole_project(p3d, params)
        else:
            raise ValueError("only fisheye624, pinhole, kb4 implemented")

        in_image = self.in_image(p2d)
        in_radius = self.in_radius(p2d)
        in_front = p3d[..., -1] > 0
        valid = in_image & in_radius & in_front

        valid = valid & self.in_fov(p3d, fov_deg)

        # pyre-fixme[7]: Expected `Tuple[Tensor]` but got `Tuple[Any, Any]`.
        #  Expected has length 1, but actual has length 2.
        return p2d, valid

    @autocast
    def unproject(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 2D points into 3D rays."""

        # Try to auto-determine the camera model.
        if self.is_fisheye624:  # Fisheye624.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p2d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            rays = fisheye624_unproject(p2d, params)
        elif self.is_kb4:  # KB4.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p2d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            rays = kb4_unproject(p2d, params)
        elif self.is_linear:  # Pinhole.
            params = self.params
            if params.ndim == 1:
                B = p2d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            rays = pinhole_unproject(p2d, params)
        else:
            raise ValueError("only fisheye624, pinhole, kb4 implemented")

        in_image = self.in_image(p2d)
        in_radius = self.in_radius(p2d)
        valid = in_image & in_radius
        # pyre-fixme[7]: Expected `Tuple[Tensor]` but got `Tuple[Any, Any]`.
        #  Expected has length 1, but actual has length 2.
        return rays, valid

    def rotate_90_cw(self):
        return self.rotate_90(clock_wise=True)

    def rotate_90_ccw(self):
        return self.rotate_90(clock_wise=False)

    def rotate_90(self, clock_wise: bool):
        dist_params = self.dist.clone()
        if self.is_fisheye624:
            # swap thin prism and tangential distortion parameters
            # {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3} to
            # {k_0 ... k_5} {p_1 p_0} {s_2 s_3 s_0 s_1}
            dist_p = self.dist[..., 6:8]
            dist_s = self.dist[..., 8:12]
            dist_params[..., 6] = dist_p[..., 1]
            dist_params[..., 7] = dist_p[..., 0]
            dist_params[..., 8:10] = dist_s[..., 2:]
            dist_params[..., 10:12] = dist_s[..., :2]
        elif self.is_linear:
            # no need to rotate distortion parameters since there are none
            pass
        elif self.is_kb4:
            # no need to rotate distortion parameters since they are rotationally
            # symmetric about the optical axis.
            pass
        else:
            raise NotImplementedError(f"camera model not recognized {self}")

        # clock-wise or counter clock-wise
        DIR = 1 if clock_wise else -1
        # rotate camera extrinsics by 90 degree CW
        T_rot_z = PoseTW.from_matrix3x4(get_T_rot_z(DIR * np.pi * 0.5)).to(self.device)
        if clock_wise:
            # rotate x, y of principal point
            # x_rotated = height - 1 - y_before
            # y_rotated = x_before
            rot_cx = self.size[..., 1] - self.c[..., 1] - 1
            rot_cy = self.c[..., 0].clone()
        else:
            rot_cx = self.c[..., 1].clone()
            rot_cy = self.size[..., 0] - self.c[..., 0] - 1

        return CameraTW.from_parameters(
            # swap width and height
            self.size[..., 1].clone().unsqueeze(-1),
            self.size[..., 0].clone().unsqueeze(-1),
            # swap x, y of focal lengths
            self.f[..., 1].clone().unsqueeze(-1),
            self.f[..., 0].clone().unsqueeze(-1),
            rot_cx.unsqueeze(-1),
            rot_cy.unsqueeze(-1),
            self.gain.clone(),
            self.exposure_s.clone(),
            # swap valid radius x, y
            self.valid_radius[..., 1].clone().unsqueeze(-1),
            self.valid_radius[..., 0].clone().unsqueeze(-1),
            # rotate camera extrinsics
            T_rot_z @ self.T_camera_rig,
            dist_params,
        )

    def __repr__(self):
        return f"CameraTW {self.shape} {self.dtype} {self.device}"


def grid_2d(
    width: int,
    height: int,
    output_range=(-1.0, 1.0, -1.0, 1.0),
    device="cpu",
    dtype=torch.float32,
):
    x = torch.linspace(
        output_range[0], output_range[1], width + 1, device=device, dtype=dtype
    )[:-1]
    y = torch.linspace(
        output_range[2], output_range[3], height + 1, device=device, dtype=dtype
    )[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx, yy], dim=-1)
    return grid


def pixel_grid(cam: CameraTW):
    assert cam.ndim == 1, f"Camera must be 1 dimensional {cam.shape}"
    W, H = int(cam.size[0]), int(cam.size[1])
    return grid_2d(W, H, output_range=[0, W, 0, H], device=cam.device, dtype=cam.dtype)


def scale_image_to_cam(cams: CameraTW, ims: torch.Tensor) -> torch.Tensor:
    """Scale an image to a camera."""
    T = None
    if ims.ndim == 5:
        B, T, C, H, W = ims.shape
        ims = ims.view(-1, C, H, W)
        Wo, Ho = cams[0, 0].size.int().tolist()
    elif ims.ndim == 4:
        B, C, H, W = ims.shape
        Wo, Ho = cams[0].size.int().tolist()
    else:
        raise ValueError(f"unusable image shape {ims.shape}, {cams.shape}")
    # pyre-fixme[16]: `Optional` has no attribute `Resize`.
    ims = T.Resize(
        (Ho, Wo),
        # pyre-fixme[16]: `Optional` has no attribute `InterpolationMode`.
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True,
    )(ims)
    if T is not None:
        return ims.view(B, T, C, Ho, Wo)
    return ims.view(B, C, Ho, Wo)


def vignette_image(cam="rgb"):
    """Load vignette calibration image for the given camera.

    Expects vignette images in a 'calibration/' directory next to the project root.
    """
    filenames = {
        "rgb": "vignette_imx577.png",
        "slaml": "vignette_ov7251.png",
        "slamr": "vignette_ov7251.png",
    }
    if cam not in filenames:
        raise NotImplementedError(f"{cam} does not have a vignette image")

    import os
    calib_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration"
    )
    local_path = os.path.join(calib_dir, filenames[cam])
    img = read_image(local_path)
    return img


def get_P_vec(P):
    P = P.transpose(-1, -2)
    row = [[] for _ in range(10)]
    row[0] = torch.stack(
        [
            P[..., 0, 0] * P[..., 0, 0],
            P[..., 0, 0] * P[..., 0, 1],
            P[..., 0, 0] * P[..., 0, 2],
            P[..., 0, 1] * P[..., 0, 1],
            P[..., 0, 1] * P[..., 0, 2],
            P[..., 0, 2] * P[..., 0, 2],
        ],
        dim=-1,
    )
    row[1] = torch.stack(
        [
            2 * P[..., 0, 0] * P[..., 1, 0],
            P[..., 0, 0] * P[..., 1, 1] + P[..., 1, 0] * P[..., 0, 1],
            P[..., 0, 0] * P[..., 1, 2] + P[..., 1, 0] * P[..., 0, 2],
            2 * P[..., 0, 1] * P[..., 1, 1],
            P[..., 0, 1] * P[..., 1, 2] + P[..., 1, 1] * P[..., 0, 2],
            2 * P[..., 0, 2] * P[..., 1, 2],
        ],
        dim=-1,
    )
    row[2] = torch.stack(
        [
            2 * P[..., 0, 0] * P[..., 2, 0],
            P[..., 0, 0] * P[..., 2, 1] + P[..., 2, 0] * P[..., 0, 1],
            P[..., 0, 0] * P[..., 2, 2] + P[..., 2, 0] * P[..., 0, 2],
            2 * P[..., 0, 1] * P[..., 2, 1],
            P[..., 0, 1] * P[..., 2, 2] + P[..., 2, 1] * P[..., 0, 2],
            2 * P[..., 0, 2] * P[..., 2, 2],
        ],
        dim=-1,
    )
    row[3] = torch.stack(
        [
            2 * P[..., 0, 0] * P[..., 3, 0],
            P[..., 0, 0] * P[..., 3, 1] + P[..., 3, 0] * P[..., 0, 1],
            P[..., 0, 0] * P[..., 3, 2] + P[..., 3, 0] * P[..., 0, 2],
            2 * P[..., 0, 1] * P[..., 3, 1],
            P[..., 0, 1] * P[..., 3, 2] + P[..., 3, 1] * P[..., 0, 2],
            2 * P[..., 0, 2] * P[..., 3, 2],
        ],
        dim=-1,
    )

    row[4] = torch.stack(
        [
            P[..., 1, 0] * P[..., 1, 0],
            P[..., 1, 0] * P[..., 1, 1],
            P[..., 1, 0] * P[..., 1, 2],
            P[..., 1, 1] * P[..., 1, 1],
            P[..., 1, 1] * P[..., 1, 2],
            P[..., 1, 2] * P[..., 1, 2],
        ],
        dim=-1,
    )

    row[5] = torch.stack(
        [
            2 * P[..., 1, 0] * P[..., 2, 0],
            P[..., 1, 0] * P[..., 2, 1] + P[..., 2, 0] * P[..., 1, 1],
            P[..., 1, 0] * P[..., 2, 2] + P[..., 2, 0] * P[..., 1, 2],
            2 * P[..., 1, 1] * P[..., 2, 1],
            P[..., 1, 1] * P[..., 2, 2] + P[..., 2, 1] * P[..., 1, 2],
            2 * P[..., 1, 2] * P[..., 2, 2],
        ],
        dim=-1,
    )
    row[6] = torch.stack(
        [
            2 * P[..., 1, 0] * P[..., 3, 0],
            P[..., 1, 0] * P[..., 3, 1] + P[..., 3, 0] * P[..., 1, 1],
            P[..., 1, 0] * P[..., 3, 2] + P[..., 3, 0] * P[..., 1, 2],
            2 * P[..., 1, 1] * P[..., 3, 1],
            P[..., 1, 1] * P[..., 3, 2] + P[..., 3, 1] * P[..., 1, 2],
            2 * P[..., 1, 2] * P[..., 3, 2],
        ],
        dim=-1,
    )

    row[7] = torch.stack(
        [
            P[..., 2, 0] * P[..., 2, 0],
            P[..., 2, 0] * P[..., 2, 1],
            P[..., 2, 0] * P[..., 2, 2],
            P[..., 2, 1] * P[..., 2, 1],
            P[..., 2, 1] * P[..., 2, 2],
            P[..., 2, 2] * P[..., 2, 2],
        ],
        dim=-1,
    )
    row[8] = torch.stack(
        [
            2 * P[..., 2, 0] * P[..., 3, 0],
            P[..., 2, 0] * P[..., 3, 1] + P[..., 3, 0] * P[..., 2, 1],
            P[..., 2, 0] * P[..., 3, 2] + P[..., 3, 0] * P[..., 2, 2],
            2 * P[..., 2, 1] * P[..., 3, 1],
            P[..., 2, 1] * P[..., 3, 2] + P[..., 3, 1] * P[..., 2, 2],
            2 * P[..., 2, 2] * P[..., 3, 2],
        ],
        dim=-1,
    )
    row[9] = torch.stack(
        [
            P[..., 3, 0] * P[..., 3, 0],
            P[..., 3, 0] * P[..., 3, 1],
            P[..., 3, 0] * P[..., 3, 2],
            P[..., 3, 1] * P[..., 3, 1],
            P[..., 3, 1] * P[..., 3, 2],
            P[..., 3, 2] * P[..., 3, 2],
        ],
        dim=-1,
    )

    Pv = torch.stack(row, dim=-2)
    return Pv.transpose(-1, -2)


def rectify_video(
    video_snippet: torch.Tensor,
    fisheye_cam: CameraTW,
    pinhole_fxy_factor: Optional[float] = None,
    # pyre-fixme[31]: Expression `(torch.Tensor, utils.camera.CameraTW)`
    #  is not a valid type.
) -> (torch.Tensor, CameraTW):
    """
    Rectify a video snippet.
    Reference: N3192822, N3115192
    Args:
        video_snippet (torch.Tensor): Video snippet to rectify. Expect tensor shape to be BxTxCxHxW.
        fisheye_cam (CameraTW): The Fisheye CameraTW to be rectified.
        pinhole_fxy_factors (float):
            The focal length factors to make the pinhole focal length.
            f_pinhole = f_fisheye / pinhole_fxy_factor.
            If None, default values will be pulled in, see the comments below.
    Return:
        rectified_video, pinhole_camera
    """
    assert video_snippet.ndim == 5 and (
        video_snippet.shape[-3] == 3 or video_snippet.shape[-3] == 1
    )
    # The default values are searched manually to retain the FoV as much as possible. See D43851788 for visual examples.
    if not pinhole_fxy_factor:
        if video_snippet.shape[-3] == 3:  # rgb
            pinhole_fxy_factor = 1.28
        else:  # slam
            pinhole_fxy_factor = 1.45

    fx = fisheye_cam.f[..., 0:1].clone() / pinhole_fxy_factor
    fy = fisheye_cam.f[..., 1:2].clone() / pinhole_fxy_factor
    # centralize principle points.
    cx = fisheye_cam.size.clone()[..., 0:1] / 2.0
    cy = fisheye_cam.size.clone()[..., 1:2] / 2.0
    f_scaled = torch.cat([fx, fy], -1)
    c_center = torch.cat([cx, cy], -1)
    width = fisheye_cam.size[..., 0:1]
    height = fisheye_cam.size[..., 1:2]
    pinhole_cam = CameraTW.from_surreal(
        width=width,
        height=height,
        type_str="pinhole",
        params=torch.cat([f_scaled, c_center], -1),
        gain=fisheye_cam.gain,
        exposure_s=fisheye_cam.exposure_s,
        # will be set to diagonal automatically inside constructor
        valid_radius=fisheye_cam.valid_radius[..., 0] * 10,
        T_camera_rig=fisheye_cam.T_camera_rig,
    )
    # pyre-fixme[6]: For 1st argument expected `Union[bool, float, int]` but got
    #  `Tensor`.
    yy, xx = torch.meshgrid(torch.arange(height[0, 0, 0]), torch.arange(width[0, 0, 0]))
    yy, xx = yy.to(fisheye_cam.device), xx.to(fisheye_cam.device)
    target = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).unsqueeze(0)
    target = target.expand(video_snippet.shape[1], *target.shape[1:]).unsqueeze(0)
    target = target.expand(video_snippet.shape[0], *target.shape[1:])
    rays, valid = pinhole_cam.unproject(target)
    # Note: one could buffer the `source` as LookUpTables in some way to speed up.
    source, valid = fisheye_cam.project(rays)
    source = source.reshape(
        *video_snippet.shape[:2],
        int(height[0, 0, 0].item()),
        int(width[0, 0, 0].item()),
        2,
    )
    source[..., 0] = 2 * (source[..., 0] / width[0, 0, 0]) - 1
    source[..., 1] = 2 * (source[..., 1] / height[0, 0, 0]) - 1
    has_T = video_snippet.ndim == 5
    T = None
    if has_T:
        T = video_snippet.shape[1]
        # reshape source
        video_snippet = video_snippet.flatten(0, 1)
        source = source.flatten(0, 1)
        # reshape img
    video_snippet = torch.nn.functional.grid_sample(
        video_snippet,
        source,
        "bicubic",
        padding_mode="zeros",
        align_corners=False,
    )
    if has_T:
        video_snippet = video_snippet.unflatten(0, (-1, T))

    return video_snippet, pinhole_cam


def normalize(v):
    return v / torch.norm(v, p=2, dim=-1, keepdim=True)


def homogenize(v):
    return torch.cat([v, torch.ones_like(v[..., :1])], -1)


def dehomogenize(v):
    return v[..., :-1] / v[..., -1].unsqueeze(-1)


class DistortionHandler:
    """
    computes a linear camera by matching the field of view of the distorted camera.
    Makes it easy to take a point from the linear camera space to the distorted and vize versa.

    follows https://www.internalfb.com/code/fbsource/arvr/libraries/spatialization/quadric_collider/distortion_handler.cpp
    """

    def __init__(self, cam: CameraTW):
        self.cam = cam  # B x T x cam
        self.B, self.T, _ = self.cam.shape
        self.cam_flattened = cam.view(self.B * self.T, -1)  # B * T x cam
        self.fLin = self.cam_flattened.f[..., 0]  # B * T x 1 (fx==fy)
        hFov = 1.5 * self.cameraVerticalAngleOfView()
        wFov = 1.5 * self.cameraHorizontalAngleOfView()
        # compute size of image, limiting FOV to around 160 deg
        self.w = self.fLin * torch.tan(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `float`.
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
            torch.minimum(torch.ones_like(wFov) * 80.0 * np.pi / 180.0, wFov / 2.0)
        )
        self.h = self.fLin * torch.tan(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `float`.
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
            torch.minimum(torch.ones_like(hFov) * 80.0 * np.pi / 180.0, hFov / 2.0)
        )
        self.cx = self.w / 2
        self.cy = self.h / 2
        # print(self.w, self.h, self.cx, self.cy)

    def cameraVerticalAngleOfView(self):
        w, h = self.cam_flattened.size[:, 0], self.cam_flattened.size[:, 1]
        b1 = normalize(
            self.cam_flattened.unproject(
                torch.stack([0.5 * w, torch.zeros_like(w)], dim=-1).unsqueeze(1)
            )[0]
        ).squeeze(1)
        b2 = normalize(
            self.cam_flattened.unproject(
                torch.stack([0.5 * w, h], dim=-1).unsqueeze(1)
            )[0]
        ).squeeze(1)
        return torch.acos(torch.einsum("bi,bi->b", b1, b2))

    def cameraHorizontalAngleOfView(self):
        w, h = self.cam_flattened.size[:, 0], self.cam_flattened.size[:, 1]
        b1 = normalize(
            self.cam_flattened.unproject(
                torch.stack([w, 0.5 * h], dim=-1).unsqueeze(1)
            )[0]
        ).squeeze(1)
        b2 = normalize(
            self.cam_flattened.unproject(
                torch.stack([torch.zeros_like(h), 0.5 * h], dim=-1).unsqueeze(1)
            )[0]
        ).squeeze(1)
        return torch.acos(torch.einsum("bi,bi->b", b1, b2))

    @property
    def K(self):
        return self.get_K(flattened=False)

    @property
    def K_flattened(self):
        return self.get_K(flattened=True)

    def get_K(self, flattened=False):
        K = torch.stack(
            [
                torch.stack([self.fLin, torch.zeros_like(self.fLin), self.cx], dim=-1),
                torch.stack([torch.zeros_like(self.fLin), self.fLin, self.cy], dim=-1),
                torch.stack(
                    [
                        torch.zeros_like(self.fLin),
                        torch.zeros_like(self.fLin),
                        torch.ones_like(self.fLin),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        return K if flattened else K.view(self.B, self.T, 3, 3)

    # helpers that should be folded into CameraTW as a "linear" camera model
    def linUnproject(self, p2s):
        return homogenize(p2s) @ torch.linalg.inv(self.K_flattened).transpose(-1, -2)

    def linProject(self, p3s):
        return dehomogenize(p3s @ self.K_flattened.transpose(-1, -2))

    def distort(self, p2s):
        """
        compute a distorted point from a linearized point
        """
        assert p2s.size(0) == self.B and p2s.size(1) == self.T
        p2s = p2s.flatten(0, 1)
        res = self.cam_flattened.project(self.linUnproject(p2s))
        return res[0].unflatten(0, (self.B, self.T)), res[1].unflatten(0, (self.B, self.T))

    def linearize(self, p2s):
        """
        compute a linearized point from a distorted point
        """
        assert p2s.size(0) == self.B and p2s.size(1) == self.T
        p2s = p2s.flatten(0, 1)
        return self.linProject(self.cam_flattened.unproject(p2s)[0]).unflatten(0, (self.B, self.T))


# ---------------------------------------------------------------------------
# Projection utilities (fisheye624, pinhole, kb4)
# ---------------------------------------------------------------------------

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def maybe_jit_script(fn):
    """Apply @torch.jit.script only on PyTorch >= 1.13."""
    if _TORCH_VERSION >= (1, 13):
        return torch.jit.script(fn)
    return fn


def sign_plus(x):
    """
    return +1 for positive and for 0.0 in x. This is important for our handling
    of z values that should never be 0.0
    """
    return 2.0 * (x >= 0.0).to(x.dtype) - 1.0


def _fisheye624_project_impl(xyz, params, suppress_warning=False):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model project() function.

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        params: Bx(T)x16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx(T)x15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (
        xyz.ndim == 4 and params.ndim == 3
    ), f"point dim {xyz.shape} does not match cam parameter dim {params}"
    assert xyz.shape[-1] == 3
    assert params.shape[-1] == 16 or params.shape[-1] == 15, (
        "This model allows fx != fy"
    )

    # Warn if input magnitudes are large enough to cause float32 precision issues
    if xyz.dtype == torch.float32 and xyz.numel() > 0 and not suppress_warning:
        max_abs = xyz.abs().max().item()
        if max_abs > 1000:
            import traceback
            import warnings

            stack = "".join(traceback.format_stack())
            warnings.warn(
                f"fisheye624_project: large input magnitude ({max_abs:.0f}) may cause "
                f"float32 precision loss. Consider centering points before projection.\n"
                f"Called from:\n{stack}",
                stacklevel=3,
            )

    eps = 1e-9
    T = -1
    if xyz.ndim == 4:
        T, N = xyz.shape[1], xyz.shape[2]
        xyz = xyz.reshape(-1, N, 3)
        params = params.reshape(-1, params.shape[-1])

    B, N = xyz.shape[0], xyz.shape[1]

    # Radial correction.
    z = xyz[:, :, 2].reshape(B, N, 1)
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    ab = xyz[:, :, :2] / z
    ab = torch.where(torch.abs(ab) < eps, eps * sign_plus(ab), ab)
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    th = torch.atan(r)
    ones_ab = torch.zeros_like(ab) + 1.0
    th_divr = torch.where(r < eps, ones_ab, ab / r)
    th_k = th.reshape(B, N, 1).clone()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * torch.pow(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    # Tangential correction.
    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = torch.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)

    # Thin Prism correction.
    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = torch.square(rd_sq)
    uv_dist_tp0 = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist_tp1 = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
    uv_dist = torch.stack([uv_dist_tp0, uv_dist_tp1], dim=-1)

    # Finally, apply standard terms: focal length and camera centers.
    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    result = uv_dist * fx_fy + cx_cy

    if T > 0:
        result = result.reshape(B // T, T, N, 2)

    assert result.ndim == 4 or result.ndim == 3
    assert result.shape[-1] == 2

    return result


def fisheye624_project(xyz, params, suppress_warning=False):
    """Public interface for fisheye624 projection."""
    return _fisheye624_project_impl(xyz, params, suppress_warning=suppress_warning)


@maybe_jit_script
def fisheye624_unproject(uv, params, max_iters: int = 5):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model unproject function using Newton's method.

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels to be unprojected
        params: Bx(T)x16 or Bx(T)x15 tensor of Fisheye624 parameters
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays of uv points with z = 1.
    """
    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx2"
    assert uv.shape[-1] == 2
    assert params.ndim == 2 or params.ndim == 3
    assert params.shape[-1] == 16 or params.shape[-1] == 15
    eps = 1e-6

    T = -1
    if uv.ndim == 4:
        T, N = uv.shape[1], uv.shape[2]
        uv = uv.reshape(-1, N, 2)
        params = params.reshape(-1, params.shape[-1])

    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    # Compute xr_yr using Newton's method.
    xr_yr = uv_dist.clone()
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (
            (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        )
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (
            (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
        )
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = (
            1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        )
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = (
            1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        )
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (
            xr_yr[:, :, 0] * temp1
        )
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (
            xr_yr[:, :, 1] * temp1
        )
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (
            xr_yr[:, :, 0] * temp2
        )
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (
            xr_yr[:, :, 1] * temp2
        )
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        xr_yr = xr_yr + step

    # Compute theta using Newton's method.
    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        step = torch.where(dthd_th.abs() > eps, step, sign_plus(step) * eps * 10.0)
        th = th + step
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    assert ray.shape[-1] == 3

    if T > 0:
        ray = ray.reshape(B // T, T, N, 3)

    return ray


def pinhole_project(xyz, params):
    """
    Batched pinhole camera projection.

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points
        params: Bx(T)x4 tensor [f_u f_v c_u c_v]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections
    """
    assert (xyz.ndim == 3 and params.ndim == 2) or (xyz.ndim == 4 and params.ndim == 3)
    assert params.shape[-1] == 4
    eps = 1e-9
    fx_fy = params[..., 0:2].reshape(*xyz.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*xyz.shape[:-2], 1, 2)
    z = xyz[..., 2:]
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    uv = (xyz[..., :2] / z) * fx_fy + cx_cy
    return uv


def pinhole_unproject(uv, params, max_iters: int = 5):
    """
    Batched pinhole camera unprojection.

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels
        params: Bx(T)x4 tensor [f_u f_v c_u c_v]
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays with z = 1.
    """
    assert uv.ndim == 3 or uv.ndim == 4
    assert params.ndim == 2 or params.ndim == 3
    assert params.shape[-1] == 4
    assert uv.shape[-1] == 2
    fx_fy = params[..., 0:2].reshape(*uv.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*uv.shape[:-2], 1, 2)
    uv_dist = (uv - cx_cy) / fx_fy
    ray = torch.cat([uv_dist, uv.new_ones(*uv.shape[:-1], 1)], dim=-1)
    return ray


def brown_conrady_project(xyz, params):
    """
    Batched Brown Conrady radial camera projection.

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points
        params: Bx(T)x12 tensor [f_u f_v c_u c_v {k_0..k_3} {p_0..p_3}]
    """
    assert (xyz.ndim == 3 and params.ndim == 2) or (xyz.ndim == 4 and params.ndim == 3)
    assert params.shape[-1] == 4
    eps = 1e-9
    fx_fy = params[..., 0:2].reshape(*xyz.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*xyz.shape[:-2], 1, 2)
    B, N = xyz.shape[0], xyz.shape[1]
    z = xyz[..., 2:]
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    ab = xyz[:, :, :2] / z
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    r_k = torch.ones_like(r)
    for i in range(4):
        pow = 2 + i * 2
        r_k = r_k + params[:, 4 + i].reshape(B, 1, 1) * torch.pow(r, pow)
    print("===> warning untested work in progress!!!")
    uv_dist = xyz[..., :2] / z
    uv = (uv_dist * r_k) * fx_fy + cx_cy
    return uv


def kb4_project(
    rays: torch.Tensor,
    params: torch.Tensor,
    z_clip: float = 1e-8,
) -> torch.Tensor:
    """
    Batched KB4 projection (ray -> pixel).

    Args:
        rays:   (Bx(T)xNx3) camera-frame rays
        params: (Bx(T)x8) [fu,fv,u0,v0,k0,k1,k2,k3]
        z_clip: clamp Z to avoid divide-by-zero
    Returns:
        uv: (B,N,2) pixel coordinates
    """
    print(f"==> warning kb4 project, work in progress!!!")
    assert rays.ndim == 3 or rays.ndim == 4
    assert rays.size(-1) == 3
    assert params.ndim == 2 or params.ndim == 3
    assert params.size(-1) == 8
    fu, fv, u0, v0, k0, k1, k2, k3 = params.unbind(-1)
    X, Y, Z = rays.unbind(-1)
    Z = Z.clamp_min(z_clip)
    xn, yn = X / Z, Y / Z
    rho = torch.sqrt(xn * xn + yn * yn).clamp_min(1e-12)
    theta = torch.atan(rho)
    t2 = theta * theta
    r = (
        k0[..., None] * theta
        + k1[..., None] * theta * t2
        + k2[..., None] * theta * t2 * t2
        + k3[..., None] * theta * t2 * t2 * t2
    )
    scale = r / rho
    xd, yd = xn * scale, yn * scale
    u = u0[..., None] + fu[..., None] * xd
    v = v0[..., None] + fv[..., None] * yd
    return torch.stack([u, v], dim=-1)


def kb4_unproject(
    uv: torch.Tensor,
    params: torch.Tensor,
    iters: int = 5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Batched unprojection for Kannala-Brandt KB4 camera model.

    Args:
        uv:     (Bx(T)xNx2) pixel coordinates
        params: (Bx(T)x8) [fu,fv,u0,v0,k0,k1,k2,k3]
        iters:  Newton iterations for solving theta
    Returns:
        rays:  (B,N,3) unit-length rays in camera coordinates
    """
    print(f"==> warning kb4 unproject, work in progress!!!")
    assert uv.ndim == 3 or uv.ndim == 4
    assert uv.size(-1) == 2
    assert params.ndim == 2 or params.ndim == 3
    assert params.size(-1) == 8
    fu, fv, u0, v0, k0, k1, k2, k3 = params.unbind(-1)
    u, v = uv[..., 0], uv[..., 1]
    xd = (u - u0[..., None]) / fu[..., None]
    yd = (v - v0[..., None]) / fv[..., None]
    rd = torch.sqrt(xd * xd + yd * yd).clamp_min(eps)
    theta = torch.atan(rd)
    for ii in range(iters):
        t2 = theta * theta
        r_theta = (
            k0[..., None] * theta
            + k1[..., None] * theta * t2
            + k2[..., None] * theta * t2 * t2
            + k3[..., None] * theta * t2 * t2 * t2
        )
        f = r_theta - rd
        dr_dtheta = (
            k0[..., None]
            + 3 * k1[..., None] * t2
            + 5 * k2[..., None] * t2 * t2
            + 7 * k3[..., None] * t2 * t2 * t2
        )
        theta = theta - (f / dr_dtheta)
    rho = torch.tan(theta)
    dirx = xd / (rd * rho + eps)
    diry = yd / (rd * rho + eps)
    rays = torch.stack([dirx, diry, torch.ones_like(dirx)], dim=-1)
    denom = torch.linalg.norm(rays, dim=-1, keepdim=True)
    rays = rays / denom
    return rays
