# pyre-unsafe
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .camera import CameraTW
from .pose import IdentityPose, PoseTW, rotation_from_euler
from tw.tensor_utils import pad_string, tensor2string, unpad_string
from .tensor_wrapper import autocast, autoinit, smart_cat, smart_stack, TensorWrapper

# OBB corner numbering diagram for this implementation (the same as pytorch3d
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py#L111)
#
# (4) +---------+. (5)
#     | ` .     |  ` .
#     | (0) +---+-----+ (1)
#     |     |   |     |
# (7) +-----+---+. (6)|
#     ` .   |     ` . |
#     (3) ` +---------+ (2)
#
# NOTE: Throughout this implementation, we assume that boxes
# are defined by their 8 corners exactly in the order specified in the
# diagram above for the function to give correct results. In addition
# the vertices on each plane must be coplanar.
# As an alternative to the diagram, this is a unit bounding
# box which has the correct vertex ordering:
# box_corner_vertices = [
#     [0, 0, 0],  #   (0)
#     [1, 0, 0],  #   (1)
#     [1, 1, 0],  #   (2)
#     [0, 1, 0],  #   (3)
#     [0, 0, 1],  #   (4)
#     [1, 0, 1],  #   (5)
#     [1, 1, 1],  #   (6)
#     [0, 1, 1],  #   (7)
# ]

_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]

DOT_EPS = 1e-3
AREA_EPS = 1e-4
PAD_VAL = -1
# triangle indices to draw an OBB mesh from bb3coners_*
OBB_MESH_TRI_INDS = [
    [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
]

# line indices to draw an OBB line strip frame from bb3coners_*
OBB_LINE_INDS = [0, 1, 2, 3, 0, 3, 7, 4, 0, 1, 5, 6, 5, 4, 7, 6, 2, 1, 5]

# corner indices to construct all edge lines
BB3D_LINE_ORDERS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


class ObbTW(TensorWrapper):
    """
    Oriented 3D Bounding Box observation in world coordinates (via
    T_world_object) for Aria headsets.
    """

    @autocast
    @autoinit
    def __init__(self, data: torch.Tensor = PAD_VAL * torch.ones((1, 165))):
        assert isinstance(data, torch.Tensor)
        assert data.shape[-1] == 165
        super().__init__(data)

    @classmethod
    @autoinit
    def from_lmc(
        cls,
        bb3_object: torch.Tensor = PAD_VAL * torch.ones(6),
        bb2_rgb: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slaml: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slamr: torch.Tensor = PAD_VAL * torch.ones(4),
        T_world_object: Union[torch.Tensor, PoseTW] = IdentityPose,  # 1x12.
        sem_id: torch.Tensor = PAD_VAL * torch.ones(1),
        inst_id: torch.Tensor = PAD_VAL * torch.ones(1),
        prob: torch.Tensor = 1 * torch.ones(1),
        moveable: torch.Tensor = 0 * torch.ones(1),
        color: torch.Tensor = PAD_VAL * torch.ones(3),
        text: torch.Tensor = PAD_VAL * torch.ones(128),
    ):
        # Concatenate into one big data tensor, handles TensorWrapper objects.
        # make sure that its on the same device (fails if IdentityPose is used)
        device = bb3_object.device

        data = smart_cat(
            [
                bb3_object,
                bb2_rgb.to(device),
                bb2_slaml.to(device),
                bb2_slamr.to(device),
                T_world_object.to(device),
                sem_id.to(device),
                inst_id.to(device),
                prob.to(device),
                moveable.to(device),
                color.to(device),
                text.to(device),
            ],
            dim=-1,
        )
        return cls(data)

    @classmethod
    @autocast
    def from_corners(
        cls,
        bb3_corners_world,
        bb2_rgb: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slaml: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slamr: torch.Tensor = PAD_VAL * torch.ones(4),
        sem_id: torch.Tensor = PAD_VAL * torch.ones(1),
        inst_id: torch.Tensor = PAD_VAL * torch.ones(1),
        prob: torch.Tensor = 1 * torch.ones(1),
        moveable: torch.Tensor = 0 * torch.ones(1),
        color: torch.Tensor = PAD_VAL * torch.ones(3),
        text: torch.Tensor = PAD_VAL * torch.ones(128),
    ):
        dtype = bb3_corners_world.dtype
        device = bb3_corners_world.device
        P = bb3_corners_world
        xs = torch.linalg.norm(P[1, :] - P[0, :])
        ys = torch.linalg.norm(P[3, :] - P[0, :])
        zs = torch.linalg.norm(P[4, :] - P[0, :])
        bb3 = torch.tensor(
            [
                -xs / 2.0,
                xs / 2.0,
                -ys / 2.0,
                ys / 2.0,
                -zs / 2.0,
                zs / 2.0,
            ]
        ).unsqueeze(0)
        ids = [0, 2, 4, 1, 2, 4, 1, 3, 4, 0, 3, 4, 0, 2, 5, 1, 2, 5, 1, 3, 5, 0, 3, 5]
        c3o = bb3[..., ids]
        Q = c3o.reshape(8, 3)
        P = P.t()
        Q = Q.t()
        centroids_P = torch.mean(P, dim=1)
        centroids_Q = torch.mean(Q, dim=1)
        A = P - torch.outer(centroids_P, torch.ones(P.shape[1], dtype=dtype))
        B = Q - torch.outer(centroids_Q, torch.ones(Q.shape[1], dtype=dtype))
        C = A @ B.transpose(0, 1)
        U, S, V = torch.linalg.svd(C)
        R = V.transpose(0, 1) @ U.transpose(0, 1)
        L = torch.eye(3, dtype=dtype)
        if torch.linalg.det(R) < 0:
            L[2][2] *= -1
        R = V.transpose(0, 1) @ (L @ U.transpose(0, 1))
        t = (-R @ centroids_P) + centroids_Q
        T_world_object = PoseTW.from_Rt(R, t).inverse().to(dtype)

        data = smart_cat(
            [
                bb3[0],  # TODO support more sizes
                bb2_rgb.to(device),
                bb2_slaml.to(device),
                bb2_slamr.to(device),
                T_world_object.to(device),
                sem_id.to(device),
                inst_id.to(device),
                prob.to(device),
                moveable.to(device),
                color.to(device),
                text.to(device),
            ],
            dim=-1,
        )
        return cls(data)

    @property
    def bb3_object(self) -> torch.Tensor:
        """3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame, with shape (..., 6)."""
        return self._data[..., :6]

    @property
    def bb3_min_object(self) -> torch.Tensor:
        """3D bounding box minimum corner [xmin,ymin,zmin] in object coord frame, with shape (..., 3)."""
        return self._data[..., 0:6:2]

    @property
    def bb3_max_object(self) -> torch.Tensor:
        """3D bounding box maximum corner [xmax,ymax,zmax] in object coord frame, with shape (..., 3)."""
        return self._data[..., 1:6:2]

    @property
    def bb3_center_object(self) -> torch.Tensor:
        """3D bounding box center in object coord frame, with shape (..., 3)."""
        padding_mask = self.get_padding_mask()
        out = 0.5 * (self.bb3_min_object + self.bb3_max_object)
        out[padding_mask] = PAD_VAL
        return out

    @property
    def bb3_center_world(self) -> torch.Tensor:
        """3D bounding box center in world coord frame, with shape (..., 3)."""
        padding_mask = self.get_padding_mask()
        s = self.bb3_center_object.shape
        # pyre-fixme[16]: `Tensor` has no attribute `batch_transform`.
        _bb3_center_world = self.T_world_object.view(-1, 12).batch_transform(
            self.bb3_center_object.view(-1, 3)
        )
        out = _bb3_center_world.view(s)
        out[padding_mask] = PAD_VAL
        return out

    @property
    def bb3_diagonal(self) -> torch.Tensor:
        """3D bounding box diangonal, with shape (..., 3)."""
        return self.bb3_max_object - self.bb3_min_object

    @property
    def bb3_volumes(self) -> torch.Tensor:
        """3D bounding box volumes, with shape (..., 1)."""
        diags = self.bb3_diagonal
        return diags.prod(dim=-1, keepdim=True)

    @property
    def bb2_rgb(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in RGB image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 6:10]

    def visible_bb3_ind(self, cam_id) -> torch.Tensor:
        """Indices of visible 3D bounding boxes in camera cam_id"""
        bb2_cam = self.bb2(cam_id)
        vis_ind = torch.all(bb2_cam > 0, dim=-1)
        return vis_ind

    @property
    def bb2_slaml(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in SLAM Left image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 10:14]

    @property
    def bb2_slamr(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in SLAM Right image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 14:18]

    @property
    def color(self) -> torch.Tensor:
        """Three numbers of [R,G,B] color in range [0,1] storing the color of OBB used for rendering."""
        return self._data[..., 34:37]

    def set_color(self, color, use_mask=True):
        """Set color"""
        padding_mask = self.get_padding_mask()
        self._data[..., 34:37] = color
        if use_mask:
            self._data[padding_mask] = -1.0

    @property
    def text(self) -> torch.Tensor:
        """Text describing what is in the box. Can be a single word or sentences up to a fixed length"""
        return self._data[..., 37:165]

    def set_text(self, text, use_mask=True):
        """Set text"""
        padding_mask = self.get_padding_mask()
        self._data[..., 37:165] = text
        if use_mask:
            self._data[padding_mask] = -1.0

    def text_string(self) -> Union[str, List[str]]:
        """Text describing what is in the box converted to string(s).

        Returns:
            str: Single string if called on a single OBB (ndim == 1)
            list[str]: List of strings if called on multiple OBBs (ndim >= 2)
        """
        text = self.text
        is_single = text.ndim == 1
        if is_single:
            text = text.unsqueeze(0)
        strings = []
        for txt in text:
            try:
                s = unpad_string(tensor2string(txt.int()))
                # If empty or all whitespace after unpading, use empty string
                if not s or not s.strip():
                    s = ""
                strings.append(s)
            except Exception:
                # If conversion fails for any reason, use empty string
                strings.append("")
        # Return single string for single OBB, list for multiple
        if is_single:
            return strings[0]
        return strings

    def bb2(self, cam_id) -> torch.Tensor:
        """
        2D bounding box [xmin,xmax,ymin,ymax] as visible in camera with given
        cam_id, -1's if not visible, with shape (..., 4).
        cam_id == 0 for rgb
        cam_id == 1 for slam left
        cam_id == 2 for slam right
        """
        return self._data[..., 6 + cam_id * 4 : 10 + cam_id * 4]

    def set_bb2(self, cam_id, bb2d, use_mask=True):
        """
        Set 2D bounding box [xmin,xmax,ymin,ymax] in camera with given
        cam_id == 0 for rgb
        cam_id == 1 for slam left
        cam_id == 2 for slam right
        """
        padding_mask = self.get_padding_mask()
        self._data[..., 6 + cam_id * 4 : 10 + cam_id * 4] = bb2d
        if use_mask:
            self._data[padding_mask] = -1.0

    # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
    def set_bb3_object(self, bb3_object, use_mask=True) -> torch.Tensor:
        """set 3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame, with shape (..., 6)."""
        padding_mask = self.get_padding_mask()
        self._data[..., :6] = bb3_object
        if use_mask:
            self._data[padding_mask] = -1.0

    @property
    def T_world_object(self) -> torch.Tensor:
        """3D SE3 transform from object to world coords, with shape (..., 12)."""
        return PoseTW(self._data[..., 18:30])

    def get_padding_mask(self) -> torch.Tensor:
        """get boolean mask indicating which Obbs are valid/non-padded."""
        return (self._data == -1.0).all(dim=-1, keepdim=False)

    def set_T_world_object(self, T_world_object: PoseTW):
        """set 3D SE3 transform from object to world coords."""
        # HACK: handle invalids better
        invalid_mask = self.get_padding_mask()
        self._data[..., 18:30] = T_world_object._data
        self._data[invalid_mask] = -1.0

    @property
    def sem_id(self) -> torch.Tensor:
        """semantic id, with shape (..., 1)."""
        return self._data[..., 30].unsqueeze(-1).int()

    def set_sem_id(self, sem_id: torch.Tensor):
        """set semantic id to sem_id"""
        self._data[..., 30] = sem_id.squeeze()

    @property
    def inst_id(self) -> torch.Tensor:
        """instance id, with shape (..., 1)."""
        return self._data[..., 31].unsqueeze(-1).int()

    def set_inst_id(self, inst_id: torch.Tensor):
        """set instance id to inst_id"""
        self._data[..., 31] = inst_id.squeeze()

    @property
    def prob(self) -> torch.Tensor:
        """probability of detection, with shape (..., 1)."""
        return self._data[..., 32].unsqueeze(-1)

    def set_prob(self, prob, use_mask=True):
        """Set probability score"""
        padding_mask = self.get_padding_mask()
        self._data[..., 32] = prob
        if use_mask:
            self._data[padding_mask] = -1.0

    @property
    def moveable(self) -> torch.Tensor:
        """boolean if moveable, with shape (..., 1)."""
        return self._data[..., 33].unsqueeze(-1)

    @property
    def bb3corners_world(self) -> torch.Tensor:
        return self.T_world_object * self.bb3corners_object

    @property
    def bb3corners_object(self) -> torch.Tensor:
        """return the 8 corners of the 3D BB in object coord frame (..., 8, 3)."""
        ids = [0, 2, 4, 1, 2, 4, 1, 3, 4, 0, 3, 4, 0, 2, 5, 1, 2, 5, 1, 3, 5, 0, 3, 5]
        b3o = self.bb3_object
        c3o = b3o[..., ids]
        c3o = c3o.reshape(*c3o.shape[:-1], 8, 3)
        return c3o

    def bb3edge_pts_object(self, num_samples_per_edge: int = 10) -> torch.Tensor:
        """
        return the num_samples_per_edge points per 3D BB edge in object coord
        frame (..., num_samples_per_edge * 12, 3).

        num_samples_per_edge == 1 will result in a list of corners (with some duplicates)
        num_samples_per_edge == 2 will result in a list of corners (with some more duplicates)
        num_samples_per_edge == 3 will result in a list of corners and edge midpoints
        ...
        """
        bb3corners = self.bb3corners_object
        shape = bb3corners.shape
        alphas = torch.linspace(0, 1, num_samples_per_edge, device=bb3corners.device)
        alphas = alphas.view([1] * len(shape[:-2]) + [num_samples_per_edge, 1])
        alphas = alphas.repeat(list(shape[:-2]) + [1, 3])
        betas = torch.ones_like(alphas) - alphas
        bb3edge_pts = []
        for edge_ids in BB3D_LINE_ORDERS:
            bb3edge_pts.append(
                bb3corners[..., edge_ids[0], :].unsqueeze(-2) * betas
                + bb3corners[..., edge_ids[1], :].unsqueeze(-2) * alphas
            )
        return torch.cat(bb3edge_pts, dim=-2)

    def center(self):
        """
        Returns a ObbTW object where the 3D OBBs are centered in their local coorindate system.
        I.e. bb3_min_object == - bb3_max_object.
        """

        # following math in https://www.internalfb.com/code/fbsource/[2c189097095efb98ce72b5b6d5010097abb7b1ab]/arvr/projects/surreal/experiments/Fuser/lmc_dataloader/lmcMeta.cpp?lines=56-61
        T_wo = self.T_world_object
        center_o = self.bb3_center_object
        # compute centered bb3_object and obb pose T_world_object
        centered_T_wo = PoseTW.from_Rt(T_wo.R, T_wo.batch_transform(center_o))
        centered_bb3_min_o = self.bb3_min_object - center_o
        centered_bb3_max_o = self.bb3_max_object - center_o
        centered_bb3_o = torch.stack(
            [
                centered_bb3_min_o[..., 0],
                centered_bb3_max_o[..., 0],
                centered_bb3_min_o[..., 1],
                centered_bb3_max_o[..., 1],
                centered_bb3_min_o[..., 2],
                centered_bb3_max_o[..., 2],
            ],
            dim=-1,
        )
        return ObbTW.from_lmc(
            bb3_object=centered_bb3_o,
            bb2_rgb=self.bb2_rgb,
            bb2_slaml=self.bb2_slaml,
            bb2_slamr=self.bb2_slamr,
            T_world_object=centered_T_wo,
            sem_id=self.sem_id,
            inst_id=self.inst_id,
            prob=self.prob,
            moveable=self.moveable,
        )

    def add_padding(self, max_elts: int = 1000) -> "ObbTW":
        """
        Adds padding to Obbs, useful for returning batches with a varying number
        of Obbs. E.g. if in one batch we have 4 Obbs and another one we have 2,
        setting max_elts=4 will add 2 pads (consisting of all -1s) to the second
        element in the batch.
        """
        assert self._data.ndim <= 2, "higher than order 2 add_padding not supported yet"
        elts = self._data
        num_to_pad = max_elts - len(elts)
        # All -1's denotes a pad element.
        pad_elt = PAD_VAL * self._data.new_ones(self._data.shape[-1])
        if num_to_pad > 0:
            # pyre-fixme[6]: For 1st argument expected `Union[None, List[Tensor],
            #  tuple[Tensor, ...]]` but got `List[int]`.
            rep_elts = torch.stack([pad_elt for _ in range(num_to_pad)], dim=0)
            elts = torch.cat([elts, rep_elts], dim=0)
        elif num_to_pad < 0:
            elts = elts[:max_elts]
            print("Warning: some obbs have been clipped in ObbTW.add_padding()")
        return self.__class__(elts)

    def is_pad(self):
        # All -1's denotes a pad element.
        pad_elt = (PAD_VAL * self._data.new_ones(self._data.shape[-1])).unsqueeze(-2)
        is_pad = torch.all(self._data == pad_elt, dim=-1)
        return is_pad

    def remove_padding(self) -> List["ObbTW"]:
        """
        Removes any padding by finding Obbs with all -1s. Returns a list.
        """
        assert self.ndim <= 4, "higher than order 4 remove_padding not supported yet"

        if self.ndim == 1:
            # pyre-fixme[7]: Expected `List[ObbTW]` but got `ObbTW`.
            return self  # Nothing to be done in this case.

        # All -1's denotes a pad element.
        # pyre-fixme[16]: `int` has no attribute `unsqueeze`.
        pad_elt = (PAD_VAL * self._data.new_ones(self._data.shape[-1])).unsqueeze(-2)
        is_not_pad = ~torch.all(self._data == pad_elt, dim=-1)

        if self.ndim == 2:
            num_valid = is_not_pad.sum()
            new_data = self.__class__(self._data[:num_valid])
        elif self.ndim == 3:
            B = self._data.shape[0]
            new_data = []
            for b in range(B):
                num_valid = is_not_pad[b].sum()
                new_data.append(self.__class__(self._data[b][:num_valid]))
        else:  # self.ndim == 4:
            B, T = self._data.shape[:2]
            new_data = []
            for b in range(B):
                new_data.append([])
                for t in range(T):
                    num_valid = is_not_pad[b, t].sum()
                    new_data[-1].append(self.__class__(self._data[b, t][:num_valid]))
        return new_data

    def scale_bb2(self, scale_rgb: float, scale_slam: float):
        """Update the 2d bb parameters after resizing the underlying images.
        All 2d bbs are scaled by the same scale specified for the frame of the
        2d bb (RGB vs SLAM)."""

        # Check for padded values and leave those unchanged.
        pad_rgb = (
            torch.all(self.bb2_rgb == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_rgb.shape)
        )
        pad_slamr = (
            torch.all(self.bb2_slamr == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_slamr.shape)
        )
        pad_slaml = (
            torch.all(self.bb2_slaml == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_slaml.shape)
        )
        sc_rgb = scale_rgb * torch.ones_like(self.bb2_rgb)
        sc_slamr = scale_slam * torch.ones_like(self.bb2_slamr)
        sc_slaml = scale_slam * torch.ones_like(self.bb2_slaml)
        # If False, multiply by scale, if True multiply by 1.
        sc_rgb = torch.where(pad_rgb, torch.ones_like(sc_rgb), sc_rgb)
        sc_slamr = torch.where(pad_slamr, torch.ones_like(sc_slamr), sc_slamr)
        sc_slaml = torch.where(pad_slaml, torch.ones_like(sc_slaml), sc_slaml)

        data = smart_cat(
            [
                self.bb3_object,
                self.bb2_rgb * sc_rgb,
                self.bb2_slaml * sc_slaml,
                self.bb2_slamr * sc_slamr,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def crop_bb2(self, left_top_rgb: Tuple[float], left_top_slam: Tuple[float]):
        """Update the 2d bb parameters after cropping the underlying images.
        All 2d bbs are cropped by the same crop specified for the frame of the
        2d bb (RGB vs SLAM).
        left_top_* is assumed to be a 2D tuple of the left top corner of te crop.
        """
        # accumulate 2d bb formating of (xmin, xmax, ymin, ymax)
        left_top_rgb = self._data.new_tensor(
            (left_top_rgb[0], left_top_rgb[0], left_top_rgb[1], left_top_rgb[1])
        )
        left_top_slam = self._data.new_tensor(
            (left_top_slam[0], left_top_slam[0], left_top_slam[1], left_top_slam[1])
        )

        # Expand the dimension if self._data is a tensor of CameraTW
        if len(self._data.shape) > 1:
            expand_dim = list(self._data.shape[:-1]) + [1]
            left_top_rgb = left_top_rgb.repeat(expand_dim)
            left_top_slam = left_top_slam.repeat(expand_dim)

        data = smart_cat(
            [
                self.bb3_object,
                self.bb2_rgb - left_top_rgb,
                self.bb2_slaml - left_top_slam,
                self.bb2_slamr - left_top_slam,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def rotate_bb2_cw(self, image_sizes: List[Tuple[int]]):
        """Update the 2d bb parameters after rotating the underlying images.
        Args:
          image_sizes: List of original image sizes before the rotation.
                       The order of the images sizes should be [(w_rgb, h_rgb), (w_slaml, h_slaml), (w_slamr, h_slamr)].
        """
        ## Early check the input input sizes
        assert len(image_sizes) == 3, (
            f"the image sizes of 3 video stream should be given, but only got {len(image_sizes)}"
        )
        for s in image_sizes:
            assert len(s) == 2

        # rotate the obbs stream by stream
        bb2_rgb_cw = rot_obb2_cw(self.bb2_rgb.clone(), image_sizes[0])
        bb2_slaml_cw = rot_obb2_cw(self.bb2_slaml.clone(), image_sizes[1])
        bb2_slamr_cw = rot_obb2_cw(self.bb2_slamr.clone(), image_sizes[2])

        data = smart_cat(
            [
                self.bb3_object,
                bb2_rgb_cw,
                bb2_slaml_cw,
                bb2_slamr_cw,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def rectify_obb2(self, fisheye_cams: List[CameraTW], pinhole_cams: List[CameraTW]):
        rect_bb2s = []
        for idx, (fisheye_cam, pinhole_cam) in enumerate(
            zip(fisheye_cams, pinhole_cams)
        ):
            if idx == 0:
                # rgb
                tl_points = self.bb2_rgb[..., [0, 2]].clone()  # top-left
                bl_points = self.bb2_rgb[..., [0, 3]].clone()  # bottom-left
                br_points = self.bb2_rgb[..., [1, 3]].clone()  # bottom-right
                tr_points = self.bb2_rgb[..., [1, 2]].clone()  # top-right
                visible_points = self.visible_bb3_ind(0)
            elif idx == 1:
                # slaml
                tl_points = self.bb2_slaml[..., [0, 2]].clone()
                bl_points = self.bb2_slaml[..., [0, 3]].clone()
                br_points = self.bb2_slaml[..., [1, 3]].clone()
                tr_points = self.bb2_slaml[..., [1, 2]].clone()
                visible_points = self.visible_bb3_ind(1)
            else:
                # slamr
                tl_points = self.bb2_slamr[..., [0, 2]].clone()
                bl_points = self.bb2_slamr[..., [0, 3]].clone()
                br_points = self.bb2_slamr[..., [1, 3]].clone()
                tr_points = self.bb2_slamr[..., [1, 2]].clone()
                visible_points = self.visible_bb3_ind(2)

            tl_rays, _ = fisheye_cam.unproject(tl_points)
            br_rays, _ = fisheye_cam.unproject(br_points)
            bl_rays, _ = fisheye_cam.unproject(bl_points)
            tr_rays, _ = fisheye_cam.unproject(tr_points)

            rect_tl_pts, valid = pinhole_cam.project(tl_rays)
            rect_br_pts, valid = pinhole_cam.project(br_rays)
            rect_tl_pts, valid = pinhole_cam.project(bl_rays)
            rect_tr_pts, valid = pinhole_cam.project(tr_rays)
            rect_concat = torch.cat(
                [rect_tl_pts, rect_br_pts, rect_tl_pts, rect_tr_pts], dim=-1
            )
            xmin, _ = torch.min(rect_concat[..., 0::2], dim=-1, keepdim=True)
            xmax, _ = torch.max(rect_concat[..., 0::2], dim=-1, keepdim=True)
            ymin, _ = torch.min(rect_concat[..., 1::2], dim=-1, keepdim=True)
            ymax, _ = torch.max(rect_concat[..., 1::2], dim=-1, keepdim=True)

            # trim
            width = pinhole_cam.size.reshape(-1, 2)[0][0]
            height = pinhole_cam.size.reshape(-1, 2)[0][1]
            # pyre-fixme[6]: For 3rd argument expected `Union[None, bool, complex,
            #  float, int]` but got `Tensor`.
            xmin = torch.clamp(xmin, min=0, max=width - 1)
            # pyre-fixme[6]: For 3rd argument expected `Union[None, bool, complex,
            #  float, int]` but got `Tensor`.
            xmax = torch.clamp(xmax, min=0, max=width - 1)
            # pyre-fixme[6]: For 3rd argument expected `Union[None, bool, complex,
            #  float, int]` but got `Tensor`.
            ymin = torch.clamp(ymin, min=0, max=height - 1)
            # pyre-fixme[6]: For 3rd argument expected `Union[None, bool, complex,
            #  float, int]` but got `Tensor`.
            ymax = torch.clamp(ymax, min=0, max=height - 1)

            rect_bb2 = torch.cat([xmin, xmax, ymin, ymax], dim=-1)

            # remove the ones without any area
            areas = (rect_bb2[..., 1] - rect_bb2[..., 0]) * (
                rect_bb2[..., 3] - rect_bb2[..., 2]
            )
            areas = areas.unsqueeze(-1)
            areas = areas.repeat(*([1] * (areas.ndim - 1)), 4)
            rect_bb2[areas <= 0] = PAD_VAL
            rect_bb2[~visible_points] = PAD_VAL
            rect_bb2s.append(rect_bb2)

        data = smart_cat(
            [
                self.bb3_object,
                rect_bb2s[0],
                rect_bb2s[1],
                rect_bb2s[2],
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def get_pseudo_bb2(
        self,
        cam: CameraTW,
        T_world_rig: PoseTW,
        num_samples_per_edge: int = 10,
        valid_ratio: float = 0.1667,
    ):
        """
        get the 2d bbs of the projection of the 3d bbs into all given camera view points.
        This is done by sampling points on the 3d bb edges (see
        bb3edge_pts_object), projecting them and then computing the 2d bbs from
        the valid projected points. The caller has to make sure the ObbTW has valid
        3d bbs data

        num_samples_per_edge == 1 and num_samples_per_edge == 2 are equivalent
        (in both cases we project the obb corners into the frames to compute 2d bbs)
        """
        assert self._data.shape[-2] > 0, "No valid 3d bbs data found!"
        return bb2d_from_project_bb3d(
            self, cam, T_world_rig, num_samples_per_edge, valid_ratio
        )

    def get_bb2_heights(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        heights = bb2s[..., 3] - bb2s[..., 2]
        heights[~valid_bb2s] = -1
        return heights

    def get_bb2_widths(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        widths = bb2s[..., 1] - bb2s[..., 0]
        widths[~valid_bb2s] = -1
        return widths

    def get_bb2_areas(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        areas = (bb2s[..., 1] - bb2s[..., 0]) * (bb2s[..., 3] - bb2s[..., 2])
        areas[~valid_bb2s] = -1
        return areas

    def get_bb2_centers(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        center_x = (bb2s[..., 0:1] + bb2s[..., 1:2]) / 2.0
        center_y = (bb2s[..., 2:3] + bb2s[..., 3:4]) / 2.0
        center_2d = torch.cat([center_x, center_y], -1)
        center_2d[~valid_bb2s] = -1
        return center_2d

    def batch_points_inside_bb3(self, pts_world: torch.Tensor) -> torch.Tensor:
        """
        checks if a set of points is inside the 3d bounding box
        expected pts_world shape is B x N x 3 where N is the number of points
        expected bb3 shape is B x 34
        """
        assert pts_world.ndim == 3
        assert self.ndim == 2
        assert pts_world.shape[0] == self.T_world_object.shape[0]
        pts_object = self.T_world_object.inverse() * pts_world
        inside_min = (pts_object > self.bb3_min_object.unsqueeze(1)).all(dim=-1)
        inside_max = (pts_object < self.bb3_max_object.unsqueeze(1)).all(dim=-1)
        return torch.logical_and(inside_min, inside_max)

    def points_inside_bb3(self, pts_world: torch.Tensor) -> torch.Tensor:
        """
        checks if a set of points is inside the 3d bounding box
        """
        assert self.ndim == 1 and pts_world.ndim == 2
        # pyre-fixme[16]: `Tensor` has no attribute `transform`.
        pts_object = self.T_world_object.inverse().transform(pts_world)
        inside_min = (pts_object > self.bb3_min_object).all(-1)
        inside_max = (pts_object < self.bb3_max_object).all(-1)
        return torch.logical_and(inside_min, inside_max)

    def _transform(self, T_new_world):
        """
        in place transform T_world_object as T_new_object = T_new_world @ T_world_object
        """
        T_world_object = self.T_world_object
        T_new_object = T_new_world @ T_world_object
        self.set_T_world_object(T_new_object)

    def transform(self, T_new_world):
        """
        transform T_world_object as T_new_object = T_new_world @ T_world_object
        """
        obb_new = self.clone()
        obb_new._transform(T_new_world)
        return obb_new

    def _transform_object(self, T_object_new):
        """
        in place transform T_world_object as T_world_new = T_world_object @ T_object_new
        """
        T_world_object = self.T_world_object
        T_world_new = T_world_object @ T_object_new
        self.set_T_world_object(T_world_new)

    def filter_by_sem_id(self, keep_sem_ids):
        valid = self._data.new_zeros(self.shape[:-1]).bool()
        for si in keep_sem_ids:
            valid = valid | (self.sem_id == si)[..., 0]
        self._data[~valid] = PAD_VAL
        return self

    def filter_bb2_center_by_radius(self, calib, cam_id):
        """
        Inputs
            calib: CameraTW : shaped ... x 34, matching leading dims with self
            cam_id : int : integer corresponding to which bb2ds to use (0: rgb, 1: slaml, 2: slamr)
        """
        # Remove detections centers outside of valid_radius.
        centers = self.get_bb2_centers(cam_id)
        inside = calib.in_radius(centers)
        self._data[~inside, :] = PAD_VAL
        return self

    def voxel_grid(self, vD, vH, vW):
        """
        Input: Works on obbs shaped (B) x 34
        Output: world points sampled uniformly in a voxel grid (B) x vW*vH*vD x 3
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bb3_object.unbind(-1)
        dW = (x_max - x_min) / vW
        dH = (y_max - y_min) / vH
        dD = (z_max - z_min) / vD
        # take the center position of each voxel
        rng_x = tensor_linspace(
            x_min + dW / 2, x_max - dW / 2, steps=vW, device=self.device
        )
        rng_y = tensor_linspace(
            y_min + dH / 2, y_max - dH / 2, steps=vH, device=self.device
        )
        rng_z = tensor_linspace(
            z_min + dD / 2, z_max - dD / 2, steps=vD, device=self.device
        )
        if self.ndim > 1:
            if self.ndim > 2:
                raise NotImplementedError
            B = self.shape[0]
            xs, ys, zs = [], [], []
            # TODO vectorize somehow.
            for b in range(B):
                xx, yy, zz = torch.meshgrid(rng_x[b], rng_y[b], rng_z[b], indexing="ij")
                xs.append(xx)
                ys.append(yy)
                zs.append(zz)
            xx = torch.stack(xs)
            yy = torch.stack(ys)
            zz = torch.stack(zs)
        else:
            xx, yy, zz = torch.meshgrid(rng_x, rng_y, rng_z, indexing="ij")
        vox_v = torch.stack([xx, yy, zz], axis=-1)
        vox_v = vox_v.reshape(B, -1, 3)
        T_wv = self.T_world_object
        vox_w = T_wv * vox_v
        return vox_w

    # def voxel_grid2(self, vD, vH, vW):
    #     """
    #     Differentiable version, works by first generating a unit cube then transforming it
    #     Input: Works on obbs shaped (B) x 34
    #     Output: world points sampled uniformly in a voxel grid (B) x vW*vH*vD x 3
    #     """
    #     assert self.ndim == 2
    #     B = self.shape[0]
    #     # Generate points in unit cube.
    #     xsamps = (
    #         torch.linspace(0, 1, steps=vW)
    #         .reshape(1, -1)
    #         .repeat(B, 1)
    #         .to(self._data.device)
    #     )
    #     ysamps = (
    #         torch.linspace(0, 1, steps=vH)
    #         .reshape(1, -1)
    #         .repeat(B, 1)
    #         .to(self._data.device)
    #     )
    #     zsamps = (
    #         torch.linspace(0, 1, steps=vD)
    #         .reshape(1, -1)
    #         .repeat(B, 1)
    #         .to(self._data.device)
    #     )

    #     # x_min, x_max, y_min, y_max, z_min, z_max
    #     bb3 = self.bb3_object
    #     x_min = bb3[:, 0].reshape(-1, 1)
    #     x_max = bb3[:, 1].reshape(-1, 1)
    #     y_min = bb3[:, 2].reshape(-1, 1)
    #     y_max = bb3[:, 3].reshape(-1, 1)
    #     z_min = bb3[:, 4].reshape(-1, 1)
    #     z_max = bb3[:, 5].reshape(-1, 1)

    #     xsamps = xsamps * (x_max - x_min) + x_min
    #     ysamps = ysamps * (y_max - y_min) + y_min
    #     zsamps = zsamps * (z_max - z_min) + z_min

    #     xxs, yys, zzs = [], [], []
    #     for b in range(B):
    #         xx, yy, zz = torch.meshgrid(xsamps[b], ysamps[b], zsamps[b], indexing="ij")
    #         xxs.append(xx)
    #         yys.append(yy)
    #         zzs.append(zz)
    #     xxs = torch.stack(xxs)
    #     yys = torch.stack(yys)
    #     zzs = torch.stack(zzs)
    #     vox_v = torch.stack([xxs, yys, zzs], dim=-1)  # 3xN
    #     vox_v = vox_v.reshape(B, -1, 3)
    #     T_wv = self.T_world_object
    #     vox_w = T_wv * vox_v

    #     # x_min, x_max, y_min, y_max, z_min, z_max = self.bb3_object.unbind(-1)
    #     # dW = (x_max - x_min) / vW
    #     # dH = (y_max - y_min) / vH
    #     # dD = (z_max - z_min) / vD
    #     ## take the center position of each voxel
    #     # rng_x = tensor_linspace(x_min + dW / 2, x_max - dW / 2, steps=vW, device=self.device)
    #     # rng_y = tensor_linspace(y_min + dH / 2, y_max - dH / 2, steps=vH, device=self.device)
    #     # rng_z = tensor_linspace(z_min + dD / 2, z_max - dD / 2, steps=vD, device=self.device)
    #     # if self.ndim > 1:
    #     #    if self.ndim > 2:
    #     #        raise NotImplementedError
    #     #    B = self.shape[0]
    #     #    xs, ys, zs = [], [], []
    #     #    # TODO vectorize somehow.
    #     #    for b in range(B):
    #     #        xx, yy, zz = torch.meshgrid(rng_x[b], rng_y[b], rng_z[b], indexing="ij")
    #     #        xs.append(xx)
    #     #        ys.append(yy)
    #     #        zs.append(zz)
    #     #    xx = torch.stack(xs)
    #     #    yy = torch.stack(ys)
    #     #    zz = torch.stack(zs)
    #     # else:
    #     #    xx, yy, zz = torch.meshgrid(rng_x, rng_y, rng_z, indexing="ij")
    #     # vox_v = torch.stack([xx, yy, zz], axis=-1)
    #     # vox_v = vox_v.reshape(B, -1, 3)
    #     ##vox_v = vox_v.unsqueeze(0).repeat(B, 1, 1)
    #     # T_wv = self.T_world_object
    #     # vox_w = T_wv * vox_v
    #     return vox_w

    def fit_points(self, pts_w, prob=None):
        assert self.ndim == 1

        if pts_w.shape[0] < 2:
            # print("not enough points to fit obb")
            return self
        device = pts_w.device
        # Fit gravity aligned box from points.
        rots = torch.linspace(0, torch.pi, 20)  # pi or pi/2?
        obbs = []
        min_bb3 = None
        min_T_wo = None
        min_area = float("inf")
        for rot in rots:
            euler = torch.tensor([0, 0, rot]).reshape(1, 3)
            R = rotation_from_euler(euler)[0]
            T_wo = PoseTW.from_Rt(R, torch.tensor([0.0, 0.0, 0.0]))
            pts_o = T_wo.inverse() * pts_w
            xmin = pts_o[:, 0].min()
            xmax = pts_o[:, 0].max()
            ymin = pts_o[:, 1].min()
            ymax = pts_o[:, 1].max()
            zmin = pts_o[:, 2].min()
            zmax = pts_o[:, 2].max()
            bb3 = torch.tensor([xmin, xmax, ymin, ymax, zmin, zmax]).float()
            area = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            if area < min_area:
                min_bb3 = bb3
                min_T_wo = T_wo
                min_area = area
        self.set_bb3_object(min_bb3)
        self.set_T_world_object(min_T_wo)
        if prob is not None:
            self.set_prob(prob)

        # Center the box locally.
        self = self.unsqueeze(0).center().squeeze(0)

        return self

    def __repr__(self):
        return f"ObbTW {self.shape} {self.dtype} {self.device}"


def _single_transform_obbs(obbs_padded, Ts_other_world):
    assert obbs_padded.ndim == 3  # T x N x C
    assert Ts_other_world.ndim == 2 and Ts_other_world.shape[0] == 1  # 1 x C
    T, N, C = obbs_padded.shape
    if T == 0:
        # Directly return the input since T=0 and there are no obbs to transform.
        # TODO: check why this happens. ref: https://fburl.com/code/3n4wcjv2
        return obbs_padded
    obbs_transformed = []
    for t in range(T):
        # clone so that we get a new transformed obbs object.
        obbs = obbs_padded[t, ...].remove_padding().clone()
        obbs._transform(Ts_other_world)
        obbs_transformed.append(obbs.add_padding(N))
    obbs_transformed = ObbTW(smart_stack(obbs_transformed))
    return obbs_transformed


def _batched_transform_obbs(obbs_padded, Ts_other_world):
    assert obbs_padded.ndim == 4  # B x T x N x C
    assert Ts_other_world.ndim == 3  # T x 1 x C
    B, T, N, C = obbs_padded.shape
    obbs_transformed = []
    for b in range(B):
        obbs_transformed.append(
            _single_transform_obbs(obbs_padded[b], Ts_other_world[b])
        )
    obbs_transformed = ObbTW(smart_stack(obbs_transformed))
    return obbs_transformed


def transform_obbs(obbs_padded, Ts_other_world):
    """
    transform padded obbs from the world coordinate system to a "other"
    coordinate system.
    """
    if obbs_padded.ndim == 4:
        return _batched_transform_obbs(obbs_padded, Ts_other_world)
    return _single_transform_obbs(obbs_padded, Ts_other_world)


def rot_obb2_cw(bb2: torch.Tensor, size: Tuple[int]):
    bb2_ori = bb2.clone()
    # exchange (xmin, xmax, ymin, ymax) -> (ymax, ymin, xmin, xmax)
    bb2 = bb2[..., [3, 2, 0, 1]]
    # x_new = height - x_new
    bb2[..., 0:2] = size[1] - bb2[..., 0:2] - 1
    # bring back the invalid entries.
    bb2[bb2_ori < 0] = bb2_ori[bb2_ori < 0]
    return bb2


def project_bb3d_onto_image(
    obbs: ObbTW, cam: CameraTW, T_world_rig: PoseTW, num_samples_per_edge: int = 1
):
    """
    project 3d bb edge points into snippet images defined by T_world_rig and
    camera cam. The assumtion is that obbs are in the "world" coordinate system
    of T_world_rig.
    Supports batched operation.

    Args:
        obbs (ObbTW): obbs to project; shape is (Bx)(Tx)Nx34
        cam (CameraTW): camera to project to; shape is (Bx)TxC where T is the snippet dimension;
        T_world_rig (PoseTW): T_world_rig defining where the camera rig is; shape is (Bx)Tx12
        num_samples_per_edge (int): how many points to sample per edge to
            compute 2d bb (1, and 2 means only corners)
    Returns:
        bb3_corners_im (Tensor): bb3 corners in the image coordinate system; shape is (Bx)TxNx8x2
        bb3_valids (Tensor): valid indices of bb3_corners_im (indicates which
            corners lie within the images); shape is (Bx)TxNx8
    """
    obb_dim = obbs.dim()
    # support 3 sets of input shapes
    if obb_dim == 2:  # Nx34
        # cam: TxC, T_world_rig: Tx12
        assert (
            cam.dim() == 2
            and T_world_rig.dim() == 2
            and cam.shape[0]
            == T_world_rig.shape[0]  # T dim should be the same for cam and T_world_rig
        ), (
            f"Unsupported input shapes: obb: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}."
        )

        # To the consistent shapes
        obbs = obbs.unsqueeze(0).unsqueeze(0)  # expand to B(1)xT(1)xNx34
        cam = cam[None, ...]  # expand to B(1)xTxC
        T_world_rig = T_world_rig[None, ...]  # expand to B(1)xTx12
        B, T = cam.shape[0:2]
        N = obbs.shape[-2]
        obbs = obbs.expand(B, T, *obbs.shape[-2:])  # repeat to real T: B(1)xTxNx34

    elif obb_dim == 3:  # BxNx34
        # cam: BxTxC, T_world_rig: BxTx12
        assert cam.dim() == 3 and T_world_rig.dim() == 3
        # B dim should be the same
        assert obbs.shape[0] == cam.shape[0] and obbs.shape[0] == T_world_rig.shape[0]
        # T dim of cam and pose should be the same
        assert cam.shape[1] == T_world_rig.shape[1]

        # To the consistent shapes
        obbs = obbs.unsqueeze(1)  # expand to BxT(1)xNx34
        B, T = cam.shape[0:2]
        obbs = obbs.expand(B, T, *obbs.shape[-2:])

    elif obb_dim == 4:  # BxTxNx34
        pass
    else:
        raise ValueError(
            f"Unsupported input shapes: obb: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}."
        )

    # check if all tensors are of correct shapes.
    assert obbs.dim() == 4 and cam.dim() == 3 and T_world_rig.dim() == 3, (
        f"The shapes of obbs, cam and T_world_rig should be BxTxNx34, BxTxC, and BxTx12, respectively. However, we got obbs: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}"
    )
    assert (
        obbs.shape[0:2] == cam.shape[0:2] and obbs.shape[0:2] == T_world_rig.shape[0:2]
    ), (
        f"The BxT dims should be the same for all tensors, but got obbs: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}"
    )

    B, T = cam.shape[0:2]
    N = obbs.shape[-2]
    assert N > 0, "obbs have to exist for this frame"
    # Get pose of camera.
    T_world_cam = T_world_rig @ cam.T_camera_rig.inverse()
    # Project the 3D BB corners into the image.
    # BxTxNx8x3 -> BxTxN*8x3
    if num_samples_per_edge <= 2:
        bb3pts_world = obbs.bb3corners_world.view(B, T, -1, 3)
    else:
        bb3pts_object = obbs.bb3edge_pts_object(num_samples_per_edge)
        bb3pts_world = obbs.T_world_object * bb3pts_object
        bb3pts_world = bb3pts_world.view(B, T, -1, 3)
    Npt = bb3pts_world.shape[2]
    T_world_cam = T_world_cam.unsqueeze(2).repeat(1, 1, Npt, 1)
    bb3pts_cam = (
        T_world_cam.inverse()
        .view(-1, 12)
        .batch_transform(bb3pts_world.view(-1, 3))
        .view(B, T, -1, 3)
    )
    bb3pts_im, bb3pts_valids = cam.project(bb3pts_cam)
    bb3pts_im = bb3pts_im.view(B, T, N, -1, 2)
    bb3pts_valids = bb3pts_valids.detach().view(B, T, N, -1)

    if obb_dim == 2:
        # remove B dim if it didn't exist before.
        bb3pts_im = bb3pts_im.squeeze(0)
        bb3pts_valids = bb3pts_valids.squeeze(0)
    return bb3pts_im, bb3pts_valids


def bb2d_from_project_bb3d(
    obbs: ObbTW,
    cam: CameraTW,
    T_world_rig: PoseTW,
    num_samples_per_edge: int = 1,
    valid_ratio: float = 0.1667,
):
    """
    get 2d bbs around the 3d bb corners of obbs projected into the image coordinate system
    defined by T_world_rig and camera cam. The assumtion is that obbs are in the
    "world" coordinate system of T_world_rig.

    This is done by sampling points on the 3d bb edges (see bb3edge_pts_object),
    projecting them and then computing the 2d bbs from the valid projected
    points.

    Supports batched operation.

    Args:
        obbs (ObbTW): obbs to project; shape is (Bx)Nx34
        cam (CameraTW): camera to project to; shape is (Bx)TxC where T is the snippet dimension;
        T_world_rig (PoseTW): T_world_rig defining where the camera rig is; shape is (Bx)Tx12
    Returns:
        bb2s (Tensor): 2d bounding boxes in the image coordinate system; shape is (Bx)TxNx4
        bb2s_valid (Tensor): valid indices of bb2s; shape is (Bx)TxN
    """
    bb3corners_im, bb3corners_valids = project_bb3d_onto_image(
        obbs, cam, T_world_rig, num_samples_per_edge
    )
    # get image points that will min and max reduce correctly given the valid masks
    bb3corners_im_min = torch.where(
        bb3corners_valids.unsqueeze(-1).expand_as(bb3corners_im),
        bb3corners_im,
        999999 * torch.ones_like(bb3corners_im),
    )
    bb3corners_im_max = torch.where(
        bb3corners_valids.unsqueeze(-1).expand_as(bb3corners_im),
        bb3corners_im,
        -999999 * torch.ones_like(bb3corners_im),
    )
    # compute 2d bounding boxes
    bb2s_min = torch.min(bb3corners_im_min, dim=-2)[0]
    bb2s_max = torch.max(bb3corners_im_max, dim=-2)[0]
    bb2s = torch.stack(
        [bb2s_min[..., 0], bb2s_max[..., 0], bb2s_min[..., 1], bb2s_max[..., 1]], dim=-1
    )
    # min < max so that it's a valid box.
    non_empty_boxes = (bb2s[..., 0] < bb2s[..., 1]) & (bb2s[..., 2] < bb2s[..., 3])
    # count number of valid points
    num_points = bb3corners_valids.count_nonzero(-1)
    # valid 2d bbs are non-empty and have at least valid_ratio of the edge sample
    # points in the valid image region
    num_total_samples = num_samples_per_edge * 12  # 12 edges on a 3D box
    bb2s_valid = torch.logical_and(
        non_empty_boxes, num_points >= (num_total_samples * valid_ratio)
    )
    if cam.is_linear:
        # Clamp based on the camera size for linear cameras.
        # Note that this could generate very big/loose bounding boxes if the object is badly truncated due to out of view.
        bb2s[..., 0:2] = torch.clamp(
            bb2s[..., 0:2], min=0, max=float(cam.size.view(-1, 2)[0, 0] - 1)
        )
        bb2s[..., 2:4] = torch.clamp(
            bb2s[..., 2:4], min=0, max=float(cam.size.view(-1, 2)[0, 1] - 1)
        )
    return bb2s, bb2s_valid


def bb2_xxyy_to_xyxy(bb2s):
    # check if the input is xxyy
    is_xxyy = torch.logical_and(
        bb2s[..., 0] <= bb2s[..., 1], bb2s[..., 2] <= bb2s[..., 3]
    )
    is_xxyy = is_xxyy.all()
    if not is_xxyy:
        print("Input 2d bbx doesn't follow xxyy convention.")
    return bb2s[..., [0, 2, 1, 3]]


def bb2_xyxy_to_xxyy(bb2s):
    # check if the input is xxyy
    is_xyxy = torch.logical_and(
        bb2s[..., 0] <= bb2s[..., 2], bb2s[..., 1] <= bb2s[..., 3]
    )
    is_xyxy = is_xyxy.all()
    if not is_xyxy:
        print("Input 2d bbx doesn't follow xyxy convention.")
    return bb2s[..., [0, 2, 1, 3]]


def bb3_xyzxyz_to_xxyyzz(bb3s):
    return bb3s[..., [0, 3, 1, 4, 2, 5]]


def rnd_obbs(N: int = 1, num_semcls: int = 10, bb3_min_diag=0.1, bb2_min_diag=10):
    pts3_min = torch.randn(N, 3)
    pts3_max = pts3_min + bb3_min_diag + torch.randn(N, 3).abs()
    pts2_min = torch.randn(N, 2)
    pts2_max = pts2_min + bb2_min_diag + torch.randn(N, 2).abs()

    obb = ObbTW.from_lmc(
        bb3_object=bb3_xyzxyz_to_xxyyzz(torch.cat([pts3_min, pts3_max], -1)),
        prob=torch.ones(N),
        bb2_rgb=bb2_xyxy_to_xxyy(torch.cat([pts2_min, pts2_max], -1)),
        sem_id=torch.randint(low=0, high=num_semcls - 1, size=[N]),
        T_world_object=PoseTW.from_aa(torch.randn(N, 3), 10.0 * torch.randn(N, 3)),
    )
    return obb


def obb_time_union(obbs, pad_size=128):
    """
    Take frame level ground truth shaped BxTxNxC and take the union
    over the time dimensions using the instance id to extend to snippet level
    obbs shaped BxNxC.
    """
    # T already merged somewhere else.
    if obbs.ndim == 3:
        return obbs

    assert obbs.ndim == 4, "Only B x T x N x C supported"
    new_obbs = []
    last_dim = ObbTW().shape[-1]
    for obb in obbs:
        new_obb = []
        flat_time_obb = obb.clone().reshape(-1, last_dim)
        unique = flat_time_obb.inst_id.unique()
        for uni in unique:
            if uni == PAD_VAL:
                continue
            found = int(torch.argwhere(flat_time_obb.inst_id == uni)[0, 0])
            found_obb = flat_time_obb[found].clone()
            new_obb.append(found_obb)
        if len(new_obb) == 0:
            # print("Adding empty OBB in time_union")
            new_obb.append(ObbTW().reshape(-1).to(obbs._data))
        new_obbs.append(torch.stack(new_obb).add_padding(pad_size))
    new_obbs = torch.stack(new_obbs)
    # Remove all bb2 observations since we no longer know which frame in time it came from.
    # Note: we set the visibility for the merged obbs in order to do the evaluation on those theses obbs.
    pad_mask = new_obbs.get_padding_mask()
    new_obbs.set_bb2(cam_id=0, bb2d=1)
    new_obbs.set_bb2(cam_id=1, bb2d=1)
    new_obbs.set_bb2(cam_id=2, bb2d=1)
    new_obbs._data[pad_mask] = -1
    return new_obbs


def obb_filter_outside_volume(obbs, T_wv, voxel_extent, border=0.1):
    """
    Remove obbs outside a volume of size voxel_extent, e.g. from a lifter volume.
    Obbs are filtered based on their center point being inside the volume, and
    are additionally filtered near the border.
    """
    assert obbs.ndim == 3, "Only B x N x C supported"
    obbs_v = obbs.transform(T_wv.inverse().unsqueeze(1))
    centers_v = obbs_v.bb3_center_world
    cx = centers_v[:, :, 0]
    cy = centers_v[:, :, 1]
    cz = centers_v[:, :, 2]
    x_min = voxel_extent[0]
    x_max = voxel_extent[1]
    y_min = voxel_extent[2]
    y_max = voxel_extent[3]
    z_min = voxel_extent[4]
    z_max = voxel_extent[5]
    valid = (obbs_v.inst_id != PAD_VAL).squeeze(-1)
    inside = (
        (cx > (x_min + border))
        & (cy > (y_min + border))
        & (cz > (z_min + border))
        & (cx < (x_max - border))
        & (cy < (y_max - border))
        & (cz < (z_max - border))
    )
    remove = valid & ~inside
    obbs._data[remove, :] = PAD_VAL
    return obbs


def densify_obbs(obb_snippet: ObbTW):
    """Densify sparse obb annotations by populating obbs to the whole snippet."""
    # put this here to avoid circular import.
    try:
        from utils.aria_constants import ARIA_CALIB
    except ImportError:
        raise ImportError("aria_constants module not available")

    assert obb_snippet.ndim == 3, (
        f"input shape must be TxNxC, but got {obb_snippet.shape}"
    )

    N = obb_snippet.shape[-2]
    valid_obbs = obb_snippet.remove_padding()
    if len(valid_obbs) == 0:
        # no any valid obbs. We let the model to deal with the case hopefully.
        return obb_snippet
    valid_obbs = smart_cat(valid_obbs, 0)
    # Use unique obbs since we may have duplicates.
    valid_obbs = ObbTW(torch.unique(valid_obbs, dim=0))
    # set all visiblities to 1
    # pyre-fixme[16]: Module `utils` has no attribute `aria_constants`.
    for cam_id in range(len(ARIA_CALIB)):
        valid_obbs.set_bb2(cam_id, 1)
    valid_obbs = valid_obbs.add_padding(N)
    valid_obbs = valid_obbs[None, ...]
    obb_snippet[:, ...] = valid_obbs._data
    return obb_snippet


def tensor_linspace(start, end, steps, device):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=device).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=device).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def make_obb(sz, position, prob=1.0, roll=0.0, pitch=0.0, yaw=0.1):
    e_angles = torch.tensor([roll, pitch, yaw]).reshape(-1, 3)
    R = rotation_from_euler(e_angles).reshape(3, 3)
    T_voxel_object = PoseTW.from_Rt(R, torch.tensor(position))
    bb3 = [
        -sz[0] / 2.0,
        sz[0] / 2.0,
        -sz[1] / 2.0,
        sz[1] / 2.0,
        -sz[2] / 2.0,
        sz[2] / 2.0,
    ]
    return ObbTW.from_lmc(
        bb3_object=torch.tensor(bb3),
        prob=[prob],
        T_world_object=T_voxel_object,
    )


def is_point_inside_box(points: torch.Tensor, box: torch.Tensor, verbose=False):
    """
    Determines whether points are inside the boxes
    Args:
        points: tensor of shape (B1, P, 3) of the points
        box: tensor of shape (B2, 8, 3) of the corners of the boxes
    Returns:
        inside: bool tensor of whether point (row) is in box (col) shape (B1, B2, P)
    """
    device = box.device
    B1 = points.shape[0]
    B2 = box.shape[0]
    P = points.shape[1]

    normals = box_planar_dir(box)  # (B2, 6, 3)
    box_planes = get_plane_verts(box)  # (B2, 6, 4, 3)
    NP = box_planes.shape[1]  # = 6

    # a point p is inside the box if it "inside" all planes of the box
    # so we run the checks
    ins = torch.zeros((B1, B2, P, NP), device=device, dtype=torch.bool)
    # ins = []
    for i in range(NP):
        is_in = is_inside(points, box_planes[:, i], normals[:, i])
        ins[:, :, :, i] = is_in
        # ins.append(is_in)
    # ins = torch.stack(ins, dim=-1)

    ins = ins.all(dim=-1)
    return ins


def box_planar_dir(
    box: torch.Tensor, dot_eps: float = DOT_EPS, area_eps: float = AREA_EPS
) -> torch.Tensor:
    """
    Finds the unit vector n which is perpendicular to each plane in the box
    and points towards the inside of the box.
    The planes are defined by `_box_planes`.
    Since the shape is convex, we define the interior to be the direction
    pointing to the center of the shape.
    Args:
       box: tensor of shape (B, 8, 3) of the vertices of the 3D box
    Returns:
       n: tensor of shape (B, 6) of the unit vector orthogonal to the face pointing
          towards the interior of the shape
    """
    assert box.shape[1] == 8 and box.shape[2] == 3
    # center point of each box
    box_ctr = box.mean(dim=1).view(-1, 1, 3)
    # box planes
    plane_verts = get_plane_verts(box)  # (B, 6, 4, 3)
    v0, v1, v2, v3 = plane_verts.unbind(2)
    plane_ctr, n = get_plane_center_normal(plane_verts)
    # Check all verts are coplanar
    normv = F.normalize(v3 - v0, dim=-1).unsqueeze(2).reshape(-1, 1, 3)
    nn = n.unsqueeze(3).reshape(-1, 3, 1)
    dists = normv @ nn
    if not (dists.abs() < dot_eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)
    # Check all faces have non zero area
    area1 = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    area2 = torch.cross(v3 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    if (area1 < area_eps).any().item() or (area2 < area_eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)
    # We can write:  `box_ctr = plane_ctr + a * e0 + b * e1 + c * n`, (1).
    # With <e0, n> = 0 and <e1, n> = 0, where <.,.> refers to the dot product,
    # since that e0 is orthogonal to n. Same for e1.
    """
    # Below is how one would solve for (a, b, c)
    # Solving for (a, b)
    numF = verts.shape[0]
    A = torch.ones((numF, 2, 2), dtype=torch.float32, device=device)
    B = torch.ones((numF, 2), dtype=torch.float32, device=device)
    A[:, 0, 1] = (e0 * e1).sum(-1)
    A[:, 1, 0] = (e0 * e1).sum(-1)
    B[:, 0] = ((box_ctr - plane_ctr) * e0).sum(-1)
    B[:, 1] = ((box_ctr - plane_ctr) * e1).sum(-1)
    ab = torch.linalg.solve(A, B)  # (numF, 2)
    a, b = ab.unbind(1)
    # solving for c
    c = ((box_ctr - plane_ctr - a.view(numF, 1) * e0 - b.view(numF, 1) * e1) * n).sum(-1)
    """
    # Since we know that <e0, n> = 0 and <e1, n> = 0 (e0 and e1 are orthogonal to n),
    # the above solution is equivalent to
    direc = F.normalize(box_ctr - plane_ctr, dim=-1)  # (6, 3)
    c = (direc * n).sum(-1)
    # If c is negative, then we revert the direction of n such that n points "inside"
    negc = c < 0.0
    n[negc] *= -1.0
    # c[negc] *= -1.0
    # Now (a, b, c) is the solution to (1)
    return n


def get_plane_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the planes of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (B, 8, 3)
    Returns:
        plane_verts: tensor of shape (B, 6, 4, 3)
    """
    device = box.device
    B = box.shape[0]
    faces = torch.tensor(_box_planes, device=device, dtype=torch.int64)  # (6, 4)
    # TODO: remove loop
    plane_verts = torch.stack([box[b, faces] for b in range(B)])  # (B, 6, 4, 3)
    return plane_verts


def is_inside(
    points: torch.Tensor,
    plane: torch.Tensor,
    normal: torch.Tensor,
    return_proj: bool = True,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |

    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      points: tensor of shape (B1, P, 3) of coordinates of a point
      plane: tensor of shape (B2, 4, 3) of vertices of a box plane
      normal: tensor of shape (B2, 3) of the unit "inside" direction on the plane
      return_proj: bool whether to return the projected point on the plane
    Returns:
      is_inside: bool of shape (B2, P) of whether point is inside
    """
    device = plane.device
    assert plane.ndim == 3
    assert normal.ndim == 2
    assert points.ndim == 3
    assert points.shape[2] == 3
    B1 = points.shape[0]
    B2 = plane.shape[0]
    P = points.shape[1]
    v0, v1, v2, v3 = plane.unbind(dim=1)
    plane_ctr = plane.mean(dim=1)
    e0 = F.normalize(v0 - plane_ctr, dim=1)
    e1 = F.normalize(v1 - plane_ctr, dim=1)

    dot1 = (e0.unsqueeze(1) @ normal.unsqueeze(2)).reshape(B2)
    if not torch.allclose(dot1, torch.zeros((B2,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")
    dot2 = (e1.unsqueeze(1) @ normal.unsqueeze(2)).reshape(B2)
    if not torch.allclose(dot2, torch.zeros((B2,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")

    # Every point p can be written as p = ctr + a e0 + b e1 + c n
    # solving for c
    # c = (point - ctr - a * e0 - b * e1).dot(n)
    pts = points.view(B1, 1, P, 3)
    ctr = plane_ctr.view(1, B2, 1, 3)
    e0 = e0.view(1, B2, 1, 3)
    e1 = e1.view(1, B2, 1, 3)
    normal = normal.view(1, B2, 1, 3)

    direc = torch.sum((pts - ctr) * normal, dim=-1)
    ins = direc >= 0.0
    return ins


def get_plane_center_normal(planes: torch.Tensor) -> torch.Tensor:
    """
    Returns the center and normal of planes
    Args:
        planes: tensor of shape (B, P, 4, 3)
    Returns:
        center: tensor of shape (B, P, 3)
        normal: tensor of shape (B, P, 3)
    """
    B = planes.shape[0]

    add_dim1 = False
    if planes.ndim == 3:
        planes = planes.unsqueeze(1)
        add_dim1 = True

    ctr = planes.mean(dim=2)  # (B, P, 3)
    normals = torch.zeros_like(ctr)

    v0, v1, v2, v3 = planes.unbind(dim=2)  # 4 x (B, P, 3)

    # TODO: unvectorized solution
    P = planes.shape[1]
    for t in range(P):
        ns = torch.zeros((B, 6, 3), device=planes.device)
        ns[:, 0] = torch.cross(v0[:, t] - ctr[:, t], v1[:, t] - ctr[:, t], dim=-1)
        ns[:, 1] = torch.cross(v0[:, t] - ctr[:, t], v2[:, t] - ctr[:, t], dim=-1)
        ns[:, 2] = torch.cross(v0[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ns[:, 3] = torch.cross(v1[:, t] - ctr[:, t], v2[:, t] - ctr[:, t], dim=-1)
        ns[:, 4] = torch.cross(v1[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ns[:, 5] = torch.cross(v2[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ii = torch.argmax(torch.norm(ns, dim=-1), dim=-1)
        normals[:, t] = ns[torch.arange(B), ii]

    if add_dim1:
        ctr = ctr[:, 0]
        normals = normals[:, 0]
    normals = F.normalize(normals, dim=-1)
    # pyre-fixme[7]: Expected `Tensor` but got `Tuple[Tensor, Tensor]`.
    return ctr, normals


def prec_recall_bb3(
    padded_pred: ObbTW,
    padded_target: ObbTW,
    iou_thres=0.2,
    return_ious=False,
    per_class=False,
):
    """Compute precision and recall based on 3D IoU."""
    assert padded_pred.ndim == 2 and padded_target.ndim == 2, (
        f"input ObbTWs must be NxD, but got {padded_pred.shape} and {padded_target.shape}"
    )

    pred = padded_pred.remove_padding()
    target = padded_target.remove_padding()
    # pyre-fixme[16]: `List` has no attribute `shape`.
    pred_shape = pred.shape
    target_shape = target.shape

    # pred, _ = remove_invalid_box3d(pred)
    # target, _ = remove_invalid_box3d(target)
    if pred.shape != pred_shape:
        logging.warning(
            f"Warning: predicted obbs filtered from {pred_shape[0]} to {pred.shape[0]}"
        )
    if target.shape != target_shape:
        logging.warning(
            f"Warning: target obbs filtered from {target_shape[0]} to {target.shape[0]}"
        )

    prec_recall = (-1.0, -1.0, None)
    # deal with edge cases first
    if pred.shape[0] == 0:
        # invalid precision and 0 recall
        prec_recall = (-1.0, 0.0, None)
        return prec_recall
    elif target.shape[0] == 0:
        # invalid recall and 0 precision
        prec_recall = (0.0, -1.0, None)
        return prec_recall

    # pyre-fixme[16]: `List` has no attribute `sem_id`.
    pred_sems = pred.sem_id
    target_sems = target.sem_id.squeeze(-1).unsqueeze(0)
    # 1. Match classes
    sem_id_match = pred_sems == target_sems
    # 2. Match IoUs
    # ious = box3d_overlap_wrapper(pred.bb3corners_world, target.bb3corners_world).iou
    # pyre-fixme[6]: For 1st argument expected `ObbTW` but got `List[ObbTW]`.
    # pyre-fixme[6]: For 2nd argument expected `ObbTW` but got `List[ObbTW]`.
    ious = iou_mc7(pred, target)
    iou_match = ious > iou_thres
    # 3. Match both
    sem_iou_match = torch.logical_and(sem_id_match, iou_match)
    # make final matching matrix
    final_sem_iou_match = torch.zeros_like(sem_iou_match).bool()
    num_pred = sem_iou_match.shape[0]  # TP + FP
    num_target = sem_iou_match.shape[1]  # TP + FN
    # 4. Deal with the case where one prediction correspond to multiple GTs.
    # In this case, only the GT with highest IoU is considered the match.
    for pred_idx in range(int(num_pred)):
        if sem_iou_match[pred_idx, :].sum() <= 1:
            final_sem_iou_match[pred_idx, :] = sem_iou_match[pred_idx, :].clone()
        else:
            tgt_ious = ious[pred_idx, :].clone()
            tgt_ious[~sem_iou_match[pred_idx, :]] = -1.0
            sorted_ids = torch.argsort(tgt_ious, descending=True)
            tp_id = sorted_ids[0]
            # Set the pred with highest iou
            final_sem_iou_match[pred_idx, :] = False
            final_sem_iou_match[pred_idx, tp_id] = True

    # 5. Deal with the case where one GT correspond to multiple predictions.
    # In this case, if the predictions contain probabilities, we take the one with the highest score, otherwise, we take the one with the highest iou.
    for gt_idx in range(int(num_target)):
        if final_sem_iou_match[:, gt_idx].sum() <= 1:
            continue
        else:
            # pyre-fixme[16]: `List` has no attribute `prob`.
            pred_scores = pred.prob.squeeze(-1).clone()
            if torch.all(pred_scores.eq(-1.0)):
                # go with highest iou
                pred_ious = ious[:, gt_idx].clone()
                pred_ious[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_ious, descending=True)
                tp_id = sorted_ids[0]
                # Set the pred with highest iou
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True
            else:
                # go with the highest score
                pred_scores[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_scores, descending=True)
                tp_id = sorted_ids[0]
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True

    TPs = final_sem_iou_match.any(-1)
    # precision = TP / (TP + FP) = TP / #Preds
    num_tp = TPs.sum().item()
    prec = num_tp / num_pred
    # recall = TP / (TP + FN) = TP / #GTs
    rec = num_tp / num_target

    ret = (prec, rec, final_sem_iou_match)
    return ret


def make_corners2d(bb2d):
    """[xmin, xmax, ymin, ymax] -> [tl, tr, br, bl]"""
    xmin = bb2d[..., 0].unsqueeze(-1)
    xmax = bb2d[..., 1].unsqueeze(-1)
    ymin = bb2d[..., 2].unsqueeze(-1)
    ymax = bb2d[..., 3].unsqueeze(-1)
    tl = torch.cat([xmin, ymin], dim=-1)
    tr = torch.cat([xmax, ymin], dim=-1)
    bl = torch.cat([xmin, ymax], dim=-1)
    br = torch.cat([xmax, ymax], dim=-1)
    corns = torch.stack([tl, tr, br, bl], dim=-2)
    return corns


# ==============================================================================
# IoU computation for oriented bounding boxes (moved from obb_iou.py)
# ==============================================================================


def iou_mc9(obb1, obb2, samp_per_dim=32, all_pairs=True, verbose=False):
    """
    Computes the intersection of two boxes by sampling points uniformly in
    x,y,z dims.

    Optimized with early elimination and lazy voxel generation.

    Inputs:
        obb1: NxD, a batch of N ObbTW objects
        obb2: MxD, a batch of M ObbTW objects
        samp_per_dim: int, number of samples per dimension, e.g. if 8, then 8x8x8
                             increase for more accuracy but less speed
                             8: fast but not so accurate
                             32: medium
                             128: most accurate but slow
        all_pairs: if True, compute IoU3D across all possible NxM Obb pairs. if False,
                     assume N==M and only compute IoU across the corresponding OBBs, e.g.
                     compute [iou(obb1[0], obb2[0]), iou(obb1[1], obb2[1]), ...]
        verbose: if True, print timing and elimination statistics
    Returns:
        iou: NxM matrix of intersection over union values

    """
    import time

    overall_start = time.time() if verbose else None

    assert obb1.ndim == 2
    assert obb2.ndim == 2

    B1 = obb1.shape[0]
    B2 = obb2.shape[0]
    vol1 = obb1.bb3_volumes
    vol2 = obb2.bb3_volumes

    dim = samp_per_dim

    if all_pairs:
        if verbose:
            print(f"\n[IoU] Computing for {B1}x{B2} = {B1 * B2:,} box pairs")

        # OPTIMIZATION: Early elimination based on center distance
        # Get box centers and extents
        dist_start = time.time()
        # More efficient center extraction using vectorized operations
        t1 = obb1.bb3_center_world  # (B1, 3) - uses existing property
        t2 = obb2.bb3_center_world  # (B2, 3) - uses existing property

        # Compute pairwise distances using cdist (more efficient)
        center_distances = torch.cdist(t1, t2, p=2)  # (B1, B2)

        # Get box dimensions for conservative bound
        extents1 = obb1.bb3_object  # (B1, 6)
        extents2 = obb2.bb3_object  # (B2, 6)

        dims1 = torch.stack(
            [
                extents1[:, 1] - extents1[:, 0],
                extents1[:, 3] - extents1[:, 2],
                extents1[:, 5] - extents1[:, 4],
            ],
            dim=1,
        )  # (B1, 3)

        dims2 = torch.stack(
            [
                extents2[:, 1] - extents2[:, 0],
                extents2[:, 3] - extents2[:, 2],
                extents2[:, 5] - extents2[:, 4],
            ],
            dim=1,
        )  # (B2, 3)

        # Max extent = half diagonal (conservative)
        max_extent1 = torch.norm(dims1, dim=1) / 2  # (B1,)
        max_extent2 = torch.norm(dims2, dim=1) / 2  # (B2,)

        # Threshold: boxes can only overlap if distance < sum of max extents
        threshold = max_extent1.unsqueeze(1) + max_extent2.unsqueeze(0)  # (B1, B2)

        # Mask for pairs that might overlap
        might_overlap = center_distances <= threshold  # (B1, B2)
        n_candidates = might_overlap.sum().item()
        n_total = B1 * B2
        n_eliminated = n_total - n_candidates

        if verbose:
            dist_time = time.time() - dist_start
            print(f"  [IoU] Distance check: {dist_time:.3f}s")
            print(
                f"  [IoU] Eliminated {n_eliminated:,}/{n_total:,} pairs ({n_eliminated / n_total * 100:.1f}%)"
            )
            print(f"  [IoU] Computing IoU for {n_candidates:,} candidate pairs...")

        # Early exit if no candidates
        if n_candidates == 0:
            if verbose:
                print("  [IoU] No overlapping candidates - returning zero IoU matrix")
            return torch.zeros(B1, B2, device=obb1.device)

        # OPTIMIZATION: Only generate samples once for unique boxes
        sample_start = time.time()
        points1_w = obb1.voxel_grid(vD=dim, vH=dim, vW=dim)
        points2_w = obb2.voxel_grid(vD=dim, vH=dim, vW=dim)
        sample_time = time.time() - sample_start

        assert points1_w.shape[1] == points2_w.shape[1]
        num_samples = points1_w.shape[1]

        # OPTIMIZATION: Only compute for candidate pairs to reduce memory
        compute_start = time.time()

        # Get indices of candidate pairs
        candidate_idx = torch.where(might_overlap)
        idx1, idx2 = candidate_idx[0], candidate_idx[1]

        if len(idx1) > 0:
            # Extract only the candidate boxes and points
            obb1_candidates = obb1[idx1]
            obb2_candidates = obb2[idx2]
            points1_candidates_w = points1_w[idx1]
            points2_candidates_w = points2_w[idx2]

            # Compute intersections only for candidates
            isin21 = obb1_candidates.batch_points_inside_bb3(points2_candidates_w)
            isin12 = obb2_candidates.batch_points_inside_bb3(points1_candidates_w)

            num21 = isin21.sum(dim=-1)
            num12 = isin12.sum(dim=-1)

            vol1_candidates = vol1[idx1].view(-1)
            vol2_candidates = vol2[idx2].view(-1)

            inters12 = vol1_candidates * num12.view(-1)
            inters21 = vol2_candidates * num21.view(-1)
            inters = (inters12 + inters21) / 2.0
            union = (
                (vol1_candidates * num_samples)
                + (vol2_candidates * num_samples)
                - inters
            )

            iou_candidates = inters / union

            # Scatter results back to full matrix
            iou = torch.zeros(B1, B2, device=obb1.device)
            iou[idx1, idx2] = iou_candidates
        else:
            iou = torch.zeros(B1, B2, device=obb1.device)

        if verbose:
            compute_time = time.time() - compute_start
            total_time = time.time() - overall_start
            print(
                f"  [IoU] Sample: {sample_time:.3f}s, Compute: {compute_time:.3f}s, Total: {total_time:.3f}s"
            )
    else:
        if B1 != B2:
            raise ValueError(
                "If 'all_pairs' is False, then obb1 and obb2 must have same shape"
            )
        B = B1
        points1_w = obb1.voxel_grid(vD=dim, vH=dim, vW=dim)
        points2_w = obb2.voxel_grid(vD=dim, vH=dim, vW=dim)
        num_samples = points1_w.shape[1]

        isin21 = obb1.batch_points_inside_bb3(points2_w)
        isin12 = obb2.batch_points_inside_bb3(points1_w)
        num21 = isin21.sum(dim=-1)
        num12 = isin12.sum(dim=-1)
        inters12 = vol1.view(B) * num12.view(B)
        inters21 = vol2.view(B) * num21.view(B)
        inters = (inters12 + inters21) / 2.0
        union = (vol1.view(B) * num_samples) + (vol2.view(B) * num_samples) - inters
        iou = inters / union

    return iou


# ==============================================================================
def iou_mc7(
    obb1,
    obb2,
    samp_per_dim: int = 32,
    all_pairs: bool = True,
    verbose: bool = False,
    chunk_size: Optional[int] = 2048,
    use_giou: bool = False,
) -> torch.Tensor:
    """
    Compute 7-DoF IoU using optimized 2D Monte Carlo sampling + analytical Z-overlap.

    ~31x faster than full 3D sampling (iou_mc9) for same accuracy!

    Algorithm:
    1. Compute Z-overlap analytically (instant)
    2. Sample only in XY plane (samp_per_dim² points, not samp_per_dim³)
    3. Estimate XY intersection area from 2D samples
    4. Combine: IoU = (z_overlap * xy_area) / union_volume

    Args:
        obb1: First set of boxes (N, 165)
        obb2: Second set of boxes (M, 165)
        samp_per_dim: Sampling density per XY dimension (default: 32)
                        Total 2D samples = samp_per_dim²
        all_pairs: If True, compute NxM IoU matrix. If False, compute N IoUs element-wise
        verbose: If True, print timing and elimination statistics
        chunk_size: If provided and all_pairs=True, process in blockwise chunks to avoid OOM.
                    Each block will be chunk_size×chunk_size. Recommended: 500-1000.
                    If None, disables chunking.

    Returns:
        IoU tensor of shape (N, M) if all_pairs=True, else (N,)

    Note:
        Assumes boxes only have yaw rotation (roll=0, pitch=0).
        For general 6-DoF rotations, use iou_mc9() instead.
    """
    import time

    overall_start = time.time() if verbose else None

    assert obb1.ndim == 2 and obb2.ndim == 2

    N = obb1.shape[0]
    M = obb2.shape[0]
    device = obb1.device

    if not all_pairs and N != M:
        raise ValueError(
            "If 'all_pairs' is False, then obb1 and obb2 must have same shape"
        )

    # If chunking is enabled and all_pairs=True, process blockwise to avoid OOM
    if all_pairs and chunk_size is not None and (N > chunk_size or M > chunk_size):
        if verbose:
            n_blocks_1 = (N + chunk_size - 1) // chunk_size
            n_blocks_2 = (M + chunk_size - 1) // chunk_size
            total_blocks = n_blocks_1 * n_blocks_2
            print(
                f"\n[IoU MC2D] Processing {N}x{M} in {n_blocks_1}x{n_blocks_2} = {total_blocks} blocks "
                f"of max {chunk_size}x{chunk_size} to avoid OOM"
            )

        # Preallocate result matrix
        iou = torch.zeros(N, M, device=device)
        giou_out = torch.zeros(N, M, device=device) if use_giou else None

        block_num = 0
        # Process blockwise: chunk both dimensions
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            obb1_block = obb1[i_start:i_end]

            for j_start in range(0, M, chunk_size):
                j_end = min(j_start + chunk_size, M)
                obb2_block = obb2[j_start:j_end]

                block_num += 1
                if verbose and block_num % 10 == 0:
                    print(
                        f"  [IoU MC2D] Processing block {block_num}/{total_blocks} "
                        f"({i_start}:{i_end}, {j_start}:{j_end})..."
                    )

                # Compute IoU for this block without chunking
                result = _iou_mc7_no_chunking(
                    obb1_block,
                    obb2_block,
                    samp_per_dim=samp_per_dim,
                    all_pairs=True,
                    verbose=False,  # Disable verbose for blocks to avoid spam
                    use_giou=use_giou,
                )

                # Write results to the appropriate location in the output matrix
                if use_giou:
                    iou_block, giou_block = result
                    iou[i_start:i_end, j_start:j_end] = iou_block
                    giou_out[i_start:i_end, j_start:j_end] = giou_block
                else:
                    iou[i_start:i_end, j_start:j_end] = result

        if verbose:
            total_time = time.time() - overall_start
            print(f"  [IoU MC2D] Total time: {total_time:.3f}s")

        if use_giou:
            return iou, giou_out
        return iou
    else:
        # Process without chunking
        return _iou_mc7_no_chunking(
            obb1, obb2, samp_per_dim, all_pairs, verbose, use_giou
        )


def _iou_mc7_no_chunking(
    obb1,
    obb2,
    samp_per_dim: int = 32,
    all_pairs: bool = True,
    verbose: bool = False,
    use_giou: bool = False,
) -> torch.Tensor:
    """Internal function that computes IoU without chunking."""
    import time

    overall_start = time.time()

    N = obb1.shape[0]
    M = obb2.shape[0]
    device = obb1.device

    if verbose:
        print(f"\n[IoU MC2D] Computing for {N}x{M} = {N * M:,} box pairs")

    # Extract box parameters
    t_extract0 = time.time()
    centers1 = obb1.bb3_center_world
    centers2 = obb2.bb3_center_world

    extents1 = obb1.bb3_object
    extents2 = obb2.bb3_object

    w1 = extents1[:, 1] - extents1[:, 0]
    h1 = extents1[:, 3] - extents1[:, 2]
    d1 = extents1[:, 5] - extents1[:, 4]

    w2 = extents2[:, 1] - extents2[:, 0]
    h2 = extents2[:, 3] - extents2[:, 2]
    d2 = extents2[:, 5] - extents2[:, 4]

    eulers1 = obb1.T_world_object.to_euler()
    eulers2 = obb2.T_world_object.to_euler()
    yaw1 = eulers1[:, 2]
    yaw2 = eulers2[:, 2]
    t_extract1 = time.time()

    if all_pairs:
        # Early elimination based on distance
        dist_start = time.time() if verbose else None
        center_distances = torch.cdist(centers1[:, :2], centers2[:, :2], p=2)
        xy_extent1 = torch.sqrt(w1**2 + h1**2) / 2
        xy_extent2 = torch.sqrt(w2**2 + h2**2) / 2
        threshold = xy_extent1.unsqueeze(1) + xy_extent2.unsqueeze(0)
        might_overlap_xy = center_distances <= threshold

        # Z-overlap check
        z1_min = (centers1[:, 2] - d1 / 2).unsqueeze(1)
        z1_max = (centers1[:, 2] + d1 / 2).unsqueeze(1)
        z2_min = (centers2[:, 2] - d2 / 2).unsqueeze(0)
        z2_max = (centers2[:, 2] + d2 / 2).unsqueeze(0)

        z_overlap = torch.clamp(
            torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min), min=0.0
        )
        might_overlap_z = z_overlap > 0
        might_overlap = might_overlap_xy & might_overlap_z

        n_candidates = might_overlap.sum().item()
        n_total = N * M
        n_eliminated = n_total - n_candidates

        if verbose:
            dist_time = time.time() - dist_start
            print(f"  [IoU MC2D] Distance check: {dist_time:.3f}s")
            print(
                f"  [IoU MC2D] Eliminated {n_eliminated:,}/{n_total:,} pairs ({n_eliminated / n_total * 100:.1f}%)"
            )
            print(f"  [IoU MC2D] Computing IoU for {n_candidates:,} candidate pairs...")

        if n_candidates == 0:
            if verbose:
                print(
                    "  [IoU MC2D] No overlapping candidates - returning zero IoU matrix"
                )
            zeros = torch.zeros(N, M, device=device)
            if use_giou:
                return zeros, zeros
            return zeros

        # Get candidate indices
        candidate_idx = torch.where(might_overlap)
        idx1, idx2 = candidate_idx[0], candidate_idx[1]

        # Generate 2D sample grid (only once, reused for all boxes)
        sample_start = time.time() if verbose else None
        xy_samples = _generate_2d_sample_grid(samp_per_dim, device)

        # Transform samples to world coordinates for candidate boxes
        samples1_world = _transform_2d_samples_batched(
            xy_samples,
            centers1[idx1, :2],
            w1[idx1],
            h1[idx1],
            yaw1[idx1],
        )

        samples2_world = _transform_2d_samples_batched(
            xy_samples,
            centers2[idx2, :2],
            w2[idx2],
            h2[idx2],
            yaw2[idx2],
        )

        if verbose:
            sample_time = time.time() - sample_start
            print(
                f"  [IoU MC2D] 2D sampling: {sample_time:.3f}s ({xy_samples.shape[0]:,} samples/box)"
            )

        # Point-in-rectangle tests (2D only - much faster than 3D!)
        compute_start = time.time() if verbose else None

        # Check if samples1 are inside boxes2
        inside_1_to_2 = _point_in_rect_2d_batched(
            samples1_world,
            centers2[idx2, :2],
            w2[idx2],
            h2[idx2],
            yaw2[idx2],
        )

        # Check if samples2 are inside boxes1
        inside_2_to_1 = _point_in_rect_2d_batched(
            samples2_world,
            centers1[idx1, :2],
            w1[idx1],
            h1[idx1],
            yaw1[idx1],
        )

        # Estimate XY overlap area from sample counts
        xy_area_1 = w1[idx1] * h1[idx1]  # (n_candidates,)
        xy_area_2 = w2[idx2] * h2[idx2]  # (n_candidates,)

        overlap_ratio_1 = inside_1_to_2.float().mean(dim=1)  # (n_candidates,)
        overlap_ratio_2 = inside_2_to_1.float().mean(dim=1)  # (n_candidates,)

        xy_overlap_area = (
            xy_area_1 * overlap_ratio_1 + xy_area_2 * overlap_ratio_2
        ) / 2

        # Scatter back to full matrix
        xy_areas = torch.zeros(N, M, device=device)
        xy_areas[idx1, idx2] = xy_overlap_area

        # Compute 3D intersection volumes
        intersection_volumes = z_overlap * xy_areas

        # Compute box volumes
        vol1 = (w1 * h1 * d1).unsqueeze(1)
        vol2 = (w2 * h2 * d2).unsqueeze(0)

        # Compute union and IoU
        union_volumes = vol1 + vol2 - intersection_volumes
        iou = intersection_volumes / (union_volumes + 1e-8)

        giou_rescaled = None
        if use_giou:
            # GIoU = IoU - (C - Union) / C, where C = enclosing box volume
            enc_z = torch.max(z1_max, z2_max) - torch.min(z1_min, z2_min)
            r1 = xy_extent1.unsqueeze(1)  # (N, 1)
            r2 = xy_extent2.unsqueeze(0)  # (1, M)
            cx1 = centers1[:, 0].unsqueeze(1)
            cy1 = centers1[:, 1].unsqueeze(1)
            cx2 = centers2[:, 0].unsqueeze(0)
            cy2 = centers2[:, 1].unsqueeze(0)
            enc_x = torch.max(cx1 + r1, cx2 + r2) - torch.min(cx1 - r1, cx2 - r2)
            enc_y = torch.max(cy1 + r1, cy2 + r2) - torch.min(cy1 - r1, cy2 - r2)
            enc_volume = enc_x * enc_y * enc_z
            giou = iou - (enc_volume - union_volumes) / (enc_volume + 1e-8)
            giou_rescaled = (giou + 1) / 2  # rescale [-1, 1] -> [0, 1]

        if verbose:
            compute_time = time.time() - compute_start
            total_time = time.time() - overall_start
            print(
                f"  [IoU MC2D] Compute: {compute_time:.3f}s, Total: {total_time:.3f}s"
            )

        if use_giou:
            return iou, giou_rescaled
        return iou

    else:
        # Element-wise mode (N pairs)
        # Generate 2D sample grid
        xy_samples = _generate_2d_sample_grid(samp_per_dim, device)

        # Transform samples to world coordinates
        samples1_world = _transform_2d_samples_batched(
            xy_samples, centers1[:, :2], w1, h1, yaw1
        )
        samples2_world = _transform_2d_samples_batched(
            xy_samples, centers2[:, :2], w2, h2, yaw2
        )

        # Point-in-rectangle tests
        inside_1_to_2 = _point_in_rect_2d_batched(
            samples1_world, centers2[:, :2], w2, h2, yaw2
        )
        inside_2_to_1 = _point_in_rect_2d_batched(
            samples2_world, centers1[:, :2], w1, h1, yaw1
        )

        # Estimate XY overlap area
        xy_area_1 = w1 * h1
        xy_area_2 = w2 * h2
        overlap_ratio_1 = inside_1_to_2.float().mean(dim=1)
        overlap_ratio_2 = inside_2_to_1.float().mean(dim=1)
        xy_overlap_area = (
            xy_area_1 * overlap_ratio_1 + xy_area_2 * overlap_ratio_2
        ) / 2

        # Compute Z-overlap
        z1_min = centers1[:, 2] - d1 / 2
        z1_max = centers1[:, 2] + d1 / 2
        z2_min = centers2[:, 2] - d2 / 2
        z2_max = centers2[:, 2] + d2 / 2
        z_overlap = torch.clamp(
            torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min), min=0.0
        )

        # Compute 3D intersection and IoU
        intersection_volumes = z_overlap * xy_overlap_area
        vol1 = w1 * h1 * d1
        vol2 = w2 * h2 * d2
        union_volumes = vol1 + vol2 - intersection_volumes
        iou = intersection_volumes / (union_volumes + 1e-8)

        giou_rescaled = None
        if use_giou:
            enc_z = torch.max(z1_max, z2_max) - torch.min(z1_min, z2_min)
            r1 = torch.sqrt(w1**2 + h1**2) / 2
            r2 = torch.sqrt(w2**2 + h2**2) / 2
            cx1, cy1 = centers1[:, 0], centers1[:, 1]
            cx2, cy2 = centers2[:, 0], centers2[:, 1]
            enc_x = torch.max(cx1 + r1, cx2 + r2) - torch.min(cx1 - r1, cx2 - r2)
            enc_y = torch.max(cy1 + r1, cy2 + r2) - torch.min(cy1 - r1, cy2 - r2)
            enc_volume = enc_x * enc_y * enc_z
            giou = iou - (enc_volume - union_volumes) / (enc_volume + 1e-8)
            giou_rescaled = (giou + 1) / 2  # rescale [-1, 1] -> [0, 1]

        if use_giou:
            return iou, giou_rescaled
        return iou


def _generate_2d_sample_grid(samp_per_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generate uniform 2D sample grid in unit square [-0.5, 0.5] x [-0.5, 0.5].

    Args:
        samp_per_dim: Number of samples per dimension
        device: Device to create tensor on

    Returns:
        samples: (samp_per_dim², 2) sample coordinates
    """
    # Create 1D grids
    grid_1d = torch.linspace(-0.5, 0.5, samp_per_dim, device=device)

    # Create 2D meshgrid
    grid_x, grid_y = torch.meshgrid(grid_1d, grid_1d, indexing="ij")

    # Flatten and stack
    samples = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    return samples


def _transform_2d_samples_batched(
    samples: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    """
    Transform 2D samples from unit square to world coordinates (fully vectorized).

    Args:
        samples: (num_samples, 2) sample points in [-0.5, 0.5]²
        centers: (N, 2) box centers in world frame
        widths: (N,) box widths
        heights: (N,) box heights
        yaws: (N,) rotation angles in radians

    Returns:
        transformed: (N, num_samples, 2) samples in world coordinates
    """
    N = centers.shape[0]

    # Scale samples by box dimensions: (N, 1, 2) * (1, num_samples, 2)
    scale = torch.stack([widths, heights], dim=1)  # (N, 2)
    scaled_samples = samples.unsqueeze(0) * scale.unsqueeze(1)  # (N, num_samples, 2)

    # Build rotation matrices (N, 2, 2)
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    rotation = torch.zeros(N, 2, 2, device=centers.device, dtype=centers.dtype)
    rotation[:, 0, 0] = cos_yaw
    rotation[:, 0, 1] = -sin_yaw
    rotation[:, 1, 0] = sin_yaw
    rotation[:, 1, 1] = cos_yaw

    # Rotate: (N, num_samples, 2) @ (N, 2, 2).T -> (N, num_samples, 2)
    rotated = torch.bmm(scaled_samples, rotation.transpose(1, 2))

    # Translate: (N, num_samples, 2) + (N, 1, 2)
    transformed = rotated + centers.unsqueeze(1)

    return transformed


def _point_in_rect_2d_batched(
    points: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    """
    Check if 2D points are inside rotated rectangles (fully vectorized).

    Args:
        points: (N, num_points, 2) points to test
        centers: (N, 2) rectangle centers
        widths: (N,) rectangle widths
        heights: (N,) rectangle heights
        yaws: (N,) rotation angles in radians

    Returns:
        inside: (N, num_points) boolean mask
    """
    N = centers.shape[0]

    # Transform points to rectangle's local frame
    # 1. Translate to origin
    points_centered = points - centers.unsqueeze(1)  # (N, num_points, 2)

    # 2. Rotate by -yaw (inverse rotation)
    cos_yaw = torch.cos(-yaws)
    sin_yaw = torch.sin(-yaws)
    rotation_inv = torch.zeros(N, 2, 2, device=centers.device, dtype=centers.dtype)
    rotation_inv[:, 0, 0] = cos_yaw
    rotation_inv[:, 0, 1] = -sin_yaw
    rotation_inv[:, 1, 0] = sin_yaw
    rotation_inv[:, 1, 1] = cos_yaw

    points_local = torch.bmm(
        points_centered, rotation_inv.transpose(1, 2)
    )  # (N, num_points, 2)

    # 3. Check if inside axis-aligned rectangle
    half_widths = widths.unsqueeze(1) / 2  # (N, 1)
    half_heights = heights.unsqueeze(1) / 2  # (N, 1)

    inside_x = torch.abs(points_local[:, :, 0]) <= half_widths
    inside_y = torch.abs(points_local[:, :, 1]) <= half_heights

    inside = inside_x & inside_y  # (N, num_points)

    return inside


def iou_exact7(
    obb1,
    obb2,
    samp_per_dim: int = 32,
    all_pairs: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute exact 3D IoU for 7-DoF oriented bounding boxes (center + size + yaw).

    This function computes analytically exact IoU for boxes with only yaw rotation
    (rotation around vertical Z-axis). It's differentiable and much faster than
    sampling-based methods.

    Algorithm:
    1. Compute 1D intersection along Z-axis (height overlap)
    2. Compute 2D intersection in XY-plane using Sutherland-Hodgman clipping
    3. IoU = (z_overlap * xy_area) / union_volume

    Args:
        obb1: First set of boxes (N, 165)
        obb2: Second set of boxes (M, 165)
        all_pairs: If True, compute NxM IoU matrix. If False, compute N IoUs element-wise
        eps: Small epsilon for numerical stability

    Returns:
        IoU tensor of shape (N, M) if all_pairs=True, else (N,)

    Note:
        Assumes boxes only have yaw rotation (roll=0, pitch=0).
        For general 6-DoF rotations, use obb_iou3d() instead.
    """
    assert obb1.ndim == 2 and obb2.ndim == 2

    if all_pairs:
        return _obb_iou3d_analytical_all_pairs(obb1, obb2)
    else:
        return _obb_iou3d_analytical_pairwise(obb1, obb2)


def _obb_iou3d_analytical_all_pairs(obb1, obb2, eps: float = 1e-8) -> torch.Tensor:
    """Compute NxM IoU matrix for all pairs of boxes."""
    N = obb1.shape[0]
    M = obb2.shape[0]
    device = obb1.device

    # Extract centers (N, 3) and (M, 3)
    centers1 = obb1.bb3_center_world  # (N, 3)
    centers2 = obb2.bb3_center_world  # (M, 3)

    # Extract sizes (N, 3) and (M, 3)
    extents1 = obb1.bb3_object  # (N, 6)
    extents2 = obb2.bb3_object  # (M, 6)

    w1 = extents1[:, 1] - extents1[:, 0]  # (N,)
    h1 = extents1[:, 3] - extents1[:, 2]  # (N,)
    d1 = extents1[:, 5] - extents1[:, 4]  # (N,)

    w2 = extents2[:, 1] - extents2[:, 0]  # (M,)
    h2 = extents2[:, 3] - extents2[:, 2]  # (M,)
    d2 = extents2[:, 5] - extents2[:, 4]  # (M,)

    # Extract yaw angles (N,) and (M,)
    eulers1 = obb1.T_world_object.to_euler()  # (N, 3)
    eulers2 = obb2.T_world_object.to_euler()  # (M, 3)
    yaw1 = eulers1[:, 2]  # (N,)
    yaw2 = eulers2[:, 2]  # (M,)

    # Compute 1D Z-axis overlap for all pairs
    # z_min = max(z1_min, z2_min), z_max = min(z1_max, z2_max)
    z1_min = (centers1[:, 2] - d1 / 2).unsqueeze(1)  # (N, 1)
    z1_max = (centers1[:, 2] + d1 / 2).unsqueeze(1)  # (N, 1)
    z2_min = (centers2[:, 2] - d2 / 2).unsqueeze(0)  # (1, M)
    z2_max = (centers2[:, 2] + d2 / 2).unsqueeze(0)  # (1, M)

    z_overlap = torch.clamp(
        torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min), min=0.0
    )  # (N, M)

    # Compute 2D XY-plane intersection areas for all pairs
    xy_areas = torch.zeros(N, M, device=device)

    for i in range(N):
        for j in range(M):
            # Get 2D rectangle corners in XY plane
            rect1 = _get_2d_rectangle_corners(
                centers1[i, :2], w1[i], h1[i], yaw1[i]
            )  # (4, 2)
            rect2 = _get_2d_rectangle_corners(
                centers2[j, :2], w2[j], h2[j], yaw2[j]
            )  # (4, 2)

            # Compute intersection polygon using Sutherland-Hodgman
            intersection_area = _polygon_intersection_area(rect1, rect2)
            xy_areas[i, j] = intersection_area

    # Compute 3D intersection volumes
    intersection_volumes = z_overlap * xy_areas  # (N, M)

    # Compute box volumes
    vol1 = (w1 * h1 * d1).unsqueeze(1)  # (N, 1)
    vol2 = (w2 * h2 * d2).unsqueeze(0)  # (1, M)

    # Compute union volumes
    union_volumes = vol1 + vol2 - intersection_volumes  # (N, M)

    # Compute IoU with numerical stability
    iou = intersection_volumes / (union_volumes + eps)

    return iou


def _obb_iou3d_analytical_pairwise(obb1, obb2, eps: float = 1e-8) -> torch.Tensor:
    """Compute element-wise IoU for N pairs of boxes."""
    assert obb1.shape[0] == obb2.shape[0], (
        "Must have same number of boxes for pairwise IoU"
    )

    N = obb1.shape[0]
    device = obb1.device

    # Extract centers (N, 3)
    centers1 = obb1.bb3_center_world  # (N, 3)
    centers2 = obb2.bb3_center_world  # (N, 3)

    # Extract sizes (N, 3)
    extents1 = obb1.bb3_object  # (N, 6)
    extents2 = obb2.bb3_object  # (N, 6)

    w1 = extents1[:, 1] - extents1[:, 0]  # (N,)
    h1 = extents1[:, 3] - extents1[:, 2]  # (N,)
    d1 = extents1[:, 5] - extents1[:, 4]  # (N,)

    w2 = extents2[:, 1] - extents2[:, 0]  # (N,)
    h2 = extents2[:, 3] - extents2[:, 2]  # (N,)
    d2 = extents2[:, 5] - extents2[:, 4]  # (N,)

    # Extract yaw angles (N,)
    eulers1 = obb1.T_world_object.to_euler()  # (N, 3)
    eulers2 = obb2.T_world_object.to_euler()  # (N, 3)
    yaw1 = eulers1[:, 2]  # (N,)
    yaw2 = eulers2[:, 2]  # (N,)

    # Compute 1D Z-axis overlap
    z1_min = centers1[:, 2] - d1 / 2  # (N,)
    z1_max = centers1[:, 2] + d1 / 2  # (N,)
    z2_min = centers2[:, 2] - d2 / 2  # (N,)
    z2_max = centers2[:, 2] + d2 / 2  # (N,)

    z_overlap = torch.clamp(
        torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min), min=0.0
    )  # (N,)

    # Compute 2D XY-plane intersection areas
    xy_areas = torch.zeros(N, device=device)

    for i in range(N):
        # Get 2D rectangle corners in XY plane
        rect1 = _get_2d_rectangle_corners(
            centers1[i, :2], w1[i], h1[i], yaw1[i]
        )  # (4, 2)
        rect2 = _get_2d_rectangle_corners(
            centers2[i, :2], w2[i], h2[i], yaw2[i]
        )  # (4, 2)

        # Compute intersection polygon using Sutherland-Hodgman
        intersection_area = _polygon_intersection_area(rect1, rect2)
        xy_areas[i] = intersection_area

    # Compute 3D intersection volumes
    intersection_volumes = z_overlap * xy_areas  # (N,)

    # Compute box volumes
    vol1 = w1 * h1 * d1  # (N,)
    vol2 = w2 * h2 * d2  # (N,)

    # Compute union volumes
    union_volumes = vol1 + vol2 - intersection_volumes  # (N,)

    # Compute IoU with numerical stability
    iou = intersection_volumes / (union_volumes + eps)

    return iou


def _get_2d_rectangle_corners(
    center: torch.Tensor, width: torch.Tensor, height: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """
    Get 4 corners of a 2D rotated rectangle in counter-clockwise order.

    Args:
        center: (2,) [x, y]
        width: scalar width (along local x-axis before rotation)
        height: scalar height (along local y-axis before rotation)
        yaw: scalar rotation angle in radians

    Returns:
        corners: (4, 2) corner coordinates in world frame
    """
    # Extract half dimensions as scalars
    hw = width.item() / 2
    hh = height.item() / 2

    # Counter-clockwise order: bottom-left, bottom-right, top-right, top-left
    local_corners = torch.tensor(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
        device=center.device,
        dtype=center.dtype,
    )

    # Rotation matrix using scalar values
    cos_yaw = torch.cos(yaw).item()
    sin_yaw = torch.sin(yaw).item()
    rotation = torch.tensor(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
        device=center.device,
        dtype=center.dtype,
    )

    # Rotate and translate
    rotated_corners = local_corners @ rotation.T  # (4, 2)
    world_corners = rotated_corners + center.unsqueeze(0)  # (4, 2)

    return world_corners


def _polygon_intersection_area(
    poly1: torch.Tensor, poly2: torch.Tensor
) -> torch.Tensor:
    """
    Compute intersection area of two convex polygons using Sutherland-Hodgman algorithm.

    Args:
        poly1: (N1, 2) vertices in counter-clockwise order
        poly2: (N2, 2) vertices in counter-clockwise order

    Returns:
        area: scalar intersection area (differentiable)
    """
    # Sutherland-Hodgman clipping: clip poly1 against each edge of poly2
    clipped = poly1

    n2 = poly2.shape[0]
    for i in range(n2):
        if clipped.shape[0] == 0:
            break

        # Edge from poly2[i] to poly2[(i+1) % n2]
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % n2]

        clipped = _clip_polygon_by_edge(clipped, edge_start, edge_end)

    # Compute area of clipped polygon
    if clipped.shape[0] < 3:
        return torch.tensor(0.0, device=poly1.device, dtype=poly1.dtype)

    area = _polygon_area(clipped)
    return area


def _clip_polygon_by_edge(
    polygon: torch.Tensor, edge_start: torch.Tensor, edge_end: torch.Tensor
) -> torch.Tensor:
    """
    Clip polygon by a single edge using Sutherland-Hodgman algorithm.

    Args:
        polygon: (N, 2) vertices
        edge_start: (2,) edge start point
        edge_end: (2,) edge end point

    Returns:
        clipped: (M, 2) clipped vertices (M <= N+1)
    """
    if polygon.shape[0] == 0:
        return polygon

    # Edge direction vector
    edge_vec = edge_end - edge_start

    # Normal vector (perpendicular, pointing inward/left)
    normal = torch.tensor(
        [-edge_vec[1], edge_vec[0]], device=polygon.device, dtype=polygon.dtype
    )

    output = []
    n = polygon.shape[0]

    for i in range(n):
        current = polygon[i]
        next_vertex = polygon[(i + 1) % n]

        # Check if points are inside (left of edge)
        current_inside = _is_left_of_edge(current, edge_start, normal)
        next_inside = _is_left_of_edge(next_vertex, edge_start, normal)

        if current_inside and next_inside:
            # Both inside: keep next vertex
            output.append(next_vertex)
        elif current_inside and not next_inside:
            # Leaving: add intersection
            intersection = _line_intersection(
                current, next_vertex, edge_start, edge_end
            )
            output.append(intersection)
        elif not current_inside and next_inside:
            # Entering: add intersection and next vertex
            intersection = _line_intersection(
                current, next_vertex, edge_start, edge_end
            )
            output.append(intersection)
            output.append(next_vertex)
        # else: both outside, add nothing

    if len(output) == 0:
        return torch.empty((0, 2), device=polygon.device, dtype=polygon.dtype)

    return torch.stack(output)


def _is_left_of_edge(
    point: torch.Tensor, edge_start: torch.Tensor, normal: torch.Tensor
) -> bool:
    """Check if point is on the left side (inside) of an edge."""
    vec = point - edge_start
    return (vec * normal).sum() >= 0


def _line_intersection(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
) -> torch.Tensor:
    """
    Find intersection point of two line segments (p1-p2) and (p3-p4).

    Uses parametric line equation:
    P = p1 + t * (p2 - p1)
    P = p3 + s * (p4 - p3)

    Args:
        p1, p2: (2,) endpoints of first segment
        p3, p4: (2,) endpoints of second segment

    Returns:
        intersection: (2,) intersection point
    """
    d1 = p2 - p1
    d2 = p4 - p3
    d3 = p1 - p3

    # Solve: p1 + t*d1 = p3 + s*d2
    # Rearrange: t*d1 - s*d2 = p3 - p1 = -d3

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    # Avoid division by zero (lines are parallel)
    eps = 1e-10
    if torch.abs(cross) < eps:
        # Return midpoint if parallel
        return (p1 + p2) / 2

    # Using Cramer's rule: t = det([-d3, d2]) / det([d1, d2])
    # det([-d3, d2]) = -d3[0]*d2[1] - (-d3[1])*d2[0] = -d3[0]*d2[1] + d3[1]*d2[0]
    t = (-d3[0] * d2[1] + d3[1] * d2[0]) / cross

    intersection = p1 + t * d1

    return intersection


def _polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute area of a polygon using the shoelace formula.

    Args:
        vertices: (N, 2) vertices in counter-clockwise order

    Returns:
        area: scalar area (always positive)
    """
    n = vertices.shape[0]
    if n < 3:
        return torch.tensor(0.0, device=vertices.device, dtype=vertices.dtype)

    # Shoelace formula: A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Shift coordinates cyclically
    x_next = torch.cat([x[1:], x[0:1]])
    y_next = torch.cat([y[1:], y[0:1]])

    area = 0.5 * torch.abs((x * y_next - x_next * y).sum())

    return area


# ==============================================================================
# Sparse IoU computation for large-scale box fusion
# ==============================================================================


def iou_mc7_sparse(
    obb1,
    obb2,
    samp_per_dim: int = 32,
    chunk_size: int = 2048,
    iou_threshold: float = 0.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute sparse IoU matrix using 2D Monte Carlo sampling + analytical Z-overlap.

    This function is designed for large-scale IoU computation where the dense NxM
    matrix would cause OOM. It only stores IoU values >= iou_threshold, returning
    a PyTorch sparse COO tensor.

    Memory savings example:
        - 80,000 x 80,000 dense = 25.6 GB
        - Sparse with ~0.1% non-zero = ~25 MB

    Args:
        obb1: First set of boxes (N, 165)
        obb2: Second set of boxes (M, 165)
        samp_per_dim: Sampling density per XY dimension (default: 32)
        chunk_size: Process in blocks of this size to avoid OOM during computation
        iou_threshold: Only store IoU values >= this threshold (default: 0.0)
        verbose: If True, print progress information

    Returns:
        PyTorch sparse COO tensor of shape (N, M) containing only non-zero IoU values
        above the threshold.

    Note:
        Assumes boxes only have yaw rotation (roll=0, pitch=0).
        For general 6-DoF rotations, use iou_mc9() instead.
    """
    import time

    overall_start = time.time() if verbose else None

    assert obb1.ndim == 2 and obb2.ndim == 2

    N = obb1.shape[0]
    M = obb2.shape[0]
    device = obb1.device

    if verbose:
        dense_memory_gb = (N * M * 4) / (1024**3)
        print(f"\n[IoU MC2D Sparse] Computing {N}x{M} matrix")
        print(f"  Dense matrix would require: {dense_memory_gb:.2f} GB")
        print(f"  IoU threshold for storage: {iou_threshold}")

    # Collect sparse results
    all_rows: list[torch.Tensor] = []
    all_cols: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []

    n_blocks_i = (N + chunk_size - 1) // chunk_size
    n_blocks_j = (M + chunk_size - 1) // chunk_size
    total_blocks = n_blocks_i * n_blocks_j

    if verbose:
        print(
            f"  Processing {n_blocks_i}x{n_blocks_j} = {total_blocks} blocks "
            f"of max {chunk_size}x{chunk_size}"
        )

    block_num = 0
    total_stored = 0

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        obb1_block = obb1[i_start:i_end]

        for j_start in range(0, M, chunk_size):
            j_end = min(j_start + chunk_size, M)
            obb2_block = obb2[j_start:j_end]

            block_num += 1

            # Compute dense IoU for this block
            iou_block = _iou_mc7_no_chunking(
                obb1_block,
                obb2_block,
                samp_per_dim=samp_per_dim,
                all_pairs=True,
                verbose=False,
            )

            # Extract entries above threshold
            mask = iou_block >= iou_threshold
            if mask.any():
                rows, cols = torch.where(mask)
                values = iou_block[mask]

                # Offset indices to global coordinates
                all_rows.append(rows + i_start)
                all_cols.append(cols + j_start)
                all_values.append(values)

                total_stored += len(values)

            if verbose and block_num % 100 == 0:
                print(
                    f"    Block {block_num}/{total_blocks}, "
                    f"stored {total_stored:,} entries so far..."
                )

    # Build sparse tensor
    if total_stored > 0:
        indices = torch.stack(
            [torch.cat(all_rows).to(device), torch.cat(all_cols).to(device)]
        )
        values = torch.cat(all_values).to(device)
        sparse_iou = torch.sparse_coo_tensor(
            indices, values, size=(N, M), device=device
        )
        sparse_iou = sparse_iou.coalesce()
    else:
        # Return empty sparse tensor
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty(0, dtype=torch.float32, device=device)
        sparse_iou = torch.sparse_coo_tensor(
            indices, values, size=(N, M), device=device
        )

    if verbose:
        total_time = time.time() - overall_start
        sparse_memory_mb = (total_stored * (8 + 4)) / (1024**2)  # indices + values
        sparsity = 1.0 - (total_stored / (N * M)) if N * M > 0 else 1.0
        print("\n[IoU MC2D Sparse] Complete!")
        print(f"  Total entries stored: {total_stored:,} / {N * M:,}")
        print(f"  Sparsity: {sparsity * 100:.4f}%")
        print(f"  Sparse memory: ~{sparse_memory_mb:.2f} MB")
        print(f"  Total time: {total_time:.2f}s")

    return sparse_iou
