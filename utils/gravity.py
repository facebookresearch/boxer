# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import numpy as np
import torch

from utils.tw.pose import PoseTW

GRAVITY_DIRECTION_VIO = np.array([0.0, 0.0, -1.0], np.float32)


def reject_vector_a_from_b(a, b):
    # https://en.wikipedia.org/wiki/Vector_projection
    b_norm = torch.sqrt((b**2).sum(-1, keepdim=True))
    b_unit = b / b_norm
    # batched dot product for variable dimensions
    a_proj = b_unit * (a * b_unit).sum(-1, keepdim=True)
    a_rej = a - a_proj
    return a_rej


def gravity_align_T_world_cam(
    T_world_cam, gravity_w=GRAVITY_DIRECTION_VIO, z_grav=False
):
    """
    get T_world_gravity from T_world_cam such that the x axis of T_world_gravity is gravity.
    """
    assert T_world_cam.dim() > 1, f"{T_world_cam} has wrong dimension; expected >1"
    dim = T_world_cam.dim()
    device = T_world_cam.device
    R_wc = T_world_cam.R
    dir_shape = [1] * (dim - 1) + [3]
    g_w = torch.from_numpy(gravity_w.copy()).view(dir_shape).to(R_wc)
    g_w = g_w.expand_as(R_wc[..., 1])
    # forward vector (z) that is orthogonal to gravity direction
    d3 = reject_vector_a_from_b(a=R_wc[..., 2], b=g_w)
    # optionally add a tiny offset to avoid cross product two identical vectors.
    d3_is_zeros = (d3 == 0.0).all(dim=-1).unsqueeze(-1).expand_as(d3)
    d3_offset = torch.zeros(*d3.shape).to(T_world_cam._data.device)
    d3_offset[..., 1] += 0.001
    d3 = torch.where(d3_is_zeros, d3 + d3_offset, d3)
    d2 = torch.linalg.cross(d3, g_w, dim=-1)
    # camera down vector is x direction since Aria cameras are rotated by 90 degree CW
    # hence the new x direction is gravity
    R_wcg = torch.cat([g_w.unsqueeze(-1), d2.unsqueeze(-1), d3.unsqueeze(-1)], -1)
    # normalize to unit length
    R_world_cg = torch.nn.functional.normalize(R_wcg, p=2, dim=-2)
    if z_grav:
        # add extra rotation to make z gravity direction, not x.
        R_cg_cgz = torch.tensor(
            [[[0, -1, 0], [0, 0, 1], [-1, 0, 0]]], dtype=torch.float32
        ).to(device)
        R_world_cgz = R_world_cg @ R_cg_cgz.inverse()
        T_world_cgz = PoseTW.from_Rt(R_world_cgz, T_world_cam.t)
        return T_world_cgz
    else:
        R_world_cg = R_world_cg
        T_world_cg = PoseTW.from_Rt(R_world_cg, T_world_cam.t)
        return T_world_cg
