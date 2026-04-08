# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from utils.tw.obb import PAD_VAL, ObbTW, iou_mc9, make_obb
from utils.tw.pose import PoseTW

torch.set_printoptions(precision=5, sci_mode=False)


class TestObb(unittest.TestCase):
    def test_constructor(self):
        bb2 = np.array([2.0, 3.0, 102.0, 103.0]).astype(np.float32)
        bb3_o = np.array([0.0, 0.0, 0.0, 2.0, 3.0, 4.0]).astype(np.float32)
        T_wo = PoseTW()
        obb = ObbTW.from_lmc(bb3_o, bb2, moveable=1, prob=[0.5])
        obb = ObbTW.from_lmc(bb3_o, bb2, None, None, T_wo, 0)
        self.assertTrue(torch.allclose(obb.bb2_rgb, torch.tensor(bb2)))
        self.assertTrue(torch.allclose(obb.bb2_slaml, PAD_VAL * torch.ones(4)))
        self.assertTrue(torch.allclose(obb.bb2_slamr, PAD_VAL * torch.ones(4)))

        # Test list input.
        bb3_o_list = [0, 0, 0, 1, 1, 1]
        obb = ObbTW.from_lmc(bb3_o_list)
        self.assertTrue(obb.bb3_object.tolist() == bb3_o_list)
        obb = ObbTW.from_lmc([0, 0, 0, 1, 1, 1])
        obb = ObbTW.from_lmc([0, 0, 0, 1, 1, 1], moveable=1)

        # Should we handle this?
        bb3_o = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).astype(np.float32)
        bb3_o = bb3_o[None, ...]
        obb = ObbTW.from_lmc(bb3_o, moveable=[1])

    def test_inside(self):
        obb = make_obb(sz=[1.0, 1.0, 1.0], position=[0.0, 0.0, 0.0], yaw=0.0)[None]
        obb = obb.reshape(-1)
        pts = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3)
        self.assertTrue(obb.points_inside_bb3(pts))

    def test_voxel_grid(self):
        """
        Test that voxel_grid():
          - returns correct shape
          - samples centers uniformly
          - applies T_world_object correctly
        """

        # ------------------------------
        # Construct a simple OBB (axis-aligned, no rotation)
        # ------------------------------
        # bb3_object = (x_min, x_max, y_min, y_max, z_min, z_max)
        bb3 = torch.tensor(
            [
                [
                    0.0,
                    1.0,  # x
                    2.0,
                    4.0,  # y
                    5.0,
                    6.0,  # z
                ]
            ],
            dtype=torch.float32,
        )  # (1, 6)
        bb3 = bb3.reshape(-1)

        T_world_object = PoseTW()

        # Make the ObbTW
        obb = ObbTW.from_lmc(bb3, None, None, None, T_world_object)
        obb = obb.reshape(1, -1)

        # Sample resolution
        vW, vH, vD = 2, 2, 2  # small grid: 8 points

        # ------------------------------
        # Call voxel_grid
        # ------------------------------
        pts = obb.voxel_grid(vD=vD, vH=vH, vW=vW)
        # Expect shape (1, 8, 3)
        self.assertEqual(list(pts.shape), [1, vW * vH * vD, 3])

        pts = pts[0]  # (8, 3)

        # ------------------------------
        # Expected grid centers
        # ------------------------------
        # X from 0 to 1 with centers at 0.25, 0.75
        xs_exp = torch.tensor([0.25, 0.75])
        ys_exp = torch.tensor([2.5, 3.5])
        zs_exp = torch.tensor([5.25, 5.75])

        # Build all (x,y,z) combos in ij indexing order
        X, Y, Z = torch.meshgrid(xs_exp, ys_exp, zs_exp, indexing="ij")
        expected = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

        # ------------------------------
        # Compare
        # ------------------------------
        self.assertTrue(torch.allclose(pts, expected, atol=1e-6))

    def test_voxel_grid_batched(self):
        """
        Test that voxel_grid() works correctly for B > 1 batches:
          - correct shape
          - correct uniform voxel centers per batch
          - independent box boundaries per batch
          - correct application of T_world_object (identity)
        """

        # --------------------------------------------------------
        # Construct two simple axis-aligned OBBs
        # --------------------------------------------------------
        bb3_batched = torch.tensor(
            [
                [
                    0.0,
                    1.0,  # x
                    2.0,
                    4.0,  # y
                    5.0,
                    6.0,  # z
                ],
                [
                    10.0,
                    14.0,  # x
                    -3.0,
                    1.0,  # y
                    20.0,
                    22.0,  # z
                ],
            ],
            dtype=torch.float32,
        )  # (2,6)

        # Flatten last dim so ObbTW.from_lmc(...) matches your usage
        bb3_batched = bb3_batched.reshape(2, -1)

        # Identity transforms for both OBBs
        T_world_object = PoseTW()
        T_world_object = T_world_object.reshape(1, -1).expand(2, -1)

        # Build ObbTW from your factory, maintaining batch shape
        obb = ObbTW.from_lmc(bb3_batched, None, None, None, T_world_object)
        self.assertEqual(obb.shape[0], 2)

        # Sampling resolution
        vW, vH, vD = 2, 2, 2  # → 8 points per box

        # --------------------------------------------------------
        # Call voxel_grid
        # --------------------------------------------------------
        pts = obb.voxel_grid(vD=vD, vH=vH, vW=vW)

        # Expect shape (2, 8, 3)
        self.assertEqual(list(pts.shape), [2, vW * vH * vD, 3])

        # --------------------------------------------------------
        # Expected grid centers for batch 0
        # --------------------------------------------------------
        xs0 = torch.tensor([0.25, 0.75])
        ys0 = torch.tensor([2.5, 3.5])
        zs0 = torch.tensor([5.25, 5.75])
        X0, Y0, Z0 = torch.meshgrid(xs0, ys0, zs0, indexing="ij")
        expected0 = torch.stack([X0, Y0, Z0], dim=-1).reshape(-1, 3)

        # --------------------------------------------------------
        # Expected grid centers for batch 1
        # --------------------------------------------------------
        xs1 = torch.tensor([11.0, 13.0])  # from x=10..14 → width=4 → centers at 11,13
        ys1 = torch.tensor([-2.0, 0.0])  # from y=-3..1 → width=4 → centers at -2,0
        zs1 = torch.tensor(
            [20.5, 21.5]
        )  # from z=20..22 → width=2 → centers at 20.5,21.5

        X1, Y1, Z1 = torch.meshgrid(xs1, ys1, zs1, indexing="ij")
        expected1 = torch.stack([X1, Y1, Z1], dim=-1).reshape(-1, 3)

        # --------------------------------------------------------
        # Compare batch 0
        # --------------------------------------------------------
        self.assertTrue(torch.allclose(pts[0], expected0, atol=1e-6))

        # --------------------------------------------------------
        # Compare batch 1
        # --------------------------------------------------------
        self.assertTrue(torch.allclose(pts[1], expected1, atol=1e-6))

        # Additional sanity check that the two batches differ
        self.assertFalse(torch.allclose(pts[0], pts[1], atol=1e-6))

    def test_iou3d_batch(self):
        def _make_batch(i):
            obbs = []
            for _ in range(i):
                sz = 0.1 + torch.rand((2, 3))
                pos = 0.5 * torch.rand((2, 3))
                roll = torch.rand((2,))
                pitch = torch.rand((2,))
                yaw = torch.rand((2,))
                obb = make_obb(
                    sz=sz[0, :],
                    position=pos[0, :],
                    roll=roll[0],
                    pitch=pitch[0],
                    yaw=yaw[0],
                )
                obbs.append(obb)
            obbs = torch.stack(obbs)
            return obbs

        N = 3
        M = 9
        obb1 = _make_batch(N)
        obb2 = _make_batch(M)

        print("LOOP MODE (all_pairs=False)")
        iou1 = torch.zeros((N, M))
        for n in range(N):
            for m in range(M):
                obb1n = obb1[n].reshape(1, -1).clone()
                obb2m = obb2[m].reshape(1, -1).clone()
                iou1[n, m] = iou_mc9(obb1n, obb2m, samp_per_dim=32, all_pairs=False)

        print("BATCHED MODE (all_pairs=True)")
        iou2 = iou_mc9(obb1, obb2, samp_per_dim=32, all_pairs=True)

        self.assertTrue(torch.allclose(iou1, iou2, atol=1e-6))
        print("iou_mc9: Loop is equivalent to batched mode")

    def test_iou3d_all_pairs_false(self):
        def _make_batch(i):
            obbs = []
            for _ in range(i):
                sz = 0.1 + torch.rand((2, 3))
                pos = 0.5 * torch.rand((2, 3))
                roll = torch.rand((2,))
                pitch = torch.rand((2,))
                yaw = torch.rand((2,))
                obb = make_obb(
                    sz=sz[0, :],
                    position=pos[0, :],
                    roll=roll[0],
                    pitch=pitch[0],
                    yaw=yaw[0],
                )
                obbs.append(obb)
            obbs = torch.stack(obbs)
            return obbs

        N = 4
        obb1 = _make_batch(N)
        obb2 = _make_batch(N)

        print("LOOP MODE (all_pairs=True")
        iou1 = torch.zeros(N)
        for n in range(N):
            obb1n = obb1[n].reshape(1, -1).clone()
            obb2m = obb2[n].reshape(1, -1).clone()
            iou1[n] = iou_mc9(obb1n, obb2m, samp_per_dim=32, all_pairs=True)

        print("BATCHED MODE (all_pairs=False)")
        iou2 = iou_mc9(obb1, obb2, samp_per_dim=32, all_pairs=False)

        self.assertTrue(torch.allclose(iou1, iou2, atol=1e-6))
        print("iou_mc9: Loop is equivalent to batched mode")
