# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from utils.tw.obb import iou_exact7, iou_mc9, make_obb


class TestIouExact7(unittest.TestCase):
    """Simple tests for iou_exact7 function."""

    def test_same_box_returns_one(self) -> None:
        """IoU of a box with itself should be 1.0"""
        # Setup: create a simple box
        obb = make_obb(sz=[1.0, 1.0, 1.0], position=[0.0, 0.0, 0.0], yaw=0.0)
        obb = obb.unsqueeze(0)  # Shape: (1, 165)

        # Execute: compute IoU with itself using all_pairs=True
        iou_all_pairs = iou_exact7(obb, obb, all_pairs=True)

        # Assert: IoU should be 1.0
        self.assertAlmostEqual(iou_all_pairs[0, 0].item(), 1.0, places=5)

    def test_same_box_pairwise_returns_one(self) -> None:
        """IoU of a box with itself (pairwise mode) should be 1.0"""
        # Setup: create a simple box
        obb = make_obb(sz=[1.0, 1.0, 1.0], position=[0.0, 0.0, 0.0], yaw=0.0)
        obb = obb.unsqueeze(0)  # Shape: (1, 165)

        # Execute: compute IoU with itself using all_pairs=False
        iou_pairwise = iou_exact7(obb, obb, all_pairs=False)

        # Assert: IoU should be 1.0
        self.assertAlmostEqual(iou_pairwise[0].item(), 1.0, places=5)

    def test_rotated_boxes_small_angle(self) -> None:
        """Test IoU of slightly rotated boxes"""
        # Setup: create two boxes with small rotation difference
        obb1 = make_obb(sz=[0.5, 0.2, 0.1], position=[-0.1, 0.2, 0.4], yaw=0.0)
        obb2 = make_obb(sz=[0.5, 0.2, 0.1], position=[-0.1, 0.2, 0.4], yaw=0.2)
        obb1 = obb1.unsqueeze(0)
        obb2 = obb2.unsqueeze(0)

        # Execute: compute exact IoU
        iou_exact = iou_exact7(obb1, obb2, all_pairs=False)

        # Execute: compute sampling-based IoU for comparison
        iou_sampling = iou_mc9(obb1, obb2, samp_per_dim=32, all_pairs=False)

        # Assert: exact and sampling should be close
        print(
            f"Exact IoU: {iou_exact[0].item():.5f}, Sampling IoU: {iou_sampling[0].item():.5f}"
        )
        self.assertTrue(
            torch.allclose(iou_exact, iou_sampling, atol=0.05),
            f"Exact IoU {iou_exact[0].item():.5f} differs from sampling {iou_sampling[0].item():.5f}",
        )

    def test_all_pairs_vs_pairwise_consistency(self) -> None:
        """Test that all_pairs=True and all_pairs=False give same results for diagonal"""
        # Setup: create two boxes
        obb1 = make_obb(sz=[1.0, 0.5, 0.3], position=[0.0, 0.0, 0.0], yaw=0.1)
        obb2 = make_obb(sz=[1.0, 0.5, 0.3], position=[0.0, 0.0, 0.0], yaw=0.3)
        obbs = torch.stack([obb1, obb2])  # Shape: (2, 165)

        # Execute: compute with all_pairs=True (gives 2x2 matrix)
        iou_matrix = iou_exact7(obbs, obbs, all_pairs=True)

        # Execute: compute with all_pairs=False (gives vector of length 2)
        iou_vector = iou_exact7(obbs, obbs, all_pairs=False)

        # Assert: diagonal of matrix should match vector
        diagonal = torch.diagonal(iou_matrix)
        print(f"Matrix diagonal: {diagonal}")
        print(f"Pairwise vector: {iou_vector}")
        self.assertTrue(
            torch.allclose(diagonal, iou_vector, atol=1e-5),
            f"Diagonal {diagonal} != pairwise {iou_vector}",
        )

    def test_no_overlap_returns_zero(self) -> None:
        """IoU of non-overlapping boxes should be 0.0"""
        # Setup: create two boxes far apart
        obb1 = make_obb(sz=[1.0, 1.0, 1.0], position=[0.0, 0.0, 0.0], yaw=0.0)
        obb2 = make_obb(sz=[1.0, 1.0, 1.0], position=[10.0, 10.0, 10.0], yaw=0.0)
        obb1 = obb1.unsqueeze(0)
        obb2 = obb2.unsqueeze(0)

        # Execute: compute IoU
        iou = iou_exact7(obb1, obb2, all_pairs=False)

        # Assert: IoU should be 0.0
        self.assertAlmostEqual(iou[0].item(), 0.0, places=5)

    def test_half_size_box(self) -> None:
        """Test IoU when one box is half the size of another, at same center"""
        # Setup: create two boxes, one half the size
        obb1 = make_obb(sz=[1.0, 1.0, 1.0], position=[0.0, 0.0, 0.0], yaw=0.0)
        obb2 = make_obb(sz=[0.5, 0.5, 0.5], position=[0.0, 0.0, 0.0], yaw=0.0)
        obb1 = obb1.unsqueeze(0)
        obb2 = obb2.unsqueeze(0)

        # Execute: compute exact IoU
        iou_exact = iou_exact7(obb1, obb2, all_pairs=False)

        # Execute: compute sampling-based IoU for comparison
        iou_sampling = iou_mc9(obb1, obb2, samp_per_dim=32, all_pairs=False)

        # Assert: exact and sampling should be close
        # Expected: smaller box volume = 0.125, larger = 1.0, intersection = 0.125
        # IoU = 0.125 / (1.0 + 0.125 - 0.125) = 0.125 / 1.0 = 0.125
        print(
            f"Exact IoU: {iou_exact[0].item():.5f}, Sampling IoU: {iou_sampling[0].item():.5f}"
        )
        self.assertAlmostEqual(iou_exact[0].item(), 0.125, places=3)
        self.assertTrue(
            torch.allclose(iou_exact, iou_sampling, atol=0.05),
            f"Exact IoU {iou_exact[0].item():.5f} differs from sampling {iou_sampling[0].item():.5f}",
        )


if __name__ == "__main__":
    unittest.main()
