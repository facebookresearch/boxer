# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from utils.gravity import gravity_align_T_world_cam, reject_vector_a_from_b
from utils.tw.pose import PoseTW


class TestRejectVector(unittest.TestCase):
    def test_parallel_vectors(self):
        """Rejecting a from b when a is parallel to b should give zero."""
        a = torch.tensor([[3.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        rej = reject_vector_a_from_b(a, b)
        self.assertTrue(torch.allclose(rej, torch.zeros_like(rej), atol=1e-6))

    def test_orthogonal_vectors(self):
        """Rejecting a from b when a is orthogonal to b should return a unchanged."""
        a = torch.tensor([[0.0, 1.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        rej = reject_vector_a_from_b(a, b)
        self.assertTrue(torch.allclose(rej, a, atol=1e-6))

    def test_result_orthogonal_to_b(self):
        """The rejection should always be orthogonal to b."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0, 1.0, 0.0]])
        rej = reject_vector_a_from_b(a, b)
        dot = (rej * b).sum(-1)
        self.assertTrue(torch.allclose(dot, torch.zeros_like(dot), atol=1e-6))

    def test_batched(self):
        """Should work with batched inputs."""
        a = torch.randn(5, 3)
        b = torch.randn(5, 3)
        rej = reject_vector_a_from_b(a, b)
        dot = (rej * b).sum(-1)
        self.assertTrue(torch.allclose(dot, torch.zeros_like(dot), atol=1e-5))


class TestGravityAlign(unittest.TestCase):
    def _make_pose(self, R, t=None):
        if t is None:
            t = torch.zeros(3)
        return PoseTW.from_Rt(R.unsqueeze(0), t.unsqueeze(0))

    def test_identity_pose(self):
        """Gravity-aligning an identity pose should produce a valid rotation."""
        pose = self._make_pose(torch.eye(3))
        aligned = gravity_align_T_world_cam(pose)
        R = aligned.R
        # R should be orthogonal: R^T R = I
        eye = R[0].T @ R[0]
        self.assertTrue(torch.allclose(eye, torch.eye(3), atol=1e-5))

    def test_x_axis_is_gravity(self):
        """The x-axis of the aligned frame should be the gravity direction."""
        torch.manual_seed(42)
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        # Ensure proper rotation (det=+1)
        if torch.det(R) < 0:
            R[:, 0] *= -1
        pose = self._make_pose(R)
        aligned = gravity_align_T_world_cam(pose)
        x_axis = aligned.R[0, :, 0]
        gravity = torch.tensor([0.0, 0.0, -1.0])
        # x_axis should be parallel to gravity (up to sign from normalization)
        cos = torch.dot(x_axis, gravity) / (x_axis.norm() * gravity.norm())
        self.assertAlmostEqual(abs(cos.item()), 1.0, places=5)

    def test_z_grav_mode(self):
        """With z_grav=True, the z-axis should align with gravity."""
        pose = self._make_pose(torch.eye(3))
        aligned = gravity_align_T_world_cam(pose, z_grav=True)
        R = aligned.R[0]
        # R should still be orthogonal
        eye = R.T @ R
        self.assertTrue(torch.allclose(eye, torch.eye(3), atol=1e-5))
        # det should be +1 (proper rotation)
        self.assertAlmostEqual(torch.det(R).item(), 1.0, places=4)

    def test_translation_preserved(self):
        """Translation should be preserved after gravity alignment."""
        t = torch.tensor([1.0, 2.0, 3.0])
        pose = self._make_pose(torch.eye(3), t)
        aligned = gravity_align_T_world_cam(pose)
        self.assertTrue(torch.allclose(aligned.t[0], t, atol=1e-6))

    def test_batched(self):
        """Should work with a batch of poses."""
        torch.manual_seed(0)
        B = 4
        Rs = torch.linalg.qr(torch.randn(B, 3, 3))[0]
        # Fix determinants
        for i in range(B):
            if torch.det(Rs[i]) < 0:
                Rs[i, :, 0] *= -1
        ts = torch.randn(B, 3)
        pose = PoseTW.from_Rt(Rs, ts)
        aligned = gravity_align_T_world_cam(pose)
        # Each should have orthogonal R
        for i in range(B):
            eye = aligned.R[i].T @ aligned.R[i]
            self.assertTrue(torch.allclose(eye, torch.eye(3), atol=1e-5))
        # Translations preserved
        self.assertTrue(torch.allclose(aligned.t, ts, atol=1e-6))

    def test_custom_gravity(self):
        """Should work with a custom gravity direction."""
        gravity_w = np.array([0.0, -1.0, 0.0], np.float32)
        pose = self._make_pose(torch.eye(3))
        aligned = gravity_align_T_world_cam(pose, gravity_w=gravity_w)
        x_axis = aligned.R[0, :, 0]
        expected = torch.tensor([0.0, -1.0, 0.0])
        cos = torch.dot(x_axis, expected) / (x_axis.norm() * expected.norm())
        self.assertAlmostEqual(abs(cos.item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
