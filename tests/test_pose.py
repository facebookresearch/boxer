#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import unittest

import numpy as np
import torch

from utils.tw.pose import (
    PoseTW,
    all_rot90,
    closest_timed_poses,
    find_r90,
    fit_to_SO3,
    get_T_rot_z,
    interpolate_timed_poses,
    interpolation_boundaries_alphas,
    inv_skew_symmetric,
    lower_timed_poses,
    quaternion_to_matrix,
    rotation_from_ortho_6d,
    skew_symmetric,
    so3exp_map,
    so3log_map,
)

EPS = 1e-6


class TestPose(unittest.TestCase):
    def test_constructor(self):
        RR = torch.eye(3)
        tt = torch.tensor([0.0, 1.0, 0.0])
        pose = PoseTW.from_Rt(RR, tt)
        self.assertTrue(torch.allclose(pose.R, RR))
        self.assertTrue(torch.allclose(pose.t, tt))

    def test_constructor_batch(self):
        B = 4
        RR = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1)
        tt = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3).repeat(B, 1)
        pose = PoseTW.from_Rt(RR, tt)
        self.assertTrue(torch.allclose(pose.R, RR))
        self.assertTrue(torch.allclose(pose.t, tt))

    def test_inverse(self):
        pose = PoseTW.from_aa(np.float64([0, 1, 0]), np.float64([1, 2, 3]))
        pose2 = pose.inverse()
        pose3 = pose2.inverse()
        self.assertTrue(torch.allclose(pose._data, pose3._data))

    def test_inverse_batch(self):
        B = 4
        poses = []
        for b in range(B):
            poses.append(PoseTW.from_aa(np.float64([0, 1, 0]), np.float64([0, 0, b])))
        poses = torch.stack(poses)
        poses2 = poses.inverse()
        poses3 = poses2.inverse()
        self.assertTrue(torch.allclose(poses._data, poses3._data))

    def test_euler_angle(self):
        torch.manual_seed(666)
        Rx = torch.eye(3)
        Ry = torch.eye(3)
        Rz = torch.eye(3)
        tt = torch.tensor([0.0, 1.0, 0.0])
        # Generate random poses around x, y, z
        x_angle = torch.rand(1)
        y_angle = torch.rand(1)
        z_angle = torch.rand(1)
        # Convert rad to degree to make full coverage of the test
        x_degree = x_angle * 180.0 / torch.pi
        y_degree = y_angle * 180.0 / torch.pi
        z_degree = z_angle * 180.0 / torch.pi
        # Compose rotation matrix from the euler angles
        Rx[1, 1] = torch.cos(x_angle)
        Rx[1, 2] = -torch.sin(x_angle)
        Rx[2, 1] = torch.sin(x_angle)
        Rx[2, 2] = torch.cos(x_angle)

        Ry[0, 0] = torch.cos(y_angle)
        Ry[0, 2] = torch.sin(y_angle)
        Ry[2, 0] = -torch.sin(y_angle)
        Ry[2, 2] = torch.cos(y_angle)

        Rz[0, 0] = torch.cos(z_angle)
        Rz[0, 1] = -torch.sin(z_angle)
        Rz[1, 0] = torch.sin(z_angle)
        Rz[1, 1] = torch.cos(z_angle)

        # ZYX multipliaction order.
        R_compose = Rz @ (Ry @ Rx)

        # Convert
        pose = PoseTW.from_Rt(R_compose, tt)
        result_angles = pose.to_euler()
        result_angles_degree = pose.to_euler(rad=False)
        # Test conversion
        self.assertTrue(
            torch.allclose(result_angles, torch.tensor([x_angle, y_angle, z_angle]))
        )
        # Test degree output
        self.assertTrue(
            torch.allclose(
                result_angles_degree, torch.tensor([x_degree, y_degree, z_degree])
            )
        )

    def test_euler_angle_batch(self):
        B = 1000
        torch.manual_seed(666)
        Rx = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1)
        Ry = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1)
        Rz = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1)
        tt = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3).repeat(B, 1)
        # Generate random poses around x, y, z
        x_angle = torch.rand(B)
        y_angle = torch.rand(B)
        z_angle = torch.rand(B)
        # Convert rad to degree to make full coverage of the test
        x_degree = x_angle * 180.0 / torch.pi
        y_degree = y_angle * 180.0 / torch.pi
        z_degree = z_angle * 180.0 / torch.pi
        # Compose rotation matrix from the euler angles
        Rx[:, 1, 1] = torch.cos(x_angle)
        Rx[:, 1, 2] = -torch.sin(x_angle)
        Rx[:, 2, 1] = torch.sin(x_angle)
        Rx[:, 2, 2] = torch.cos(x_angle)

        Ry[:, 0, 0] = torch.cos(y_angle)
        Ry[:, 0, 2] = torch.sin(y_angle)
        Ry[:, 2, 0] = -torch.sin(y_angle)
        Ry[:, 2, 2] = torch.cos(y_angle)

        Rz[:, 0, 0] = torch.cos(z_angle)
        Rz[:, 0, 1] = -torch.sin(z_angle)
        Rz[:, 1, 0] = torch.sin(z_angle)
        Rz[:, 1, 1] = torch.cos(z_angle)

        # ZYX multipliaction order.
        R_compose = Rz @ (Ry @ Rx)

        # Convert
        pose = PoseTW.from_Rt(R_compose, tt)
        result_angles = pose.to_euler()  # Bx3
        result_angles_degree = pose.to_euler(rad=False)
        # Test conversion
        self.assertTrue(
            torch.allclose(result_angles, torch.stack([x_angle, y_angle, z_angle], 1))
        )
        # Test degree output
        self.assertTrue(
            torch.allclose(
                result_angles_degree, torch.stack([x_degree, y_degree, z_degree], 1)
            )
        )

    def test_transform(self):
        t = torch.randn(3)
        T_ab = PoseTW.from_aa(torch.randn(3), t)
        R_ab = T_ab.R
        t_ab = T_ab.t
        self.assertTrue(torch.allclose(t_ab, t))
        p_b = torch.randn(10, 3)
        p_a = T_ab.transform(p_b)
        # use the more common left multiply to heck the more efficient right-multiply in the PoseTW implementation
        pDirect_a = (R_ab @ p_b.transpose(1, 0)).transpose(1, 0) + t_ab.unsqueeze(0)
        self.assertTrue(torch.allclose(p_a, pDirect_a))

    def test_rotate(self):
        T_ab = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        R_ab = T_ab.R
        p_b = torch.randn(10, 3)
        p_a = T_ab.rotate(p_b)
        # use the more common left multiply to heck the more efficient right-multiply in the PoseTW implementation
        pDirect_a = (R_ab @ p_b.transpose(1, 0)).transpose(1, 0)
        self.assertTrue(torch.allclose(p_a, pDirect_a))

    def test_interpolation_boundaries(self):
        T = 10
        times = torch.FloatTensor([float(t) / float(T) for t in range(0, T)])
        interp_times = torch.FloatTensor(
            [float(t) / (2.0 * T) for t in range(-1, 2 * T)]
        )
        # check interpolation timestamps with all of them inside the times interval
        lower_ids, upper_ids, alphas, good = interpolation_boundaries_alphas(
            times, interp_times[1:-1]
        )
        self.assertEqual(lower_ids.shape, interp_times[1:-1].shape)
        self.assertEqual(upper_ids.shape, interp_times[1:-1].shape)
        self.assertEqual(alphas.shape, interp_times[1:-1].shape)
        self.assertEqual(good.shape, interp_times[1:-1].shape)
        self.assertTrue((lower_ids <= upper_ids).all())
        self.assertTrue((alphas >= 0.0).all())
        self.assertTrue((alphas <= 1.0).all())
        self.assertTrue(good.all())
        self.assertEqual(lower_ids[0].item(), 0)
        self.assertEqual(upper_ids[0].item(), 0)
        self.assertAlmostEqual(alphas[0].item(), 0.0, delta=EPS)
        self.assertEqual(lower_ids[1].item(), 0)
        self.assertEqual(upper_ids[1].item(), 1)
        self.assertAlmostEqual(alphas[1].item(), 0.5, delta=EPS)
        self.assertEqual(lower_ids[2].item(), 1)
        self.assertEqual(upper_ids[2].item(), 1)
        self.assertAlmostEqual(alphas[2].item(), 0.0, delta=EPS)

        # check interpolation timestamps with some of the outside the times interval
        lower_ids, upper_ids, alphas, good = interpolation_boundaries_alphas(
            times, interp_times
        )
        self.assertEqual(lower_ids.shape, interp_times.shape)
        self.assertEqual(upper_ids.shape, interp_times.shape)
        self.assertEqual(alphas.shape, interp_times.shape)
        self.assertEqual(good.shape, interp_times.shape)
        self.assertTrue((lower_ids <= upper_ids).all())
        self.assertTrue((alphas >= 0.0).all())
        self.assertTrue((alphas <= 1.0).all())
        self.assertTrue(good[1:-1].all().item())
        self.assertTrue(good[0].item() is False)
        self.assertTrue(good[-1].item() is False)
        bad = torch.logical_not(good)
        self.assertTrue(torch.allclose(alphas[bad], torch.zeros_like(alphas[bad])))

        # check some random interpolation timestamps, some of the outside the times interval
        interp_times = torch.rand(100)
        lower_ids, upper_ids, alphas, good = interpolation_boundaries_alphas(
            times, interp_times
        )
        self.assertEqual(lower_ids.shape, interp_times.shape)
        self.assertEqual(upper_ids.shape, interp_times.shape)
        self.assertEqual(alphas.shape, interp_times.shape)
        self.assertEqual(good.shape, interp_times.shape)
        self.assertTrue((lower_ids <= upper_ids).all())
        self.assertTrue((alphas >= 0.0).all())
        self.assertTrue((alphas <= 1.0).all())
        self.assertTrue(good[interp_times <= times.max().item()].all())
        self.assertTrue(not good[interp_times > times.max().item()].any())
        bad = torch.logical_not(good)
        self.assertTrue(torch.allclose(alphas[bad], torch.zeros_like(alphas[bad])))

    def test_interpolation_translation(self):
        T = 4
        poses, pose_times = [], []
        for t in range(T):
            poses.append(PoseTW.from_aa(np.float64([0, 1, 0]), np.float64([0, 0, t])))
            pose_times.append(float(t) / float(T))
        poses = torch.stack(poses)
        pose_times = torch.FloatTensor(pose_times)
        interp_times = torch.FloatTensor(
            [float(i) / (2.0 * T) for i in range(1, 2 * T - 1)]
        )
        poses_int, good = poses.interpolate(pose_times, interp_times)
        for i, pose in enumerate(poses_int):
            self.assertTrue(torch.allclose(poses[0].R, pose.R))
            self.assertTrue(
                torch.allclose(
                    pose.t, torch.from_numpy(np.float64([0.0, 0.0, (i + 1) * 0.5]))
                )
            )

    def test_so3_maps(self):
        torch.manual_seed(42)  # Fixed seed for reproducibility
        # Use bounded random to avoid numerical instability near rotations of pi
        xi = torch.randn(10, 3) * 0.5
        R = so3exp_map(xi)
        xi_after = so3log_map(R)
        self.assertTrue(torch.allclose(xi, xi_after, rtol=1e-3))

    def test_so3log_map_near_pi_rotation(self):
        theta_true = math.pi - 1e-4
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        omega_in = axis * theta_true
        R = so3exp_map(omega_in.unsqueeze(0))

        omega_out = so3log_map(R)
        recovered_theta = torch.norm(omega_out).item()

        self.assertAlmostEqual(
            recovered_theta,
            theta_true,
            places=5,
            msg=f"Expected theta ≈ {theta_true}, got {recovered_theta}",
        )

    def test_all_rot90(self):
        R90s = all_rot90()
        self.assertEqual(R90s.shape, (24, 3, 3))
        dets = torch.linalg.det(R90s)
        self.assertTrue(((dets - 1.0).abs() < 1e-6).all())

    def test_find_r90(self):
        R90s = all_rot90()
        Ta = PoseTW.from_aa(torch.tensor([0.1, 0.2, 0.3]), torch.randn(3))
        for i in range(24):
            Tb = PoseTW.from_Rt(Ta.R @ R90s[i], Ta.t)
            Tc, R90min = find_r90(Ta, Tb, R90s)
            dR = R90s[i] @ R90min
            self.assertTrue(torch.allclose(dR.abs(), torch.eye(3)))

    def test_quaternion(self):
        quat_identity = torch.tensor([1.0, 0, 0, 0])
        translation = torch.randn(3)
        rot_identity = torch.eye(3)
        pose = PoseTW.from_qt(quat_identity, translation)

        self.assertTrue(torch.allclose(pose.q, quat_identity, rtol=1e-3))
        self.assertTrue(torch.allclose(pose.R, rot_identity, rtol=1e-3))
        self.assertTrue(torch.allclose(pose.t, translation, rtol=1e-3))

        # test batched input
        B, T = 4, 10
        quat_wxyz = torch.nn.functional.normalize(torch.randn(B, T, 4), p=2, dim=-1)
        quat_xyzw = torch.concat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)
        rotations = quaternion_to_matrix(quat_wxyz)
        translation = torch.randn(B, T, 3)
        pose = PoseTW.from_qt(quat_wxyz, translation)

        quat_wxyz = quat_wxyz.view(-1, 4)
        quat_xyzw = quat_xyzw.view(-1, 4)
        quat_pose = pose.q.view(-1, 4)
        quat_pose_xyzw = pose.q_xyzw.view(-1, 4)

        # make the real part of the quaternions positive because -q and q are the same for quaternions
        neg_ind = torch.nonzero(quat_wxyz[:, 0] < 0).squeeze()
        quat_wxyz[neg_ind, :] *= -1
        neg_ind = torch.nonzero(quat_xyzw[:, -1] < 0).squeeze()
        quat_xyzw[neg_ind, :] *= -1

        self.assertTrue(torch.allclose(quat_pose, quat_wxyz, rtol=1e-3))
        self.assertTrue(torch.allclose(quat_pose_xyzw, quat_xyzw, rtol=1e-3))
        self.assertTrue(torch.allclose(pose.R, rotations, rtol=1e-3))
        self.assertTrue(torch.allclose(pose.t, translation, rtol=1e-3))

        # test init from angle-axis and convert to quaternion
        T = PoseTW.from_aa(torch.randn(B, T, 3), torch.randn(B, T, 3))
        q, t = T.q, T.t
        T2 = PoseTW.from_qt(q, t)
        assert torch.allclose(T.matrix3x4, T2.matrix3x4, rtol=1e-3)

    def test_geodesic(self):
        tt = torch.zeros(3)
        a = PoseTW.from_qt(torch.tensor([1, 0, 0, 0]), tt)
        N = 100
        for _ in range(N):
            rot_z_angle = np.random.rand() * np.pi  # [0, pi]
            b_3x4 = get_T_rot_z(rot_z_angle)
            b = PoseTW.from_Rt(b_3x4[:3, :3], b_3x4[:3, 3])
            assert torch.allclose(
                a.so3_geodesic(b, deg=True),
                torch.tensor(rot_z_angle) * 180.0 / torch.pi,
                rtol=1e-2,
            )

        # random pose
        shape = [10, 4, 3]
        pose1 = PoseTW.from_aa(torch.rand(shape), torch.randn(shape))
        pose2 = PoseTW.from_aa(torch.rand(shape), torch.randn(shape))
        dr1 = pose1.so3_geodesic(pose2, deg=False)

        pose_error = pose1.compose(pose2.inverse())
        dr2, _ = pose_error.magnitude(deg=False)
        assert torch.allclose(dr1, dr2)

        dr1 = pose1.so3_geodesic(pose2, deg=True)
        dr2, _ = pose_error.magnitude(deg=True)
        assert torch.allclose(dr1, dr2)

    def test_align(self):
        torch.manual_seed(0)
        shape = [10, 3]
        # Generate a random trajectory.
        T_aw = PoseTW.from_aa(torch.rand(shape), torch.randn(shape))
        # Generate a random rigid body transformation.
        T_ab = PoseTW.from_aa(torch.rand(3), torch.randn(3))
        T_bw = T_ab.inverse() @ T_aw
        T_ab2, mean_error = T_aw.align(T_bw)
        self.assertTrue(torch.allclose(T_ab, T_ab2))
        self.assertTrue(mean_error < 0.00001)
        # Add some noise to second trajectory translation.
        T_bw2 = T_bw.clone()
        T_bw2._data[:, -3:] += 0.1 * torch.rand(shape)
        T_ab3, mean_error2 = T_aw.align(T_bw2)
        self.assertTrue(torch.allclose(T_ab, T_ab3, atol=1e-1, rtol=1e-1))
        self.assertTrue(mean_error2 < 0.1)

    def test_align_interp(self):
        """Test two timestamped trajectories with different lengths"""
        torch.manual_seed(0)
        N = 10
        M = 20
        t_a = torch.linspace(0, 1, steps=N).reshape(N, 1).repeat(1, 3)
        t_b = torch.linspace(0, 1, steps=M).reshape(M, 1).repeat(1, 3)
        R_a = torch.eye(3).reshape(1, 3, 3).repeat(N, 1, 1)
        R_b = torch.eye(3).reshape(1, 3, 3).repeat(M, 1, 1)
        T_aw = PoseTW.from_Rt(R_a, t_a)
        T_bw = PoseTW.from_Rt(R_b, t_b)
        times_a = torch.linspace(0, 10, steps=N)
        times_b = torch.linspace(0, 10, steps=M)
        # Perfectly aligned, just different times.
        T_ab, mean_error = T_aw.align(T_bw, times_a, times_b)
        self.assertTrue(mean_error < 0.00001)
        # No error but off by random SE3.
        T_bc = PoseTW.from_aa(torch.rand(3), torch.randn(3))
        T_cw = T_bc.inverse() @ T_bw
        T_ac, mean_error = T_aw.align(T_cw, times_a, times_b)
        self.assertTrue(mean_error < 0.00001)
        # Small error and off by random SE3.
        T_bd = PoseTW.from_aa(torch.rand(3), torch.randn(3))
        T_dw = T_bd.inverse() @ T_bw
        T_dw._data[:, -3:] += 0.1 * torch.rand((M, 3))
        T_ad, mean_error = T_aw.align(T_dw, times_a, times_b)
        self.assertTrue(mean_error < 0.1)

    def test_fit_to_SO3(self):
        torch.manual_seed(0)
        N = 10
        EPS = 1e-5
        for n in range(N):
            pose = PoseTW.from_aa(torch.rand(3), torch.randn(3)).double()

            # Test that column 1 and 2 are orthogonal.
            dot1 = pose.R[:, 0] @ pose.R[:, 1]
            self.assertTrue(float(torch.abs(dot1)) < EPS)
            pose._data[:9] += 0.01 * torch.randn(9)
            dot2 = pose.R[:, 0] @ pose.R[:, 1]
            self.assertTrue(float(torch.abs(dot2)) > EPS)
            pose2 = pose.fit_to_SO3()
            dot3 = pose2.R[:, 0] @ pose2.R[:, 1]
            self.assertTrue(float(torch.abs(dot3)) < EPS)

            # Test that det(R) == 1.
            self.assertFalse(torch.allclose(pose.R.det(), torch.ones(1).double()))
            self.assertTrue(torch.allclose(pose2.R.det(), torch.ones(1).double()))

    # ==================== Additional Comprehensive Tests ====================

    def test_identity_pose(self):
        """Test that default constructor creates identity pose."""
        pose = PoseTW()
        self.assertTrue(torch.allclose(pose.R, torch.eye(3)))
        self.assertTrue(torch.allclose(pose.t, torch.zeros(3)))
        self.assertTrue(torch.allclose(pose.matrix, torch.eye(4)))

    def test_from_matrix(self):
        """Test construction from 4x4 transformation matrix."""
        # Identity matrix
        T_identity = torch.eye(4)
        pose = PoseTW.from_matrix(T_identity)
        self.assertTrue(torch.allclose(pose.R, torch.eye(3)))
        self.assertTrue(torch.allclose(pose.t, torch.zeros(3)))

        # Random pose
        T_rand = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        pose_from_matrix = PoseTW.from_matrix(T_rand.matrix)
        self.assertTrue(
            torch.allclose(T_rand.matrix, pose_from_matrix.matrix, atol=1e-6)
        )

        # Batched
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        poses_from_matrix = PoseTW.from_matrix(poses.matrix)
        self.assertTrue(
            torch.allclose(poses.matrix, poses_from_matrix.matrix, atol=1e-6)
        )

    def test_from_matrix3x4(self):
        """Test construction from 3x4 transformation matrix."""
        # Identity
        T_3x4 = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)
        pose = PoseTW.from_matrix3x4(T_3x4)
        self.assertTrue(torch.allclose(pose.R, torch.eye(3)))
        self.assertTrue(torch.allclose(pose.t, torch.zeros(3)))

        # Random pose
        T_rand = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        pose_from_3x4 = PoseTW.from_matrix3x4(T_rand.matrix3x4)
        self.assertTrue(
            torch.allclose(T_rand.matrix3x4, pose_from_3x4.matrix3x4, atol=1e-6)
        )

        # Batched
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        poses_from_3x4 = PoseTW.from_matrix3x4(poses.matrix3x4)
        self.assertTrue(
            torch.allclose(poses.matrix3x4, poses_from_3x4.matrix3x4, atol=1e-6)
        )

    def test_matrix_properties(self):
        """Test matrix and matrix3x4 properties."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))

        # matrix3x4 should be first 3 rows of matrix
        self.assertTrue(torch.allclose(pose.matrix[:3, :], pose.matrix3x4))

        # matrix should have [0,0,0,1] as last row
        self.assertTrue(torch.allclose(pose.matrix[3, :3], torch.zeros(3)))
        self.assertTrue(torch.allclose(pose.matrix[3, 3], torch.tensor(1.0)))

        # Batched
        B = 4
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        self.assertTrue(torch.allclose(poses.matrix[:, :3, :], poses.matrix3x4))
        self.assertTrue(torch.allclose(poses.matrix[:, 3, :3], torch.zeros(B, 3)))
        self.assertTrue(torch.allclose(poses.matrix[:, 3, 3], torch.ones(B)))

    def test_mul_operator(self):
        """Test __mul__ operator for transforming points."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        points = torch.randn(10, 3)

        # __mul__ should be equivalent to transform
        result_mul = pose * points
        result_transform = pose.transform(points)
        self.assertTrue(torch.allclose(result_mul, result_transform))

    def test_matmul_operator(self):
        """Test __matmul__ operator for composing poses."""
        pose1 = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        pose2 = PoseTW.from_aa(torch.randn(3), torch.randn(3))

        # __matmul__ should be equivalent to compose
        result_matmul = pose1 @ pose2
        result_compose = pose1.compose(pose2)
        self.assertTrue(torch.allclose(result_matmul._data, result_compose._data))

    def test_magnitude(self):
        """Test magnitude method."""
        # Identity pose should have zero magnitude
        pose_id = PoseTW()
        dr, dt = pose_id.magnitude()
        self.assertTrue(torch.allclose(dr, torch.tensor(0.0), atol=1e-5))
        self.assertTrue(torch.allclose(dt, torch.tensor(0.0), atol=1e-5))

        # Pure translation
        pose_t = PoseTW.from_Rt(torch.eye(3), torch.tensor([1.0, 0.0, 0.0]))
        dr, dt = pose_t.magnitude()
        self.assertTrue(torch.allclose(dr, torch.tensor(0.0), atol=1e-5))
        self.assertTrue(torch.allclose(dt, torch.tensor(1.0), atol=1e-5))

        # 90 degree rotation around z-axis
        angle = torch.tensor(np.pi / 2)
        R_90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose_r = PoseTW.from_Rt(R_90, torch.zeros(3))
        dr, dt = pose_r.magnitude(deg=True)
        self.assertTrue(torch.allclose(dr, torch.tensor(90.0), atol=1e-3))
        self.assertTrue(torch.allclose(dt, torch.tensor(0.0), atol=1e-5))

        # Radians
        dr_rad, _ = pose_r.magnitude(deg=False)
        self.assertTrue(torch.allclose(dr_rad, angle, atol=1e-3))

        # Batched
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        dr, dt = poses.magnitude()
        self.assertEqual(dr.shape, (B,))
        self.assertEqual(dt.shape, (B,))

    def test_numpy_method(self):
        """Test numpy method returns R and t as numpy arrays."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        R_np, t_np = pose.numpy()
        self.assertIsInstance(R_np, np.ndarray)
        self.assertIsInstance(t_np, np.ndarray)
        self.assertEqual(R_np.shape, (3, 3))
        self.assertEqual(t_np.shape, (3,))
        self.assertTrue(np.allclose(R_np, pose.R.numpy()))
        self.assertTrue(np.allclose(t_np, pose.t.numpy()))

    def test_batch_transform(self):
        """Test batch_transform method."""
        B = 10
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        points = torch.randn(B, 3)

        result = poses.batch_transform(points)
        self.assertEqual(result.shape, (B, 3))

        # Verify against individual transforms
        for i in range(B):
            expected = poses[i].transform(points[i : i + 1]).squeeze(0)
            self.assertTrue(torch.allclose(result[i], expected, atol=1e-5))

    def test_skew_symmetric(self):
        """Test skew_symmetric and inv_skew_symmetric functions."""
        v = torch.randn(3)
        M = skew_symmetric(v)

        # Skew symmetric: M = -M^T
        self.assertTrue(torch.allclose(M, -M.T, atol=1e-6))

        # Inverse should recover original vector
        v_recovered = inv_skew_symmetric(M)
        self.assertTrue(torch.allclose(v, v_recovered, atol=1e-6))

        # Batched
        B = 5
        v_batch = torch.randn(B, 3)
        M_batch = skew_symmetric(v_batch)
        self.assertEqual(M_batch.shape, (B, 3, 3))
        v_recovered_batch = inv_skew_symmetric(M_batch)
        self.assertTrue(torch.allclose(v_batch, v_recovered_batch, atol=1e-6))

    def test_so3_maps_near_identity(self):
        """Test so3 exp/log maps near identity (numerical stability)."""
        # Very small rotation
        small_xi = torch.tensor([1e-8, 1e-8, 1e-8])
        R = so3exp_map(small_xi)
        xi_recovered = so3log_map(R)
        self.assertTrue(torch.allclose(small_xi, xi_recovered, atol=1e-6))

        # Zero rotation
        zero_xi = torch.zeros(3)
        R_identity = so3exp_map(zero_xi)
        self.assertTrue(torch.allclose(R_identity, torch.eye(3), atol=1e-6))

    def test_so3_maps_large_rotation(self):
        """Test so3 exp/log maps for large rotations (near pi)."""
        # Rotation near pi (but not exactly)
        large_xi = torch.tensor([np.pi - 0.1, 0.0, 0.0])
        R = so3exp_map(large_xi)
        xi_recovered = so3log_map(R)
        # Should be close (may have sign ambiguity near pi)
        self.assertTrue(torch.allclose(large_xi.abs(), xi_recovered.abs(), atol=0.1))

    def test_exp_log_roundtrip(self):
        """Test SE3 exp/log roundtrip."""
        for _ in range(10):
            u_omega = torch.randn(6) * 0.5  # Keep small for numerical stability
            pose = PoseTW.exp(u_omega)
            log_result = pose.log()
            pose_recovered = PoseTW.exp(log_result)
            self.assertTrue(
                torch.allclose(pose.matrix, pose_recovered.matrix, atol=1e-4)
            )

    def test_exp_log_batched(self):
        """Test SE3 exp/log with batched inputs."""
        B = 5
        u_omega = torch.randn(B, 6) * 0.5
        poses = PoseTW.exp(u_omega)
        self.assertEqual(poses.shape, (B, 12))

        log_result = poses.log()
        self.assertEqual(log_result.shape, (B, 6))

        poses_recovered = PoseTW.exp(log_result)
        self.assertTrue(torch.allclose(poses.matrix, poses_recovered.matrix, atol=1e-4))

    def test_compose_associativity(self):
        """Test that pose composition is associative."""
        T1 = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        T2 = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        T3 = PoseTW.from_aa(torch.randn(3), torch.randn(3))

        # (T1 @ T2) @ T3 == T1 @ (T2 @ T3)
        left = (T1 @ T2) @ T3
        right = T1 @ (T2 @ T3)
        self.assertTrue(torch.allclose(left.matrix, right.matrix, atol=1e-5))

    def test_inverse_compose_identity(self):
        """Test that T @ T.inverse() == identity."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        identity = pose @ pose.inverse()
        self.assertTrue(torch.allclose(identity.R, torch.eye(3), atol=1e-5))
        self.assertTrue(torch.allclose(identity.t, torch.zeros(3), atol=1e-5))

        # Also T.inverse() @ T
        identity2 = pose.inverse() @ pose
        self.assertTrue(torch.allclose(identity2.R, torch.eye(3), atol=1e-5))
        self.assertTrue(torch.allclose(identity2.t, torch.zeros(3), atol=1e-5))

    def test_transform_inverse(self):
        """Test that T.inverse() * (T * p) == p."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        points = torch.randn(10, 3)

        transformed = pose.transform(points)
        recovered = pose.inverse().transform(transformed)
        self.assertTrue(torch.allclose(points, recovered, atol=1e-5))

    def test_dtype_conversions(self):
        """Test dtype conversions (float, double)."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        self.assertEqual(pose.dtype, torch.float32)

        pose_double = pose.double()
        self.assertEqual(pose_double.dtype, torch.float64)
        self.assertTrue(
            torch.allclose(pose.matrix.double(), pose_double.matrix, atol=1e-6)
        )

        pose_float = pose_double.float()
        self.assertEqual(pose_float.dtype, torch.float32)

    def test_clone(self):
        """Test clone creates independent copy."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        pose_clone = pose.clone()

        # Modify original
        original_data = pose._data.clone()
        pose._data[0] = 999.0

        # Clone should be unchanged
        self.assertFalse(torch.allclose(pose._data, pose_clone._data))
        self.assertTrue(torch.allclose(original_data, pose_clone._data))

    def test_reshape_and_view(self):
        """Test reshape and view methods."""
        B, T = 4, 3
        poses = PoseTW.from_aa(torch.randn(B, T, 3), torch.randn(B, T, 3))
        self.assertEqual(poses.shape, (B, T, 12))

        # Reshape to flat
        poses_flat = poses.reshape(-1, 12)
        self.assertEqual(poses_flat.shape, (B * T, 12))

        # View
        poses_view = poses.view(B * T, 12)
        self.assertEqual(poses_view.shape, (B * T, 12))

    def test_indexing(self):
        """Test indexing (__getitem__ and __setitem__)."""
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))

        # Get single pose
        pose0 = poses[0]
        self.assertEqual(pose0.shape, (12,))

        # Get slice
        pose_slice = poses[1:3]
        self.assertEqual(pose_slice.shape, (2, 12))

        # Set item
        new_pose = PoseTW.from_aa(torch.zeros(3), torch.ones(3))
        poses[0] = new_pose
        self.assertTrue(torch.allclose(poses[0]._data, new_pose._data))

    def test_rotation_from_ortho_6d(self):
        """Test rotation_from_ortho_6d function."""
        B = 10
        # Create valid 6D representation from known rotations
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        R_orig = poses.R.reshape(B, 3, 3)

        # Extract first two columns as 6D representation
        ortho6d = torch.cat([R_orig[:, :, 0], R_orig[:, :, 1]], dim=-1)
        R_recovered = rotation_from_ortho_6d(ortho6d)

        self.assertEqual(R_recovered.shape, (B, 3, 3))
        self.assertTrue(torch.allclose(R_orig, R_recovered, atol=1e-5))

    def test_fit_to_SO3_function(self):
        """Test standalone fit_to_SO3 function."""
        # Create slightly non-orthogonal matrix
        R = torch.eye(3) + 0.01 * torch.randn(3, 3)

        R_fixed = fit_to_SO3(R)

        # Should be orthogonal now
        should_be_identity = R_fixed @ R_fixed.T
        self.assertTrue(torch.allclose(should_be_identity, torch.eye(3), atol=1e-5))

        # Det should be 1
        self.assertTrue(
            torch.allclose(torch.det(R_fixed), torch.tensor(1.0), atol=1e-5)
        )

        # Multiple individual calls
        for _ in range(5):
            R_i = torch.eye(3) + 0.01 * torch.randn(3, 3)
            R_i_fixed = fit_to_SO3(R_i)
            self.assertTrue(
                torch.allclose(torch.det(R_i_fixed), torch.tensor(1.0), atol=1e-5)
            )

    def test_interpolate_timed_poses(self):
        """Test interpolate_timed_poses function."""
        # Create timed poses dict
        timed_poses = {}
        for t in [0.0, 1.0, 2.0]:
            timed_poses[t] = PoseTW.from_Rt(torch.eye(3), torch.tensor([t, 0.0, 0.0]))

        # Interpolate at midpoint
        pose_05 = interpolate_timed_poses(timed_poses, 0.5)
        self.assertTrue(
            torch.allclose(pose_05.t, torch.tensor([0.5, 0.0, 0.0]), atol=1e-4)
        )

        pose_15 = interpolate_timed_poses(timed_poses, 1.5)
        self.assertTrue(
            torch.allclose(pose_15.t, torch.tensor([1.5, 0.0, 0.0]), atol=1e-4)
        )

    def test_lower_timed_poses(self):
        """Test lower_timed_poses function."""
        timed_poses = {}
        for t in [0.0, 1.0, 2.0]:
            timed_poses[t] = PoseTW.from_Rt(torch.eye(3), torch.tensor([t, 0.0, 0.0]))

        pose, dt = lower_timed_poses(timed_poses, 1.5)
        self.assertTrue(torch.allclose(pose.t, torch.tensor([1.0, 0.0, 0.0])))
        self.assertAlmostEqual(dt, 1.0 - 1.5, places=5)

    def test_closest_timed_poses(self):
        """Test closest_timed_poses function."""
        timed_poses = {}
        for t in [0.0, 1.0, 2.0]:
            timed_poses[t] = PoseTW.from_Rt(torch.eye(3), torch.tensor([t, 0.0, 0.0]))

        # Should return pose at t=1.0 (closer than t=2.0)
        pose, dt = closest_timed_poses(timed_poses, 1.3)
        self.assertTrue(torch.allclose(pose.t, torch.tensor([1.0, 0.0, 0.0])))

        # Should return pose at t=2.0 (closer than t=1.0)
        pose, dt = closest_timed_poses(timed_poses, 1.7)
        self.assertTrue(torch.allclose(pose.t, torch.tensor([2.0, 0.0, 0.0])))

    def test_q_xyzw_property(self):
        """Test q_xyzw property returns correct format."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))

        q_wxyz = pose.q
        q_xyzw = pose.q_xyzw

        # wxyz to xyzw: [w,x,y,z] -> [x,y,z,w]
        self.assertTrue(torch.allclose(q_wxyz[0], q_xyzw[3]))  # w
        self.assertTrue(torch.allclose(q_wxyz[1:4], q_xyzw[0:3]))  # xyz

        # Batched
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        q_wxyz = poses.q
        q_xyzw = poses.q_xyzw

        self.assertTrue(torch.allclose(q_wxyz[..., 0], q_xyzw[..., 3]))
        self.assertTrue(torch.allclose(q_wxyz[..., 1:4], q_xyzw[..., 0:3]))

    def test_repr(self):
        """Test __repr__ method."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        repr_str = repr(pose)
        self.assertIn("PoseTW", repr_str)
        self.assertIn("float32", repr_str)
        self.assertIn("cpu", repr_str)

    def test_stack_and_cat(self):
        """Test stack and cat class methods."""
        poses = [PoseTW.from_aa(torch.randn(3), torch.randn(3)) for _ in range(5)]

        # Stack
        stacked = torch.stack(poses)
        self.assertEqual(stacked.shape, (5, 12))

        # Cat (need matching batch dims)
        poses_2d = [p.unsqueeze(0) for p in poses]
        catted = torch.cat(poses_2d, dim=0)
        self.assertEqual(catted.shape, (5, 12))

    def test_expand_and_repeat(self):
        """Test expand and repeat methods."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))

        # Expand
        expanded = pose.unsqueeze(0).expand(5, 12)
        self.assertEqual(expanded.shape, (5, 12))

        # Repeat
        repeated = pose.unsqueeze(0).repeat(5, 1)
        self.assertEqual(repeated.shape, (5, 12))

    def test_requires_grad(self):
        """Test gradient tracking."""
        pose = PoseTW.from_aa(torch.randn(3), torch.randn(3))
        self.assertFalse(pose.requires_grad)

        pose.requires_grad_(True)
        self.assertTrue(pose.requires_grad)

        # Transform should maintain gradient tracking
        points = torch.randn(10, 3)
        result = pose.transform(points)
        self.assertTrue(result.requires_grad)

    def test_get_T_rot_z(self):
        """Test get_T_rot_z utility function."""
        # 0 degrees - identity rotation
        T_0 = get_T_rot_z(0.0)
        self.assertTrue(torch.allclose(T_0[:3, :3], torch.eye(3), atol=1e-5))

        # 90 degrees
        T_90 = get_T_rot_z(np.pi / 2)
        expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(T_90[:3, :3], expected, atol=1e-5))

        # Translation should be zero
        self.assertTrue(torch.allclose(T_90[:, 3], torch.zeros(3), atol=1e-5))

    def test_interpolation_long_timestamps(self):
        """Test interpolation with nanosecond-scale timestamps."""
        T = 4
        poses, pose_times = [], []
        base_time = 1000000000000  # 10^12 nanoseconds
        for t in range(T):
            poses.append(
                PoseTW.from_Rt(torch.eye(3), torch.tensor([0.0, 0.0, float(t)]))
            )
            pose_times.append(base_time + t * 1000000)  # 1ms apart
        poses = torch.stack(poses)
        pose_times = torch.LongTensor(pose_times)

        # Interpolate at midpoint
        interp_time = torch.LongTensor([base_time + 500000])  # 0.5ms
        poses_int, good = poses.interpolate(pose_times, interp_time)
        self.assertTrue(good[0])
        self.assertTrue(torch.allclose(poses_int[0].t[2], torch.tensor(0.5), atol=0.01))

    def test_compose_batched(self):
        """Test batched pose composition."""
        B = 5
        poses1 = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        poses2 = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))

        composed = poses1 @ poses2
        self.assertEqual(composed.shape, (B, 12))

        # Verify against individual compositions
        for i in range(B):
            expected = poses1[i] @ poses2[i]
            self.assertTrue(
                torch.allclose(composed[i]._data, expected._data, atol=1e-5)
            )

    def test_inverse_batched(self):
        """Test batched pose inversion."""
        B = 5
        poses = PoseTW.from_aa(torch.randn(B, 3), torch.randn(B, 3))
        poses_inv = poses.inverse()

        self.assertEqual(poses_inv.shape, (B, 12))

        # Verify T @ T^-1 = I for each
        identity = poses @ poses_inv
        for i in range(B):
            self.assertTrue(torch.allclose(identity[i].R, torch.eye(3), atol=1e-5))
            self.assertTrue(torch.allclose(identity[i].t, torch.zeros(3), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
