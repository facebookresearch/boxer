#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

#
# Test for CameraTW without a raynier dependecy since it sometimes breaks.
#

# pyre-unsafe

import os
import random
import unittest

import cv2
import numpy as np
import torch

from boxernet.boxernet import generate_patch_centers
from utils.tw.camera import (
    CameraTW,
    get_aria_camera,
    get_base_aria_rgb_camera,
    get_kb4_camera,
    random_fisheye_pixels,
)

RGB_PARAMS = np.float32(
    # pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str,
    #  SupportsFloat, SupportsIndex]` but got `List[float]`.
    [600.0, 352.0, 352.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)  # Single focal
RGB_PARAMS2 = np.float32(
    # pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str,
    #  SupportsFloat, SupportsIndex]` but got `List[float]`.
    [600.0, 700.0, 352.0, 352.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)  # Double focal
# pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str, SupportsFloat,
#  SupportsIndex]` but got `List[float]`.
SLAM_PARAMS = np.float32([500.0, 320.0, 240.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
RGB_RADIUS = 1415
SLAM_RADIUS = 330


class TestCamera2(unittest.TestCase):
    """
    Test for CameraTW project / unproject without a raynier dependecy
    since it sometimes breaks.
    """

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_project_unproject(self):
        B, N, radius = 10, 1000, 1415 / 4.0
        cams = []
        cams.append(get_aria_camera(RGB_PARAMS, B=B))
        cams.append(get_aria_camera(RGB_PARAMS2, B=B))

        for i, cam in enumerate(cams):
            pixels = random_fisheye_pixels(B, N, radius, cam[0].c.numpy())
            pixels = torch.tensor(pixels)

            # Test round trip project, unproject, errors should be very small with double,
            pixels = pixels.double()
            cam = cam.double()
            rays, _ = cam.unproject(pixels)
            pixels2, _ = cam.project(rays)
            self.assertTrue(torch.allclose(pixels, pixels2))

            # Test round trip project, unproject, errors are a bit bigger for float32.
            pixels = pixels.float()
            cam = cam.float()
            rays, _ = cam.unproject(pixels)
            pixels2, _ = cam.project(rays)
            self.assertTrue(torch.allclose(pixels, pixels2, rtol=1e-3, atol=1e-6))

            # test camera center unproject
            rays, _ = cam.unproject(cam.c.unsqueeze(1))
            self.assertTrue(
                torch.allclose(
                    torch.norm(rays[..., :2], p=2, dim=-1), torch.zeros(1, 1)
                )
            )

            if i == 1:
                # Project cam1 rays in to cam0 to make sure the cam1 and cam0 are different. fx != fy is picked properly.
                pixels_cam0, _ = cam[0].project(rays)
                self.assertFalse(
                    torch.allclose(pixels, pixels_cam0, rtol=1e-3, atol=1e-6)
                )

    def test_backward_unproj(self):
        B, N = 10, 1000
        cams = []
        cams.append(get_aria_camera(RGB_PARAMS, B=B))
        cams.append(get_aria_camera(RGB_PARAMS2, B=B))

        for cam in cams:
            pixels = torch.rand(B, N, 2)
            pixels.requires_grad = True

            # NOTE(dd): If dealing with a JIT complaint about in-place operation,
            # uncomment "@torch.jit.script" above fisheye624_unproj to get offending line.
            with torch.autograd.set_detect_anomaly(True):
                rays, _ = cam.unproject(pixels)
                placeholder_grads1 = torch.ones(B, N, 3)
                rays.backward(placeholder_grads1)

    def test_backward_proj(self):
        B, N = 10, 1000
        cams = []
        cams.append(get_aria_camera(RGB_PARAMS, B=B))
        cams.append(get_aria_camera(RGB_PARAMS2, B=B))

        for cam in cams:
            rays = torch.rand(B, N, 3)
            rays.requires_grad = True

            # NOTE(dd): If dealing with a JIT complaint about in-place operation,
            # uncomment "@torch.jit.script" above fisheye624_proj to get offending line.
            with torch.autograd.set_detect_anomaly(True):
                pixels, _ = cam.project(rays)
                placeholder_grads2 = torch.ones(B, N, 2)
                pixels.backward(placeholder_grads2)

    def test_zero_depth(self):
        B, N = 2, 20
        cam = get_aria_camera(RGB_PARAMS, B=B)
        p3d = torch.zeros(B, N, 3)
        p3d[:, :, 2] += 4  # Create some 3d  that will definitely have valid projection.
        _, valid = cam.project(p3d)
        self.assertTrue(torch.all(valid))
        p3d[1, 9, 2] = 0  # Set depth of one of those points to be zero.
        _, valid2 = cam.project(p3d)
        self.assertTrue(bool(valid2[1, 9]) is False)

    def test_in_radius(self):
        H, W = 240, 240
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H))
        pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
        cam_orig = get_base_aria_rgb_camera()
        cam = cam_orig.scale_to_size((W, H))
        valid = cam.in_radius(pts)
        num_valid = valid.sum()
        self.assertTrue(num_valid > 0)
        self.assertTrue(num_valid < pts.shape[0])

        # Optionally render a debug image.
        render = False
        # render = True
        if render:
            viz = np.zeros((H, W, 3)).astype(np.uint8)
            for pt, val in zip(pts, valid):
                pxy = int(round(float(pt[1]))), int(round(float(pt[0])))
                cv2.circle(viz, pxy, 1, (0, 0, 255), -1, lineType=16)
                if val:
                    cv2.circle(viz, pxy, 3, (0, 255, 0), 2, lineType=16)
            out_path = os.path.expanduser("~/viz_test_in_radius.png")
            cv2.imwrite(out_path, viz)
            print("Wrote debug image to %s" % out_path)

    def test_dtype_check(self):
        cam = get_aria_camera(SLAM_PARAMS, height=480, width=640)

        # Test unproject.
        pts0 = torch.zeros(1, 1, 2, device="cpu").double()
        assert_hit = False
        try:
            _, _ = cam.unproject(pts0.float())
        except Exception:
            assert_hit = True
        # Exceptions are not passed because of jit.script so we use this hacky test.
        self.assertFalse(assert_hit)

        rays0 = torch.zeros(1, 1, 3, device="cpu").double()
        # Test project.
        assert_hit = False
        try:
            _, _ = cam.project(rays0.float())
        except Exception:
            assert_hit = True
        # Exceptions are not passed because of jit.script so we use this hacky test.
        self.assertFalse(assert_hit)

    def test_kb4_fish(self):
        torch.manual_seed(0)
        B = 4
        N = 1000
        cam = get_kb4_camera(B)
        radius = cam.valid_radius[0, 0].item()
        pixels = random_fisheye_pixels(B, N, radius, cam[0].c.numpy())
        pixels = torch.tensor(pixels)
        pixels = pixels
        rays, _ = cam.unproject(pixels)
        pixels2, _ = cam.project(rays)
        # TODO(dd): work on this.
        # self.assertTrue(torch.allclose(pixels, pixels2, rtol=1e-3, atol=1e-6))

    def test_kb4_grid(self):
        B = 1
        # cam = get_kb4_camera(B)
        # TODO(dd): why does this special case fail??
        data = torch.tensor(
            [
                8.9600e02,
                8.9600e02,
                3.4991e02,
                4.6655e02,
                4.7180e02,
                3.7373e02,
                1.0000e00,
                1.0000e-03,
                4.4022e02,
                4.4022e02,
                9.9999e-01,
                -3.6686e-04,
                5.1772e-03,
                3.2707e-04,
                9.9997e-01,
                7.6839e-03,
                -5.1799e-03,
                -7.6821e-03,
                9.9996e-01,
                -2.2368e-04,
                6.8510e-05,
                -9.0108e-04,
                1.0000e00,
                2.4284e-01,
                8.9151e-03,
                -9.6023e-02,
            ]
        )
        cam = CameraTW(data)
        cam = cam.unsqueeze(0).repeat(B, 1)

        HH = int(cam.size[0, 1].item())
        WW = int(cam.size[0, 0].item())
        patch_size = 16
        fH = int(HH // patch_size)
        fW = int(WW // patch_size)

        pixels = generate_patch_centers(
            B=B, fH=fH, fW=fW, patch_size=patch_size, device="cpu"
        )

        rays, _ = cam.unproject(pixels)
        pixels2, _ = cam.project(rays)
        # TODO(dd): why does this special case fail? look into numerical stability of unproject for kb4.
        # self.assertTrue(torch.allclose(pixels, pixels2, rtol=1e-3, atol=1e-6))


class TestOmniLoaderProjection(unittest.TestCase):
    """
    Test to reproduce the "Global alloc not supported yet" error that occurs
    when projecting OBB corner points through a fisheye camera in OmniLoader.

    The error occurs in fisheye624_project -> sign_plus when:
    1. @torch.jit.script is enabled on fisheye624_project
    2. sign_plus uses in-place boolean indexing: sgn[sgn < 0.0] = -1.0
    3. Certain tensor shapes are used (e.g., [17, 120, 3] for 17 OBBs x 12 edges x 10 segments)
    """

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_fisheye624_project_torchscript_directly(self):
        """
        Call fisheye624_project directly to ensure TorchScript is actually invoked.
        This bypasses CameraTW and tests the raw JIT-compiled function.
        """
        from utils.tw.camera import fisheye624_project

        # Shape from OmniLoader: [17, 120, 3] (17 OBBs x 120 edge points)
        B, N = 17, 120
        xyz = torch.randn(B, N, 3, dtype=torch.float32)
        xyz[:, :, 2] = torch.abs(xyz[:, :, 2]) + 1.0  # Ensure positive z

        # Fisheye624 params: [fx, fy, cx, cy, k0-k5, p0, p1, s0-s3] = 16 elements
        params = torch.zeros(B, 16, dtype=torch.float32)
        params[:, 0] = 500.0  # fx
        params[:, 1] = 500.0  # fy
        params[:, 2] = 256.0  # cx
        params[:, 3] = 192.0  # cy
        # k0-k5, p0, p1, s0-s3 all zeros (no distortion)

        print("\n[TEST] Calling fisheye624_project directly with TorchScript")
        print(f"[TEST] xyz.shape={xyz.shape}, params.shape={params.shape}")

        # This should use the @torch.jit.script compiled version
        result = fisheye624_project(xyz, params)

        print(f"[TEST] Result shape: {result.shape}")
        self.assertEqual(result.shape, (B, N, 2))

    def test_omniloader_obb_projection(self):
        """
        Reproduce the exact scenario from OmniLoader + draw_bb3_lines:
        - Create a fisheye camera like OmniLoader does (get_base_aria_rgb_camera with modified params)
        - Create 3D points with shape [B, N, 3] where B=17 (OBBs) and N=120 (12 edges * 10 segments)
        - Call cam.project() which triggers fisheye624_project -> sign_plus
        """
        # Simulate OmniLoader's camera creation
        cam = get_base_aria_rgb_camera()

        # Modify camera params like OmniLoader does
        resizeW, resizeH = 512, 384
        fx, fy = 500.0, 500.0
        cx, cy = resizeW / 2, resizeH / 2

        cam._data[0] = resizeW
        cam._data[1] = resizeH
        cam._data[2] = fx * 1.15
        cam._data[3] = fy * 1.15
        cam._data[4] = cx
        cam._data[5] = cy

        # Simulate draw_bb3_lines scenario:
        # B = number of OBBs (e.g., 17)
        # N = 12 edges * T segments (e.g., 12 * 10 = 120)
        B = 17  # Number of OBBs like in OmniLoader
        num_edges = 12
        T = 10  # render_obb_corner_steps
        N = num_edges * T  # 120 points per OBB

        # Create 3D points in camera coordinates (like bb3corners_cam in draw_bb3_lines)
        # Points should be in front of camera (positive z)
        pt3s_cam = torch.randn(B, N, 3, dtype=torch.float32)
        pt3s_cam[:, :, 2] = torch.abs(pt3s_cam[:, :, 2]) + 1.0  # Ensure positive z

        print(
            f"\n[TEST] pt3s_cam.shape={pt3s_cam.shape}, dtype={pt3s_cam.dtype}, device={pt3s_cam.device}"
        )
        print(f"[TEST] cam._data.shape={cam._data.shape}, dtype={cam._data.dtype}")

        # This should trigger the error if @torch.jit.script is on fisheye624_project
        # and sign_plus uses in-place boolean indexing
        pt2s, valids = cam.project(pt3s_cam)

        print(f"[TEST] pt2s.shape={pt2s.shape}")
        print(f"[TEST] valids.shape={valids.shape}")

        # Basic validation
        self.assertEqual(pt2s.shape, (B, N, 2))
        self.assertEqual(valids.shape, (B, N))

    def test_omniloader_single_obb(self):
        """Test with a single OBB (B=1) - this typically works fine."""
        cam = get_base_aria_rgb_camera()
        B, N = 1, 120
        pt3s_cam = torch.randn(B, N, 3, dtype=torch.float32)
        pt3s_cam[:, :, 2] = torch.abs(pt3s_cam[:, :, 2]) + 1.0

        pt2s, valids = cam.project(pt3s_cam)
        self.assertEqual(pt2s.shape, (B, N, 2))

    def test_omniloader_many_obbs(self):
        """Test with many OBBs (B=50) - stress test."""
        cam = get_base_aria_rgb_camera()
        B, N = 50, 120
        pt3s_cam = torch.randn(B, N, 3, dtype=torch.float32)
        pt3s_cam[:, :, 2] = torch.abs(pt3s_cam[:, :, 2]) + 1.0

        pt2s, valids = cam.project(pt3s_cam)
        self.assertEqual(pt2s.shape, (B, N, 2))

    def test_omniloader_cpu_vs_cuda(self):
        """Test projection on CPU (where the error occurs) vs CUDA."""
        cam = get_base_aria_rgb_camera()
        B, N = 17, 120
        pt3s_cam = torch.randn(B, N, 3, dtype=torch.float32)
        pt3s_cam[:, :, 2] = torch.abs(pt3s_cam[:, :, 2]) + 1.0

        # Test on CPU
        pt2s_cpu, valids_cpu = cam.project(pt3s_cam)
        self.assertEqual(pt2s_cpu.shape, (B, N, 2))

        # Test on CUDA if available
        if torch.cuda.is_available():
            cam_cuda = cam.cuda()
            pt3s_cuda = pt3s_cam.cuda()
            pt2s_cuda, valids_cuda = cam_cuda.project(pt3s_cuda)
            self.assertEqual(pt2s_cuda.shape, (B, N, 2))
            # Results should be similar
            self.assertTrue(
                torch.allclose(pt2s_cpu, pt2s_cuda.cpu(), rtol=1e-4, atol=1e-4)
            )


if __name__ == "__main__":
    unittest.main()
