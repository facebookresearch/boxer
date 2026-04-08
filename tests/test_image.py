# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utils/image.py — image conversion and normalization utilities."""

import numpy as np
import pytest
import torch

from utils.image import normalize, put_text, rotate_image90, string2color, torch2cv2


class TestString2Color:
    def test_known_colors(self):
        assert string2color("white") == (255, 255, 255)
        assert string2color("green") == (0, 255, 0)
        assert string2color("red") == (0, 0, 255)
        assert string2color("black") == (0, 0, 0)
        assert string2color("blue") == (255, 0, 0)

    def test_case_insensitive(self):
        assert string2color("White") == (255, 255, 255)
        assert string2color("RED") == (0, 0, 255)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            string2color("magenta")


class TestNormalize:
    def test_numpy_basic(self):
        img = np.array([0.0, 5.0, 10.0])
        result = normalize(img)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-5)

    def test_torch_basic(self):
        img = torch.tensor([0.0, 5.0, 10.0])
        result = normalize(img)
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)

    def test_constant_image(self):
        img = np.ones((10, 10)) * 5.0
        result = normalize(img)
        # All same value, eps prevents div by zero, result should be clamped
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_robust_quantile(self):
        img = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = normalize(img, robust=0.1)
        # With robust quantile, the outlier 100.0 should be clipped to 1.0
        assert result[-1] == 1.0

    def test_output_range(self):
        img = np.random.randn(50, 50)
        result = normalize(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestTorch2Cv2:
    def test_basic_conversion(self):
        img = torch.rand(3, 32, 64)
        result = torch2cv2(img)
        assert result.shape == (32, 64, 3)
        assert result.dtype == np.uint8

    def test_no_rgb2bgr(self):
        # Create image with known channel values
        img = torch.zeros(3, 4, 4)
        img[0] = 1.0  # R channel
        result = torch2cv2(img, rgb2bgr=False)
        assert result[0, 0, 0] == 255  # R stays first
        assert result[0, 0, 2] == 0

    def test_rgb2bgr(self):
        img = torch.zeros(3, 4, 4)
        img[0] = 1.0  # R channel
        result = torch2cv2(img, rgb2bgr=True)
        assert result[0, 0, 2] == 255  # R moved to last channel (BGR)
        assert result[0, 0, 0] == 0

    def test_grayscale_2d(self):
        img = torch.rand(8, 16)
        result = torch2cv2(img, rgb2bgr=False)
        assert result.shape == (8, 16, 1)

    def test_ensure_rgb_single_channel(self):
        img = torch.rand(1, 8, 16)
        result = torch2cv2(img, rgb2bgr=False, ensure_rgb=True)
        assert result.shape == (8, 16, 3)

    def test_batch_dim_stripped(self):
        img = torch.rand(2, 3, 8, 8)
        result = torch2cv2(img, rgb2bgr=False)
        assert result.shape == (8, 8, 3)  # first batch element used

    def test_rotate(self):
        img = torch.rand(3, 8, 16)
        result = torch2cv2(img, rotate=True, rgb2bgr=False)
        assert result.shape == (16, 8, 3)  # 90 deg rotation swaps H and W

    def test_numpy_input(self):
        img = np.random.rand(3, 8, 8).astype(np.float32)
        result = torch2cv2(img, rgb2bgr=False)
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.uint8


class TestRotateImage90:
    def test_single_rotation(self):
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        rotated = rotate_image90(img, k=1)
        assert rotated.shape == (8, 4, 3)

    def test_double_rotation(self):
        img = np.zeros((4, 8), dtype=np.uint8)
        rotated = rotate_image90(img, k=2)
        assert rotated.shape == (4, 8)  # 180 deg preserves shape

    def test_contiguous(self):
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        rotated = rotate_image90(img, k=1)
        assert rotated.flags["C_CONTIGUOUS"]

    def test_identity(self):
        img = np.arange(12).reshape(3, 4)
        rotated = rotate_image90(img, k=4)
        np.testing.assert_array_equal(rotated, img)


class TestPutText:
    def test_basic(self):
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        result = put_text(img, "hello")
        assert result.shape == img.shape
        # Text should have been drawn (some pixels non-zero)
        assert result.sum() > 0

    def test_truncate(self):
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        result = put_text(img, "a very long text string", truncate=5)
        assert result.shape == img.shape

    def test_batch(self):
        imgs = np.zeros((2, 320, 640, 3), dtype=np.uint8)
        result = put_text(imgs, "test")
        assert result.shape == (2, 320, 640, 3)

    def test_color_string(self):
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        result = put_text(img, "hello", color="green")
        assert result.sum() > 0
