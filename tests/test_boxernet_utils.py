# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for boxernet/boxernet.py utility functions (no model weights needed)."""

import pytest
import torch

from boxernet.boxernet import generate_patch_centers, image_to_patches, masked_median


class TestImageToPatches:
    def test_basic_shape(self):
        x = torch.randn(2, 1, 28, 42)  # 2x3 patches of size 14
        patches = image_to_patches(x, patch_size=14)
        assert patches.shape == (2, 6, 14 * 14)

    def test_single_patch(self):
        x = torch.randn(1, 1, 14, 14)
        patches = image_to_patches(x, patch_size=14)
        assert patches.shape == (1, 1, 196)

    def test_values_preserved(self):
        x = torch.zeros(1, 1, 14, 28)
        x[0, 0, :, :14] = 1.0  # left patch all ones
        patches = image_to_patches(x, patch_size=14)
        assert patches[0, 0].sum().item() == 14 * 14
        assert patches[0, 1].sum().item() == 0

    def test_non_divisible_raises(self):
        x = torch.randn(1, 1, 15, 14)
        with pytest.raises(AssertionError):
            image_to_patches(x, patch_size=14)

    def test_different_patch_size(self):
        x = torch.randn(1, 1, 16, 32)
        patches = image_to_patches(x, patch_size=16)
        assert patches.shape == (1, 2, 256)


class TestMaskedMedian:
    def test_all_valid(self):
        x = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask, dim=1)
        assert result.item() == 5.0

    def test_partial_mask(self):
        x = torch.tensor([[1.0, 100.0, 3.0, 200.0, 5.0]])
        mask = torch.tensor([[True, False, True, False, True]])
        result = masked_median(x, mask, dim=1)
        assert result.item() == 3.0  # median of [1, 3, 5]

    def test_single_valid(self):
        x = torch.tensor([[0.0, 0.0, 42.0, 0.0]])
        mask = torch.tensor([[False, False, True, False]])
        result = masked_median(x, mask, dim=1)
        assert result.item() == 42.0

    def test_batch(self):
        x = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ]
        )
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask, dim=1)
        assert result[0].item() == 2.0
        assert result[1].item() == 20.0

    def test_unsupported_dim_raises(self):
        x = torch.randn(3, 4)
        mask = torch.ones_like(x, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            masked_median(x, mask, dim=0)


class TestGeneratePatchCenters:
    def test_shape(self):
        uv = generate_patch_centers(B=2, fH=3, fW=4, patch_size=14, device="cpu")
        assert uv.shape == (2, 12, 2)

    def test_batch_identical(self):
        uv = generate_patch_centers(B=3, fH=2, fW=2, patch_size=16, device="cpu")
        assert torch.equal(uv[0], uv[1])
        assert torch.equal(uv[1], uv[2])

    def test_first_center(self):
        uv = generate_patch_centers(B=1, fH=2, fW=3, patch_size=14, device="cpu")
        # First patch center should be at (patch_size/2, patch_size/2)
        assert uv[0, 0, 0].item() == 7.0  # x = 14/2
        assert uv[0, 0, 1].item() == 7.0  # y = 14/2

    def test_spacing(self):
        uv = generate_patch_centers(B=1, fH=1, fW=3, patch_size=16, device="cpu")
        # Centers should be spaced by patch_size
        dx = uv[0, 1, 0] - uv[0, 0, 0]
        assert dx.item() == 16.0

    def test_single_patch(self):
        uv = generate_patch_centers(B=1, fH=1, fW=1, patch_size=14, device="cpu")
        assert uv.shape == (1, 1, 2)
        assert uv[0, 0, 0].item() == 7.0
        assert uv[0, 0, 1].item() == 7.0
