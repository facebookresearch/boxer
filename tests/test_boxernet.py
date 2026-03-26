#!/usr/bin/env python3

# pyre-unsafe

"""
Simple unit tests for boxernet/boxernet.py and boxernet/dinov3_wrapper.py.
Tests pure functions with synthetic inputs (no checkpoint needed).
"""

import unittest

import torch


class TestImageToPatches(unittest.TestCase):
    def test_basic_shape(self):
        from boxernet.boxernet import image_to_patches

        B, C, H, W = 2, 1, 32, 48
        patch_size = 16
        x = torch.randn(B, C, H, W)
        patches = image_to_patches(x, patch_size)
        num_patches = (H // patch_size) * (W // patch_size)
        self.assertEqual(patches.shape, (B, num_patches, patch_size * patch_size))

    def test_values_identity(self):
        """Each patch should contain the correct pixels from the input."""
        from boxernet.boxernet import image_to_patches

        x = torch.arange(16 * 16).float().reshape(1, 1, 16, 16)
        patches = image_to_patches(x, patch_size=16)
        # Single patch containing all values
        self.assertEqual(patches.shape, (1, 1, 256))
        self.assertTrue(torch.allclose(patches[0, 0], x.reshape(-1)))

    def test_not_divisible_raises(self):
        from boxernet.boxernet import image_to_patches

        x = torch.randn(1, 1, 15, 16)
        with self.assertRaises(AssertionError):
            image_to_patches(x, patch_size=16)


class TestMaskedMedian(unittest.TestCase):
    def test_all_valid(self):
        from boxernet.boxernet import masked_median

        x = torch.tensor([[1.0, 3.0, 2.0]])
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask, dim=1)
        self.assertAlmostEqual(result.item(), 2.0)

    def test_partial_mask(self):
        from boxernet.boxernet import masked_median

        x = torch.tensor([[10.0, 1.0, 5.0]])
        mask = torch.tensor([[True, False, True]])
        result = masked_median(x, mask, dim=1)
        # Valid values: [10, 5], sorted: [5, 10], median index (2-1)//2=0 → 5.0
        self.assertAlmostEqual(result.item(), 5.0)

    def test_single_valid(self):
        from boxernet.boxernet import masked_median

        x = torch.tensor([[0.0, 42.0, 0.0]])
        mask = torch.tensor([[False, True, False]])
        result = masked_median(x, mask, dim=1)
        self.assertAlmostEqual(result.item(), 42.0)


class TestGeneratePatchCenters(unittest.TestCase):
    def test_shape(self):
        from boxernet.boxernet import generate_patch_centers

        B, fH, fW, patch_size = 3, 4, 5, 16
        uv = generate_patch_centers(B, fH, fW, patch_size, "cpu")
        self.assertEqual(uv.shape, (B, fH * fW, 2))

    def test_first_center(self):
        from boxernet.boxernet import generate_patch_centers

        uv = generate_patch_centers(1, 2, 3, 16, "cpu")
        # First patch center should be at (patch_size/2, patch_size/2)
        self.assertAlmostEqual(uv[0, 0, 0].item(), 8.0)  # x
        self.assertAlmostEqual(uv[0, 0, 1].item(), 8.0)  # y

    def test_batch_identical(self):
        from boxernet.boxernet import generate_patch_centers

        uv = generate_patch_centers(4, 2, 2, 16, "cpu")
        # All batch elements should be identical
        self.assertTrue(torch.allclose(uv[0], uv[1]))
        self.assertTrue(torch.allclose(uv[0], uv[3]))


class TestSmartLoad(unittest.TestCase):
    def test_exact_match(self):
        from boxernet.boxernet import smart_load

        model_dict = {"a": torch.randn(3, 4), "b": torch.randn(5)}
        ckpt_dict = {"a": torch.randn(3, 4), "b": torch.randn(5)}
        result = smart_load(model_dict, ckpt_dict)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_shape_mismatch_skipped(self):
        from boxernet.boxernet import smart_load

        model_dict = {"a": torch.randn(3, 4)}
        ckpt_dict = {"a": torch.randn(5, 4)}  # different shape
        result = smart_load(model_dict, ckpt_dict)
        self.assertNotIn("a", result)

    def test_strips_orig_mod_prefix(self):
        from boxernet.boxernet import smart_load

        model_dict = {"layer.weight": torch.randn(3, 4)}
        ckpt_dict = {"_orig_mod.layer.weight": torch.randn(3, 4)}
        result = smart_load(model_dict, ckpt_dict)
        self.assertIn("layer.weight", result)

    def test_extra_ckpt_keys_ignored(self):
        from boxernet.boxernet import smart_load

        model_dict = {"a": torch.randn(2)}
        ckpt_dict = {"a": torch.randn(2), "extra": torch.randn(3)}
        result = smart_load(model_dict, ckpt_dict)
        self.assertIn("a", result)
        self.assertNotIn("extra", result)


class TestMake2Tuple(unittest.TestCase):
    def test_int(self):
        from boxernet.dinov3_wrapper import make_2tuple

        self.assertEqual(make_2tuple(5), (5, 5))

    def test_tuple_passthrough(self):
        from boxernet.dinov3_wrapper import make_2tuple

        self.assertEqual(make_2tuple((3, 7)), (3, 7))

    def test_wrong_tuple_length_raises(self):
        from boxernet.dinov3_wrapper import make_2tuple

        with self.assertRaises(AssertionError):
            make_2tuple((1, 2, 3))


class TestDinoConstants(unittest.TestCase):
    def test_output_dims_positive(self):
        from boxernet.dinov3_wrapper import DINOV3_OUTPUT_DIM

        for name, dim in DINOV3_OUTPUT_DIM.items():
            self.assertGreater(dim, 0, f"{name} should have positive output dim")

    def test_all_models_have_func_and_dim(self):
        from boxernet.dinov3_wrapper import DINOV3_MODEL_FUNC, DINOV3_OUTPUT_DIM

        self.assertEqual(set(DINOV3_MODEL_FUNC.keys()), set(DINOV3_OUTPUT_DIM.keys()))

    def test_vits16plus_is_384(self):
        from boxernet.dinov3_wrapper import DINOV3_OUTPUT_DIM

        self.assertEqual(DINOV3_OUTPUT_DIM["dinov3_vits16plus"], 384)


if __name__ == "__main__":
    unittest.main()
