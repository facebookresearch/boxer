#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import numpy as np
import torch

from utils.tw.tensor_wrapper import (
    TensorWrapper,
    autocast,
    smart_cat,
    smart_stack,
)


class TestTensorWrapperBasics(unittest.TestCase):
    """Tests for basic TensorWrapper functionality."""

    def test_constructor(self):
        """Test basic construction."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        self.assertEqual(tw.shape, (10, 3))
        self.assertTrue(torch.allclose(tw._data, data))

    def test_constructor_numpy_input(self):
        """Test construction from numpy array via autocast."""
        data_np = np.random.randn(10, 3).astype(np.float32)
        tw = TensorWrapper(data_np)
        self.assertEqual(tw.shape, (10, 3))
        self.assertTrue(torch.is_tensor(tw._data))

    def test_properties(self):
        """Test basic properties."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        self.assertEqual(tw.shape, data.shape)
        self.assertEqual(tw.device, data.device)
        self.assertEqual(tw.dtype, data.dtype)
        self.assertEqual(tw.ndim, data.ndim)
        self.assertEqual(tw.dim(), data.dim())
        self.assertEqual(tw.nelement(), data.nelement())
        self.assertEqual(tw.numel(), data.numel())

    def test_is_cuda(self):
        """Test is_cuda property."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        self.assertEqual(tw.is_cuda, data.is_cuda)
        self.assertFalse(tw.is_cuda)

    def test_is_contiguous(self):
        """Test is_contiguous property."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        # Should return bool from the property
        self.assertTrue(tw.is_contiguous)

    def test_requires_grad(self):
        """Test requires_grad property and setter."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        self.assertFalse(tw.requires_grad)
        tw.requires_grad_(True)
        self.assertTrue(tw.requires_grad)
        tw.requires_grad_(False)
        self.assertFalse(tw.requires_grad)

    def test_grad_properties(self):
        """Test grad and grad_fn properties."""
        data = torch.randn(10, 3, requires_grad=True)
        tw = TensorWrapper(data)
        self.assertIsNone(tw.grad)
        self.assertIsNone(tw.grad_fn)


class TestTensorWrapperIndexing(unittest.TestCase):
    """Tests for TensorWrapper indexing operations."""

    def test_getitem_single(self):
        """Test single element indexing."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        result = tw[0]
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(torch.allclose(result._data, data[0]))

    def test_getitem_slice(self):
        """Test slice indexing."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        result = tw[2:5]
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.allclose(result._data, data[2:5]))

    def test_setitem_tensor(self):
        """Test setting item with tensor."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        new_val = torch.ones(3)

        tw[0] = new_val
        self.assertTrue(torch.allclose(tw[0]._data, new_val))

    def test_setitem_tensor_wrapper(self):
        """Test setting item with TensorWrapper."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        new_val = TensorWrapper(torch.ones(3))

        tw[0] = new_val
        self.assertTrue(torch.allclose(tw[0]._data, new_val._data))

    def test_len(self):
        """Test __len__."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)
        self.assertEqual(len(tw), 10)


class TestTensorWrapperOperations(unittest.TestCase):
    """Tests for TensorWrapper tensor operations."""

    def test_to(self):
        """Test to() method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        # Change dtype
        tw_double = tw.to(dtype=torch.float64)
        self.assertIsInstance(tw_double, TensorWrapper)
        self.assertEqual(tw_double.dtype, torch.float64)

    def test_reshape(self):
        """Test reshape method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        reshaped = tw.reshape(2, 5, 3)
        self.assertIsInstance(reshaped, TensorWrapper)
        self.assertEqual(reshaped.shape, (2, 5, 3))

    def test_repeat(self):
        """Test repeat method."""
        data = torch.randn(2, 3)
        tw = TensorWrapper(data)

        repeated = tw.repeat(3, 1)
        self.assertIsInstance(repeated, TensorWrapper)
        self.assertEqual(repeated.shape, (6, 3))

    def test_expand(self):
        """Test expand method."""
        data = torch.randn(1, 3)
        tw = TensorWrapper(data)

        expanded = tw.expand(5, 3)
        self.assertIsInstance(expanded, TensorWrapper)
        self.assertEqual(expanded.shape, (5, 3))

    def test_clone(self):
        """Test clone method creates independent copy."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        cloned = tw.clone()
        self.assertIsInstance(cloned, TensorWrapper)
        self.assertTrue(torch.allclose(tw._data, cloned._data))

        # Modify original
        tw._data[0] = 999.0
        self.assertFalse(torch.allclose(tw._data, cloned._data))

    def test_cpu(self):
        """Test cpu method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        tw_cpu = tw.cpu()
        self.assertIsInstance(tw_cpu, TensorWrapper)
        self.assertEqual(tw_cpu.device.type, "cpu")

    def test_cuda_without_gpu(self):
        """Test cuda method returns same type."""
        if torch.cuda.is_available():
            data = torch.randn(10, 3)
            tw = TensorWrapper(data)
            tw_cuda = tw.cuda()
            self.assertIsInstance(tw_cuda, TensorWrapper)
            self.assertTrue(tw_cuda.is_cuda)

    def test_contiguous(self):
        """Test contiguous method."""
        data = torch.randn(10, 3).t()  # Non-contiguous
        tw = TensorWrapper(data)

        tw_contig = tw.contiguous()
        self.assertIsInstance(tw_contig, TensorWrapper)
        self.assertTrue(tw_contig._data.is_contiguous())

    def test_pin_memory(self):
        """Test pin_memory method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        # Only works if CUDA is available
        if torch.cuda.is_available():
            tw_pinned = tw.pin_memory()
            self.assertIsInstance(tw_pinned, TensorWrapper)
            self.assertTrue(tw_pinned._data.is_pinned())

    def test_float(self):
        """Test float method."""
        data = torch.randn(10, 3).double()
        tw = TensorWrapper(data)

        tw_float = tw.float()
        self.assertIsInstance(tw_float, TensorWrapper)
        self.assertEqual(tw_float.dtype, torch.float32)

    def test_double(self):
        """Test double method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        tw_double = tw.double()
        self.assertIsInstance(tw_double, TensorWrapper)
        self.assertEqual(tw_double.dtype, torch.float64)

    def test_detach(self):
        """Test detach method."""
        data = torch.randn(10, 3, requires_grad=True)
        tw = TensorWrapper(data)

        detached = tw.detach()
        self.assertIsInstance(detached, TensorWrapper)
        self.assertFalse(detached.requires_grad)

    def test_numpy_conversion(self):
        """Test numpy method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        np_array = tw.numpy()
        self.assertIsInstance(np_array, np.ndarray)
        self.assertTrue(np.allclose(np_array, data.numpy()))

    def test_tensor_method(self):
        """Test tensor method returns underlying data."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        result = tw.tensor()
        self.assertTrue(torch.is_tensor(result))
        self.assertTrue(result is tw._data)

    def test_tolist(self):
        """Test tolist method."""
        data = torch.randn(2, 3)
        tw = TensorWrapper(data)

        result = tw.tolist()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


class TestTensorWrapperSqueezeUnsqueeze(unittest.TestCase):
    """Tests for squeeze and unsqueeze methods."""

    def test_squeeze_no_dim(self):
        """Test squeeze without dimension argument."""
        data = torch.randn(1, 10, 1, 3)
        tw = TensorWrapper(data)

        squeezed = tw.squeeze()
        self.assertEqual(squeezed.shape, (10, 3))

    def test_squeeze_with_dim(self):
        """Test squeeze with dimension argument."""
        data = torch.randn(1, 10, 3)
        tw = TensorWrapper(data)

        squeezed = tw.squeeze(dim=0)
        self.assertEqual(squeezed.shape, (10, 3))

    def test_squeeze_dim_negative_one(self):
        """Test squeeze with dim=-1 should raise or handle properly."""
        data = torch.randn(10, 1, 3)
        tw = TensorWrapper(data)

        # The current assertion prevents dim=-1
        # This test documents current behavior
        with self.assertRaises(AssertionError):
            tw.squeeze(dim=-1)

    def test_unsqueeze(self):
        """Test unsqueeze method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        unsqueezed = tw.unsqueeze(dim=0)
        self.assertEqual(unsqueezed.shape, (1, 10, 3))

    def test_unsqueeze_dim_negative_one(self):
        """Test unsqueeze with dim=-1 should raise or handle properly."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        # The current assertion prevents dim=-1
        with self.assertRaises(AssertionError):
            tw.unsqueeze(dim=-1)


class TestTensorWrapperView(unittest.TestCase):
    """Tests for view method."""

    def test_view(self):
        """Test view method."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        viewed = tw.view(2, 5, 3)
        self.assertEqual(viewed.shape, (2, 5, 3))

    def test_view_with_negative_one(self):
        """Test view with -1 for last dim."""
        data = torch.randn(10, 3)
        tw = TensorWrapper(data)

        # -1 preserves the last dimension (must equal original last dim)
        viewed = tw.view(-1, 3)
        self.assertEqual(viewed.shape, (10, 3))

        # Can also use -1 to infer the first dimension
        viewed2 = tw.view(5, 2, -1)
        self.assertEqual(viewed2.shape, (5, 2, 3))


class TestSmartCatStack(unittest.TestCase):
    """Tests for smart_cat and smart_stack functions."""

    def test_smart_cat_tensors(self):
        """Test smart_cat with regular tensors."""
        tensors = [torch.randn(5, 3) for _ in range(3)]
        result = smart_cat(tensors, dim=0)
        self.assertEqual(result.shape, (15, 3))

    def test_smart_cat_tensor_wrappers(self):
        """Test smart_cat with TensorWrapper objects."""
        wrappers = [TensorWrapper(torch.randn(5, 3)) for _ in range(3)]
        result = smart_cat(wrappers, dim=0)
        self.assertEqual(result.shape, (15, 3))

    def test_smart_cat_mixed(self):
        """Test smart_cat with mixed tensors and TensorWrappers."""
        items = [
            torch.randn(5, 3),
            TensorWrapper(torch.randn(5, 3)),
            torch.randn(5, 3),
        ]
        result = smart_cat(items, dim=0)
        self.assertEqual(result.shape, (15, 3))

    def test_smart_stack_tensors(self):
        """Test smart_stack with regular tensors."""
        tensors = [torch.randn(5, 3) for _ in range(3)]
        result = smart_stack(tensors, dim=0)
        self.assertEqual(result.shape, (3, 5, 3))

    def test_smart_stack_tensor_wrappers(self):
        """Test smart_stack with TensorWrapper objects."""
        wrappers = [TensorWrapper(torch.randn(5, 3)) for _ in range(3)]
        result = smart_stack(wrappers, dim=0)
        self.assertEqual(result.shape, (3, 5, 3))

    def test_smart_cat_multi_device_raises(self):
        """Test that smart_cat raises on multi-device input."""
        if torch.cuda.is_available():
            tensors = [torch.randn(5, 3), torch.randn(5, 3).cuda()]
            with self.assertRaises(RuntimeError):
                smart_cat(tensors, dim=0)


class TestTorchFunction(unittest.TestCase):
    """Tests for __torch_function__ protocol."""

    def test_torch_stack(self):
        """Test torch.stack with TensorWrappers."""
        wrappers = [TensorWrapper(torch.randn(3)) for _ in range(5)]
        result = torch.stack(wrappers, dim=0)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (5, 3))

    def test_torch_cat(self):
        """Test torch.cat with TensorWrappers."""
        wrappers = [TensorWrapper(torch.randn(1, 3)) for _ in range(5)]
        result = torch.cat(wrappers, dim=0)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (5, 3))

    def test_torch_allclose(self):
        """Test torch.allclose with TensorWrappers."""
        tw1 = TensorWrapper(torch.tensor([1.0, 2.0, 3.0]))
        tw2 = TensorWrapper(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(tw1, tw2))

        tw3 = TensorWrapper(torch.tensor([1.0, 2.0, 4.0]))
        self.assertFalse(torch.allclose(tw1, tw3))

    def test_torch_take_along_dim(self):
        """Test torch.take_along_dim with TensorWrapper."""
        data = torch.randn(3, 4, 5)
        tw = TensorWrapper(data)
        indices = torch.randint(0, 5, (3, 4, 2))

        result = torch.take_along_dim(tw, indices, dim=-1)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (3, 4, 2))

    def test_torch_flatten(self):
        """Test torch.flatten with TensorWrapper."""
        tw = TensorWrapper(torch.randn(2, 3, 4))

        result = torch.flatten(tw, start_dim=0, end_dim=1)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (6, 4))


class TestClassMethods(unittest.TestCase):
    """Tests for class methods."""

    def test_stack_classmethod(self):
        """Test stack class method."""
        wrappers = [TensorWrapper(torch.randn(3)) for _ in range(5)]
        result = TensorWrapper.stack(wrappers, dim=0)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (5, 3))

    def test_cat_classmethod(self):
        """Test cat class method."""
        wrappers = [TensorWrapper(torch.randn(2, 3)) for _ in range(5)]
        result = TensorWrapper.cat(wrappers, dim=0)
        self.assertIsInstance(result, TensorWrapper)
        self.assertEqual(result.shape, (10, 3))

    def test_allclose_classmethod(self):
        """Test allclose class method."""
        tw1 = TensorWrapper(torch.tensor([1.0, 2.0, 3.0]))
        tw2 = TensorWrapper(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(TensorWrapper.allclose(tw1, tw2))


class TestAutocastDecorator(unittest.TestCase):
    """Tests for @autocast decorator."""

    def test_autocast_preserves_device_dtype(self):
        """Test that autocast uses device and dtype from wrapper."""
        # Create wrapper with double precision
        tw_double = TensorWrapper(torch.randn(3).double())

        # When autocast is used in method, numpy should be cast to same dtype
        @autocast
        def test_method(wrapper, arr):
            return arr

        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = test_method(tw_double, np_arr)
        self.assertEqual(result.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
