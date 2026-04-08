# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utils/tensor_utils.py — string/tensor conversions and array utilities."""

import numpy as np
import pytest
import torch

from utils.tw.tensor_utils import (
    find_nearest,
    find_nearest2,
    pad_points,
    pad_string,
    string2tensor,
    tensor2string,
    unpad_string,
)


class TestFindNearest:
    def test_exact_match(self):
        arr = np.array([1.0, 3.0, 5.0, 7.0])
        assert find_nearest(arr, 3.0) == 3.0

    def test_between_values(self):
        arr = np.array([1.0, 3.0, 5.0])
        assert find_nearest(arr, 3.8) == 3.0
        assert find_nearest(arr, 4.2) == 5.0

    def test_return_index(self):
        arr = np.array([10, 20, 30])
        idx = find_nearest(arr, 26, return_index=True)
        assert idx == 2  # 30 is closer to 26 than 20

    def test_boundary_low(self):
        arr = np.array([5.0, 10.0, 15.0])
        assert find_nearest(arr, -100.0) == 5.0

    def test_boundary_high(self):
        arr = np.array([5.0, 10.0, 15.0])
        assert find_nearest(arr, 1000.0) == 15.0

    def test_single_element(self):
        arr = np.array([42.0])
        assert find_nearest(arr, 0.0) == 42.0

    def test_list_input(self):
        assert find_nearest([1, 2, 3], 2.1) == 2


class TestFindNearest2:
    def test_exact_match(self):
        arr = [1, 3, 5, 7]
        assert find_nearest2(arr, 5) == 2

    def test_between_values(self):
        arr = [10, 20, 30]
        assert find_nearest2(arr, 24) == 1  # 20 is closer
        assert find_nearest2(arr, 26) == 2  # 30 is closer

    def test_before_first(self):
        arr = [10, 20, 30]
        assert find_nearest2(arr, -5) == 0

    def test_after_last(self):
        arr = [10, 20, 30]
        assert find_nearest2(arr, 100) == 2


class TestPadUnpadString:
    def test_roundtrip(self):
        s = "hello world"
        padded = pad_string(s, max_len=50)
        assert len(padded) == 50
        assert unpad_string(padded) == s

    def test_truncation(self):
        s = "a" * 100
        padded = pad_string(s, max_len=10)
        assert len(padded) == 10
        assert unpad_string(padded) == "a" * 10

    def test_empty_string(self):
        padded = pad_string("", max_len=20)
        assert len(padded) == 20
        assert unpad_string(padded) == ""

    def test_exact_length(self):
        s = "abcde"
        padded = pad_string(s, max_len=5)
        assert unpad_string(padded) == s

    def test_default_max_len(self):
        padded = pad_string("test")
        assert len(padded) == 200


class TestString2Tensor:
    def test_basic(self):
        t = string2tensor("abc")
        assert t.shape == (3,)
        assert t[0].item() == ord("a")
        assert t[1].item() == ord("b")

    def test_roundtrip(self):
        original = "hello world 123"
        t = string2tensor(original)
        result = tensor2string(t)
        assert result == original

    def test_padded_roundtrip(self):
        original = "test label"
        padded = pad_string(original, max_len=128, silent=True)
        t = string2tensor(padded)
        result = tensor2string(t, unpad=True)
        assert result == original

    def test_empty_string(self):
        t = string2tensor("")
        assert t.shape == (0,)

    def test_batch_tensor2string(self):
        strings = ["hello", "world"]
        tensors = torch.stack(
            [string2tensor(pad_string(s, max_len=10, silent=True)) for s in strings]
        )
        results = tensor2string(tensors, unpad=True)
        assert results == strings

    def test_invalid_values(self):
        t = torch.tensor([-1, 0, 65], dtype=torch.uint8)
        # Should not crash — safe_chr handles invalid chars
        result = tensor2string(t)
        assert isinstance(result, str)

    def test_higher_dim_raises(self):
        t = torch.zeros(2, 3, 4)
        with pytest.raises(ValueError, match="Higher dims"):
            tensor2string(t)


class TestPadPoints:
    def test_basic_shape(self):
        pts = torch.randn(100, 3)
        padded = pad_points(pts, max_num_point=200)
        assert padded.shape == (200, 3)

    def test_valid_rows_preserved(self):
        pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        padded = pad_points(pts, max_num_point=10)
        assert torch.allclose(padded[:2], pts)

    def test_nan_padding(self):
        pts = torch.randn(5, 3)
        padded = pad_points(pts, max_num_point=10)
        assert torch.isnan(padded[5:-1]).all()

    def test_last_row_count(self):
        pts = torch.randn(7, 3)
        padded = pad_points(pts, max_num_point=20)
        assert padded[-1, -1].item() == 7

    def test_1d_input(self):
        pts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2 points
        padded = pad_points(pts, max_num_point=10)
        assert padded.shape == (10, 3)
        assert padded[-1, -1].item() == 2

    def test_overflow_clamps(self):
        pts = torch.randn(50, 3)
        padded = pad_points(pts, max_num_point=10)
        assert padded.shape == (10, 3)
        assert padded[-1, -1].item() == 9  # max_num_point - 1
