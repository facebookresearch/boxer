#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from boxernet.boxernet import Attention, AttentionBlockV2, FeedForward


class TestFeedForward(unittest.TestCase):
    def test_output_shape(self):
        ff = FeedForward(dim=64, hidden_dim=256)
        x = torch.randn(2, 10, 64)
        y = ff(x)
        self.assertEqual(y.shape, x.shape)

    def test_backward(self):
        ff = FeedForward(dim=32, hidden_dim=64)
        x = torch.randn(3, 5, 32, requires_grad=True)
        y = ff(x).sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


class TestAttention(unittest.TestCase):
    def test_self_attention_shape(self):
        attn = Attention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 8, 64)
        out = attn(x, x)
        self.assertEqual(out.shape, x.shape)

    def test_cross_attention_shape(self):
        attn = Attention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 8, 64)
        y = torch.randn(2, 12, 64)
        out = attn(x, y)
        self.assertEqual(out.shape, x.shape)

    def test_grad_flow(self):
        attn = Attention(dim=32, heads=4, dim_head=8)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = attn(x, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


class TestAttentionBlockV2(unittest.TestCase):
    def test_output_shape_self_attn(self):
        model = AttentionBlockV2(dim=128, depth=2, heads=4)
        x = torch.randn(4, 16, 128)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_shape_cross_attn(self):
        model = AttentionBlockV2(dim=128, depth=3, heads=8)
        x = torch.randn(2, 10, 128)
        y = torch.randn(2, 20, 128)
        out = model(x, y)
        self.assertEqual(out.shape, x.shape)

    def test_backward(self):
        model = AttentionBlockV2(dim=64, depth=2, heads=4)
        x = torch.randn(2, 6, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_determinism(self):
        torch.manual_seed(42)
        model1 = AttentionBlockV2(dim=64, depth=2, heads=4)
        model2 = AttentionBlockV2(dim=64, depth=2, heads=4)
        model2.load_state_dict(model1.state_dict())

        x = torch.randn(2, 8, 64)
        torch.manual_seed(123)
        out1 = model1(x)
        torch.manual_seed(123)
        out2 = model2(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_query_independence(self):
        """Verify that different queries don't interfere if y is constant."""
        model = AttentionBlockV2(dim=32, depth=1, heads=4)
        x = torch.randn(1, 4, 32)
        y = torch.randn(1, 8, 32)
        out1 = model(x.clone(), y)
        x2 = x.clone()
        x2[:, 1] += 1.0  # perturb one query
        out2 = model(x2, y)
        # other queries should remain unchanged
        diff = (out1 - out2).abs().mean(dim=-1)
        self.assertAlmostEqual(diff[0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(diff[0, 2].item(), 0.0, places=5)
        self.assertAlmostEqual(diff[0, 3].item(), 0.0, places=5)
        self.assertGreater(diff[0, 1].item(), 0.0)

    def test_query_flatten(self):
        torch.manual_seed(0)
        attn = Attention(dim=64, heads=4, dim_head=16)

        B, Nq, Nk, D = 2, 5, 7, 64
        x = torch.randn(B, Nq, D)  # queries
        y = torch.randn(B, Nk, D)  # keys/values

        # --- Case 1: all queries together
        out_joint = attn(x, y)  # [B, Nq, D]

        # --- Case 2: move queries into batch dimension
        # Each query processed separately (simulate independence)
        x_flat = x.reshape(B * Nq, 1, D)
        y_flat = y.repeat_interleave(
            Nq, dim=0
        )  # each query uses its own batch's memory

        out_flat = attn(x_flat, y_flat)  # [B*Nq, 1, D]
        out_flat = out_flat.reshape(B, Nq, D)

        # --- Compare
        self.assertTrue(
            torch.allclose(out_joint, out_flat, atol=1e-6),
            msg="Outputs differ when queries are flattened into the batch dimension.",
        )


if __name__ == "__main__":
    unittest.main()
