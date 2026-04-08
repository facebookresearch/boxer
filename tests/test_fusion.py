# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fusion utilities: Hungarian algorithm, connected components, and 3D box fusion."""

import math
import unittest

import numpy as np
import pytest
import torch

from utils.fuse_3d_boxes import (
    BoundingBox3DFuser,
    align_boxes_r90,
    angular_distance,
    weighted_yaw_mean,
)
from utils.tw.obb import make_obb
from utils.tw.tensor_utils import pad_string, string2tensor

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as scipy_connected_components

    from utils.fuse_3d_boxes import linear_sum_assignment

    _has_scipy = True
except ImportError:
    _has_scipy = False


# =============================================================================
# Helpers
# =============================================================================


def _assert_same_cost(cost_matrix, row1, col1, row2, col2):
    """Assert two assignments have the same total cost."""
    cost1 = cost_matrix[row1, col1].sum()
    cost2 = cost_matrix[row2, col2].sum()
    np.testing.assert_allclose(cost1, cost2, rtol=1e-8)


def _make_test_obb(position, sz=(1.0, 1.0, 1.0), yaw=0.0, prob=0.9, text="test_object"):
    """Create an ObbTW with text label set for fusion testing."""
    obb = make_obb(sz=list(sz), position=list(position), prob=prob, yaw=yaw)
    text_tensor = string2tensor(pad_string(text, max_len=128))
    obb.set_text(text_tensor)
    return obb


def _extract_yaw(obb):
    """Extract yaw angle in radians from an ObbTW."""
    R = obb.T_world_object.R.cpu().numpy()
    return float(np.arctan2(R[1, 0], R[0, 0]))


# =============================================================================
# Hungarian algorithm tests
# =============================================================================


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestHungarian:
    def test_empty(self):
        cost = np.zeros((0, 0))
        row, col = linear_sum_assignment(cost)
        assert len(row) == 0
        assert len(col) == 0

    def test_1x1(self):
        cost = np.array([[5.0]])
        row, col = linear_sum_assignment(cost)
        assert list(row) == [0]
        assert list(col) == [0]

    def test_2x2(self):
        cost = np.array([[4, 1], [3, 2]], dtype=float)
        row, col = linear_sum_assignment(cost)
        r_sp, c_sp = scipy_linear_sum_assignment(cost)
        _assert_same_cost(cost, row, col, r_sp, c_sp)

    @pytest.mark.parametrize("size", [3, 5, 10, 20, 50])
    def test_square_random(self, size):
        rng = np.random.RandomState(42 + size)
        cost = rng.rand(size, size) * 100
        row, col = linear_sum_assignment(cost)
        r_sp, c_sp = scipy_linear_sum_assignment(cost)
        _assert_same_cost(cost, row, col, r_sp, c_sp)
        assert len(row) == size

    @pytest.mark.parametrize(
        "shape", [(3, 7), (7, 3), (10, 20), (20, 10), (1, 5), (5, 1)]
    )
    def test_rectangular(self, shape):
        rng = np.random.RandomState(hash(shape) % 2**31)
        cost = rng.rand(*shape) * 100
        row, col = linear_sum_assignment(cost)
        r_sp, c_sp = scipy_linear_sum_assignment(cost)
        _assert_same_cost(cost, row, col, r_sp, c_sp)
        assert len(row) == min(shape)

    def test_all_zeros(self):
        cost = np.zeros((5, 5))
        row, col = linear_sum_assignment(cost)
        assert len(row) == 5
        assert cost[row, col].sum() == 0.0

    def test_large_gated_costs(self):
        """Mimics tracker's invalid pair masking with 1e6 costs."""
        rng = np.random.RandomState(99)
        cost = rng.rand(10, 10) * 0.5
        cost[0, :] = 1e6
        cost[:, 0] = 1e6
        cost[3, 5] = 1e6
        row, col = linear_sum_assignment(cost)
        r_sp, c_sp = scipy_linear_sum_assignment(cost)
        _assert_same_cost(cost, row, col, r_sp, c_sp)

    def test_integer_costs(self):
        cost = np.array([[10, 5, 13], [3, 7, 11], [6, 9, 2]], dtype=float)
        row, col = linear_sum_assignment(cost)
        r_sp, c_sp = scipy_linear_sum_assignment(cost)
        _assert_same_cost(cost, row, col, r_sp, c_sp)


# =============================================================================
# Connected components (union-find) tests
# =============================================================================


def _union_find_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Run union-find (same logic as in fuse_3d_boxes.py)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    clusters_dict: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_dict.setdefault(root, []).append(i)
    return list(clusters_dict.values())


def _normalize_components(clusters: list[list[int]]) -> list[frozenset[int]]:
    """Convert to a set of frozensets for comparison (label-permutation invariant)."""
    return sorted(frozenset(c) for c in clusters)


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestConnectedComponents:
    def _compare(self, n: int, edges: list[tuple[int, int]]):
        """Compare union-find against scipy connected_components."""
        if edges:
            rows, cols = zip(*edges)
            data = np.ones(len(edges))
            adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            adj = csr_matrix((n, n))

        n_comp, labels = scipy_connected_components(csgraph=adj, directed=False)
        scipy_clusters: list[list[int]] = [[] for _ in range(n_comp)]
        for idx, label in enumerate(labels):
            scipy_clusters[label].append(idx)
        scipy_clusters = [c for c in scipy_clusters if c]

        our_clusters = _union_find_components(n, edges)
        assert _normalize_components(our_clusters) == _normalize_components(
            scipy_clusters
        )

    def test_empty(self):
        self._compare(0, [])

    def test_single_node(self):
        self._compare(1, [])

    def test_fully_disconnected(self):
        self._compare(5, [])

    def test_fully_connected(self):
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        edges_sym = edges + [(j, i) for i, j in edges]
        self._compare(5, edges_sym)

    def test_two_components(self):
        edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
        self._compare(4, edges)

    def test_chain(self):
        n = 10
        edges = [(i, i + 1) for i in range(n - 1)]
        edges_sym = edges + [(j, i) for i, j in edges]
        self._compare(n, edges_sym)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_sparse(self, seed):
        rng = np.random.RandomState(seed)
        n = 20
        adj = (rng.rand(n, n) > 0.85).astype(int)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        rows, cols = adj.nonzero()
        edges = list(zip(rows.tolist(), cols.tolist()))
        self._compare(n, edges)

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_random_dense(self, seed):
        rng = np.random.RandomState(seed)
        n = 30
        adj = (rng.rand(n, n) > 0.5).astype(int)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        rows, cols = adj.nonzero()
        edges = list(zip(rows.tolist(), cols.tolist()))
        self._compare(n, edges)


# =============================================================================
# Weighted yaw mean tests
# =============================================================================


class TestWeightedYawMean(unittest.TestCase):
    """Tests for weighted_yaw_mean with 180-degree symmetry."""

    def test_identical_angles(self):
        """Two identical angles should return that angle."""
        angles = torch.tensor([math.pi / 4, math.pi / 4])
        weights = torch.tensor([0.5, 0.5])
        mean, resultant = weighted_yaw_mean(angles, weights)
        self.assertAlmostEqual(mean.item(), math.pi / 4, places=5)
        self.assertGreater(resultant.item(), 0.9)

    def test_zero_angles(self):
        """All-zero angles should return zero."""
        angles = torch.tensor([0.0, 0.0, 0.0])
        weights = torch.tensor([1.0, 1.0, 1.0])
        mean, _ = weighted_yaw_mean(angles, weights)
        self.assertAlmostEqual(mean.item(), 0.0, places=5)

    def test_pi_symmetry(self):
        """0 and pi should be equivalent (180-degree symmetry) and resolve to one of them."""
        angles = torch.tensor([0.0, math.pi])
        weights = torch.tensor([0.5, 0.5])
        mean, _ = weighted_yaw_mean(angles, weights)
        # Should be close to 0 or pi, not pi/2
        diff_0 = abs(mean.item())
        diff_pi = abs(mean.item() - math.pi)
        min_diff = min(diff_0, diff_pi)
        self.assertLess(min_diff, math.radians(5.0))

    def test_weighted_bias(self):
        """Heavier weight should pull mean toward that angle."""
        angles = torch.tensor([0.0, math.pi / 4])
        weights = torch.tensor([0.9, 0.1])
        mean, _ = weighted_yaw_mean(angles, weights)
        # Should be closer to 0 than to pi/4
        self.assertLess(abs(mean.item()), math.pi / 8)


# =============================================================================
# Angular distance tests
# =============================================================================


class TestAngularDistance(unittest.TestCase):
    """Tests for angular_distance with 180-degree symmetry."""

    def test_same_angle(self):
        self.assertAlmostEqual(angular_distance(0.0, 0.0), 0.0)

    def test_pi_apart(self):
        """0 and pi should have zero distance (180-degree symmetry)."""
        self.assertAlmostEqual(angular_distance(0.0, math.pi), 0.0, places=5)

    def test_90_degrees(self):
        self.assertAlmostEqual(
            angular_distance(0.0, math.pi / 2), math.pi / 2, places=5
        )

    def test_45_degrees(self):
        self.assertAlmostEqual(
            angular_distance(0.0, math.pi / 4), math.pi / 4, places=5
        )

    def test_symmetric(self):
        self.assertAlmostEqual(
            angular_distance(0.3, 1.1),
            angular_distance(1.1, 0.3),
        )


# =============================================================================
# Rotation alignment tests (align_boxes_r90)
# =============================================================================


class TestAlignBoxesR90(unittest.TestCase):
    """Tests for 90-degree rotation alignment of boxes."""

    def test_no_swap_needed(self):
        """Boxes with same orientation should not be swapped."""
        sizes = torch.tensor([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        yaws = torch.tensor([0.0, 0.0])
        weights = torch.tensor([0.5, 0.5])
        aligned_sizes, aligned_yaws = align_boxes_r90(sizes, yaws, weights)
        self.assertTrue(torch.allclose(aligned_sizes, sizes))
        self.assertTrue(torch.allclose(aligned_yaws, yaws))

    def test_90_degree_swap(self):
        """A box rotated 90 degrees with swapped dims should align to the reference."""
        sizes = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        yaws = torch.tensor([0.0, math.pi / 2])
        weights = torch.tensor([0.5, 0.5])
        aligned_sizes, aligned_yaws = align_boxes_r90(sizes, yaws, weights)
        # After alignment, both should have similar width/height
        self.assertAlmostEqual(
            aligned_sizes[0, 0].item(), aligned_sizes[1, 0].item(), places=3
        )
        self.assertAlmostEqual(
            aligned_sizes[0, 1].item(), aligned_sizes[1, 1].item(), places=3
        )


# =============================================================================
# BoundingBox3DFuser integration tests
# =============================================================================


class TestBoundingBox3DFuser(unittest.TestCase):
    """Integration tests for 3D bounding box fusion."""

    def test_two_identical_boxes_fuse(self):
        """Two identical overlapping boxes should fuse into one instance."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], prob=0.9),
            _make_test_obb([0.0, 0.0, 0.5], prob=0.9),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].support_count, 2)

    def test_non_overlapping_boxes_stay_separate(self):
        """Well-separated boxes should not fuse."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], text="box_a"),
            _make_test_obb([10.0, 0.0, 0.5], text="box_b"),
            _make_test_obb([20.0, 0.0, 0.5], text="box_c"),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 3)

    def test_min_detections_filters(self):
        """Boxes below min_detections threshold should be filtered out."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=2,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        # 3 non-overlapping single boxes — each cluster has support_count=1
        obbs = [_make_test_obb([i * 5.0, 0.0, 0.5]) for i in range(3)]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 0)

    def test_min_detections_1_preserves_all(self):
        """With min_detections=1, all non-overlapping boxes should survive."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        num_boxes = 10
        obbs = [
            _make_test_obb([i * 3.0, 0.0, 0.5], text=f"box_{i}")
            for i in range(num_boxes)
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), num_boxes)
        for inst in instances:
            self.assertEqual(inst.support_count, 1)

    def test_overlapping_pairs_fuse(self):
        """Pairs of overlapping boxes should each fuse into one instance."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        num_pairs = 5
        obbs = []
        for i in range(num_pairs):
            obbs.append(_make_test_obb([i * 5.0, 0.0, 0.5], prob=0.9, text=f"pair_{i}"))
            obbs.append(
                _make_test_obb([i * 5.0, 0.0, 0.5], prob=0.85, text=f"pair_{i}")
            )
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), num_pairs)
        for inst in instances:
            self.assertEqual(inst.support_count, 2)

    def test_same_rotation_preserved(self):
        """Fusing boxes with identical yaw should preserve that yaw."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        yaw = math.pi / 4
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], yaw=yaw),
            _make_test_obb([0.0, 0.0, 0.5], yaw=yaw),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 1)
        fused_yaw = _extract_yaw(instances[0].obb)
        self.assertAlmostEqual(fused_yaw, yaw, delta=math.radians(5.0))

    def test_180_degree_symmetry(self):
        """Boxes at 0 and pi radians should fuse to ~0 or ~pi, not pi/2."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], yaw=0.0),
            _make_test_obb([0.0, 0.0, 0.5], yaw=math.pi),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 1)
        fused_yaw = _extract_yaw(instances[0].obb)
        min_diff = min(abs(fused_yaw), abs(fused_yaw - math.pi))
        self.assertLess(
            min_diff,
            math.radians(15.0),
            f"Fused yaw {fused_yaw:.4f} should be near 0 or pi, not pi/2",
        )

    def test_three_boxes_majority_rotation(self):
        """Three boxes (0, pi, 0) should fuse to ~0 (majority wins)."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], yaw=0.0),
            _make_test_obb([0.0, 0.0, 0.5], yaw=math.pi),
            _make_test_obb([0.0, 0.0, 0.5], yaw=0.0),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 1)
        fused_yaw = _extract_yaw(instances[0].obb)
        min_diff = min(abs(fused_yaw), abs(fused_yaw - math.pi))
        self.assertLess(min_diff, math.radians(15.0))

    def test_fused_position_is_weighted_average(self):
        """Fused position should be the weighted average of inputs."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            confidence_weighting="uniform",
            conf_threshold=0.0,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], prob=0.9),
            _make_test_obb([0.0, 0.0, 0.5], prob=0.9),
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 1)
        fused_pos = instances[0].obb.T_world_object.t.cpu()
        expected = torch.tensor([0.0, 0.0, 0.5])
        self.assertTrue(torch.allclose(fused_pos, expected, atol=0.1))

    def test_empty_detections(self):
        """Empty input should return empty list."""
        fuser = BoundingBox3DFuser(min_detections=1, conf_threshold=0.0)
        obb = _make_test_obb([0.0, 0.0, 0.5])
        # Create empty detections with correct shape
        empty = torch.stack([obb])[:0]
        instances = fuser.fuse(empty)
        self.assertEqual(len(instances), 0)

    def test_conf_threshold_filters(self):
        """Detections below conf_threshold should be filtered before fusion."""
        fuser = BoundingBox3DFuser(
            iou_threshold=0.3,
            min_detections=1,
            conf_threshold=0.8,
        )
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], prob=0.5),  # below threshold
            _make_test_obb([0.0, 0.0, 0.5], prob=0.5),  # below threshold
        ]
        detections = torch.stack(obbs)
        instances = fuser.fuse(detections)
        self.assertEqual(len(instances), 0)


# =============================================================================
# Rotation matrix / yaw round-trip tests
# =============================================================================


class TestYawRoundTrip(unittest.TestCase):
    """Verify yaw -> rotation matrix -> yaw round-trips correctly."""

    @pytest.mark.parametrize(
        "yaw", [0.0, math.pi / 4, math.pi / 2, -math.pi / 4, -math.pi / 2]
    )
    def test_yaw_roundtrip(self, yaw=None):
        """Create OBB with known yaw and verify it's recoverable."""
        for yaw_val in [0.0, math.pi / 4, math.pi / 2, -math.pi / 4, -math.pi / 2]:
            obb = _make_test_obb([0.0, 0.0, 0.5], yaw=yaw_val)
            extracted = _extract_yaw(obb)
            diff = abs((extracted - yaw_val + math.pi) % (2 * math.pi) - math.pi)
            self.assertLess(
                diff,
                math.radians(1.0),
                f"Extracted yaw {extracted:.4f} should match input {yaw_val:.4f}",
            )
