"""Tests comparing our scipy replacements against scipy for correctness."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_connected_components

from utils.hungarian import linear_sum_assignment


# --- Hungarian algorithm tests ---


def _assert_same_cost(cost_matrix, row1, col1, row2, col2):
    """Assert two assignments have the same total cost."""
    cost1 = cost_matrix[row1, col1].sum()
    cost2 = cost_matrix[row2, col2].sum()
    np.testing.assert_allclose(cost1, cost2, rtol=1e-8)


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

    @pytest.mark.parametrize("shape", [(3, 7), (7, 3), (10, 20), (20, 10), (1, 5), (5, 1)])
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
        # Gate some pairs
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


# --- Connected components (union-find) tests ---


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


class TestConnectedComponents:
    def _compare(self, n: int, edges: list[tuple[int, int]]):
        """Compare union-find against scipy connected_components."""
        # Build adjacency for scipy
        if edges:
            rows, cols = zip(*edges)
            data = np.ones(len(edges))
            adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            adj = csr_matrix((n, n))

        # scipy
        n_comp, labels = scipy_connected_components(csgraph=adj, directed=False)
        scipy_clusters: list[list[int]] = [[] for _ in range(n_comp)]
        for idx, label in enumerate(labels):
            scipy_clusters[label].append(idx)
        scipy_clusters = [c for c in scipy_clusters if c]

        # ours
        our_clusters = _union_find_components(n, edges)

        assert _normalize_components(our_clusters) == _normalize_components(scipy_clusters)

    def test_empty(self):
        self._compare(0, [])

    def test_single_node(self):
        self._compare(1, [])

    def test_fully_disconnected(self):
        self._compare(5, [])

    def test_fully_connected(self):
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        # Make symmetric
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
        # Random sparse adjacency
        adj = (rng.rand(n, n) > 0.85).astype(int)
        adj = np.maximum(adj, adj.T)  # symmetrize
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
