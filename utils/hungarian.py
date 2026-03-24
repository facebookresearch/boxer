"""Pure numpy implementation of the Hungarian (Munkres) algorithm."""

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find minimum cost assignment using the Hungarian algorithm.

    Args:
        cost_matrix: (n, m) cost matrix. Can be rectangular.

    Returns:
        Tuple of (row_indices, col_indices) for the optimal assignment.
    """
    cost = np.array(cost_matrix, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Transpose if more rows than columns (algorithm needs n <= m)
    transposed = False
    if n > m:
        cost = cost.T
        n, m = m, n
        transposed = True

    # Pad to square if needed
    if n < m:
        cost = np.vstack([cost, np.full((m - n, m), 0.0)])

    size = m
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1

            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract assignment
    row_ind = []
    col_ind = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] <= n:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

    row_ind = np.array(row_ind, dtype=int)
    col_ind = np.array(col_ind, dtype=int)

    if transposed:
        row_ind, col_ind = col_ind, row_ind

    # Sort by row index
    order = np.argsort(row_ind)
    return row_ind[order], col_ind[order]
