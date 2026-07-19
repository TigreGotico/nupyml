"""Score a missing edge (i,j) by how many neighbours i and j share."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def common_neighbors(A):
    """Score a missing edge (i,j) by how many neighbours i and j share.

    The simplest link-prediction heuristic and the base of the others: two people
    with many mutual friends are likely to become friends. For a 0/1 adjacency,
    the count of common neighbours is exactly ``(A @ A)[i, j]``.
    """
    B = _binary(A)
    S = B @ B
    np.fill_diagonal(S, 0)
    return S


__all__ = ["common_neighbors"]
