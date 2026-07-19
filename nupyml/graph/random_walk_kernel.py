"""Count common walks in the two graphs via their DIRECT (tensor) product."""
import numpy as np
from .line import LINE


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def random_walk_kernel(A1, A2, decay=0.1, max_len=10):
    """Count common walks in the two graphs via their DIRECT (tensor) product.

    A walk shared by both graphs is a walk in their product graph; summing over
    all lengths ``l`` with a ``decay^l`` weight gives the geometric-series kernel
    ``sum_l decay^l * 1' (A_x)^l 1`` on the product adjacency ``A_x = A1 ⊗ A2``.
    More walks in common -> more similar structure. Truncated at ``max_len``.
    """
    B1, B2 = _binary(A1), _binary(A2)
    Ax = np.kron(B1, B2)                             # product-graph adjacency
    n = Ax.shape[0]
    total = 0.0
    power = np.eye(n)
    for l in range(1, max_len + 1):
        power = power @ Ax
        total += (decay ** l) * power.sum()
    return float(total)


# --- LINE embedding -------------------------------------------------------


__all__ = ["random_walk_kernel"]
