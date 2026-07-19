"""Adamic-Adar's sibling with a 1/degree weight (a heavier penalty on hubs)."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def resource_allocation_index(A):
    """Adamic-Adar's sibling with a 1/degree weight (a heavier penalty on hubs)."""
    B = _binary(A)
    deg = B.sum(axis=1)
    with np.errstate(divide="ignore"):
        w = np.where(deg > 0, 1.0 / deg, 0.0)
    S = (B * w[None, :]) @ B.T
    np.fill_diagonal(S, 0)
    return S


__all__ = ["resource_allocation_index"]
