"""Sum over ALL paths between i and j, damped by length: (I - beta A)^-1 - I."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def katz_index(A, beta=0.01):
    """Sum over ALL paths between i and j, damped by length: (I - beta A)^-1 - I.

    Common neighbours only see length-2 paths; the Katz index counts paths of
    every length, weighting a length-``l`` path by ``beta^l`` so short paths
    dominate. ``beta`` must be below ``1/spectral-radius`` for the series to
    converge. It captures connectivity that shared-neighbour scores miss on sparse
    graphs, at the cost of a matrix inverse.
    """
    B = _binary(A)
    n = B.shape[0]
    S = np.linalg.inv(np.eye(n) - beta * B) - np.eye(n)
    np.fill_diagonal(S, 0)
    return S


# --- assortativity --------------------------------------------------------


__all__ = ["katz_index"]
