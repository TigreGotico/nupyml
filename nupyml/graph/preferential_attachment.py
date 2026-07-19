"""Score (i,j) by ``deg(i) * deg(j)`` -- the rich get richer."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def preferential_attachment(A):
    """Score (i,j) by ``deg(i) * deg(j)`` -- the rich get richer.

    No shared-neighbour information at all: it bets that high-degree nodes keep
    attracting edges. Weak alone, but it captures the growth dynamics of many real
    networks and needs nothing but degrees.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    S = np.outer(deg, deg).astype(float)
    np.fill_diagonal(S, 0)
    return S


__all__ = ["preferential_attachment"]
