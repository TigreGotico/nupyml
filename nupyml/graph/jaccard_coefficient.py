"""Common neighbours NORMALISED by the size of the combined neighbourhood."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def jaccard_coefficient(A):
    """Common neighbours NORMALISED by the size of the combined neighbourhood.

    ``|N(i) ∩ N(j)| / |N(i) ∪ N(j)|`` -- so two low-degree nodes with a couple of
    shared neighbours can outrank two hubs that share many merely because they
    have many. The right correction when degrees vary a lot.
    """
    B = _binary(A)
    inter = B @ B
    deg = B.sum(axis=1)
    union = deg[:, None] + deg[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(S, 0)
    return S


__all__ = ["jaccard_coefficient"]
