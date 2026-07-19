"""Pearson correlation of the degrees at the two ends of an edge."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def degree_assortativity(A):
    """Pearson correlation of the degrees at the two ends of an edge.

    Positive means high-degree nodes tend to link to other high-degree nodes
    (social networks); negative means hubs link to low-degree nodes (the
    technological/biological pattern). One number that captures a network's
    mixing character.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    iu = np.transpose(np.nonzero(np.triu(B, 1)))
    if len(iu) == 0:
        return 0.0
    x = deg[iu[:, 0]]
    y = deg[iu[:, 1]]
    # symmetric correlation: stack both edge orientations
    xs = np.concatenate([x, y])
    ys = np.concatenate([y, x])
    if xs.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


# --- Girvan-Newman community detection ------------------------------------


__all__ = ["degree_assortativity"]
