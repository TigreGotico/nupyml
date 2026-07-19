"""Align distributions living in DIFFERENT spaces, via intra-domain distances."""
import numpy as np


def gromov_wasserstein(D1, D2, a=None, b=None, reg=0.05, n_iter=100):
    """Align distributions living in DIFFERENT spaces, via intra-domain distances.

    THE IDEA
    --------
    Ordinary OT needs a cost between a point in X and a point in Y -- impossible if
    X and Y live in different spaces (a 2-D shape and a 3-D shape, two graphs, two
    embeddings with no shared axes). Gromov-Wasserstein compares only the
    WITHIN-domain distance matrices ``D1`` and ``D2``: it seeks a coupling that
    matches pairs (i,j) in domain 1 to pairs (k,l) in domain 2 so that
    ``D1[i,j] ≈ D2[k,l]`` -- preserving the RELATIONAL structure rather than
    absolute positions. That is what lets it match two graphs or two shapes with
    no common coordinate system.

    Solved here by entropic projected-gradient (a Sinkhorn inner loop on the
    GW gradient ``-D1 T D2``). Returns the coupling.

    Mémoli (2011); Peyré et al. (2016).
    """
    from .core import sinkhorn
    n, m = D1.shape[0], D2.shape[0]
    a = np.full(n, 1.0 / n) if a is None else np.asarray(a, float)
    b = np.full(m, 1.0 / m) if b is None else np.asarray(b, float)
    T = np.outer(a, b)
    for _ in range(n_iter):
        # gradient of the GW objective at the current coupling
        grad = -D1 @ T @ D2.T
        grad = grad - grad.min()                      # keep the cost non-negative
        T = sinkhorn(a, b, grad, reg=reg, n_iter=50)[0]
    return T


__all__ = ["gromov_wasserstein"]
