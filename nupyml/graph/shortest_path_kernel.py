"""Compare two graphs by their distributions of shortest-path LENGTHS."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def shortest_path_kernel(A1, A2):
    """Compare two graphs by their distributions of shortest-path LENGTHS.

    Compute all-pairs shortest-path lengths in each graph; the kernel counts pairs
    of paths (one from each graph) with equal length. Two graphs that route
    information over similar distances score high, regardless of node identity --
    a permutation-invariant similarity. Floyd-Warshall gives the distances.
    """
    def apsp(A):
        B = _binary(A)
        n = B.shape[0]
        D = np.where(B > 0, 1.0, np.inf)
        np.fill_diagonal(D, 0)
        for k in range(n):
            D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
        return D
    d1 = apsp(A1)[np.triu_indices(A1.shape[0], 1)]
    d2 = apsp(A2)[np.triu_indices(A2.shape[0], 1)]
    d1 = d1[np.isfinite(d1)]; d2 = d2[np.isfinite(d2)]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    # count equal-length path pairs = dot product of length histograms
    maxlen = int(max(d1.max(), d2.max()))
    h1 = np.bincount(d1.astype(int), minlength=maxlen + 1)
    h2 = np.bincount(d2.astype(int), minlength=maxlen + 1)
    return float(h1 @ h2)


__all__ = ["shortest_path_kernel"]
