"""Random projection: reduce dimensions by multiplying with random noise.

This sounds like it cannot work, and the Johnson-Lindenstrauss lemma says it
does: any n points in any number of dimensions can be projected onto
``O(log n / eps^2)` random dimensions with every pairwise distance preserved to
within ``1 +/- eps``.

Read that bound again -- the target dimension depends on the NUMBER OF POINTS,
and not at all on how many dimensions they started in. A million features
project as easily as a thousand.

WHY IT WORKS
------------
In high dimensions, random vectors are almost surely nearly orthogonal. A random
matrix is therefore close to a rotation onto a random subspace, and a rotation
preserves distances. Projecting onto a random subspace loses each point's
component perpendicular to it -- but that loss concentrates tightly around its
expectation, so every distance shrinks by nearly the same factor, and relative
distances survive.

WHAT IT COSTS AND BUYS
----------------------
PCA finds the BEST subspace, but must look at the data to do it -- O(n*d^2). A
random projection ignores the data entirely, so it is essentially free, needs no
fitting pass, and works in a streaming setting. It gives up optimality for cost.

``SparseRandomProjection`` goes further: most entries are exactly zero, so the
projection is a sparse matmul, and the JL guarantee still holds.

Useful as a preprocessing step for anything distance-based -- kNN, clustering --
where distances are all that matter and computing them in the original dimension
is the bottleneck.
"""
import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """Minimum dimension preserving pairwise distances within (1 +/- eps)."""
    n_samples = np.asarray(n_samples, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)


class _BaseRandomProjection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components="auto", eps=0.1, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        n, d = X.shape
        if self.n_components == "auto":
            k = int(johnson_lindenstrauss_min_dim(n, self.eps))
            if k <= 0 or k > d:
                k = min(d, max(1, k))
        else:
            k = int(self.n_components)
        self.n_components_ = k
        rng = check_random_state(self.random_state)
        self.components_ = self._make_matrix(d, k, rng)
        self.n_features_in_ = d
        self.n_features_out_ = k
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, accept_sparse=True)
        return np.asarray(X @ self.components_)


class GaussianRandomProjection(_BaseRandomProjection):
    @staticmethod
    def _make_matrix(d, k, rng):
        return rng.normal(0.0, 1.0 / np.sqrt(k), size=(d, k))


class SparseRandomProjection(_BaseRandomProjection):
    """Achlioptas/Li sparse projection: entries are 0 with probability 1-density."""

    def __init__(self, n_components="auto", density="auto", eps=0.1,
                 dense_output=True, random_state=None):
        super().__init__(n_components, eps, random_state)
        self.density = density
        self.dense_output = dense_output

    def _make_matrix(self, d, k, rng):
        density = 1.0 / np.sqrt(d) if self.density == "auto" else self.density
        self.density_ = density
        scale = np.sqrt(1.0 / (density * k))
        draw = rng.uniform(size=(d, k))
        M = np.zeros((d, k))
        M[draw < density / 2] = -scale
        M[draw > 1 - density / 2] = scale
        return M if self.dense_output else sp.csr_matrix(M)


__all__ = ["GaussianRandomProjection", "SparseRandomProjection",
           "johnson_lindenstrauss_min_dim"]
