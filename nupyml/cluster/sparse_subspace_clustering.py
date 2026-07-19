"""Group points that share a low-dimensional SUBSPACE (Elhamifar & Vidal, 2013)."""
import numpy as np
from ..base import BaseEstimator, ClusterMixin, check_is_fitted
from ..utils import check_array, check_random_state


class SparseSubspaceClustering(BaseEstimator, ClusterMixin):
    """Group points that share a low-dimensional SUBSPACE (Elhamifar & Vidal, 2013).

    When data lie on several intersecting planes -- faces under different lighting,
    motions of different objects -- distance-based clustering fails, because a
    point can be near others from a DIFFERENT plane. The self-expressive idea: any
    point in a subspace is a sparse linear combination of OTHER points from the
    SAME subspace. So reconstruct each point from all the others with a sparse
    code, read those codes as an affinity (who reconstructs whom), and spectral-
    cluster that. Points end up connected only to their own subspace. Here the
    sparse code is found greedily (orthogonal matching pursuit).
    """

    def __init__(self, n_clusters=2, n_nonzero=5, random_state=None):
        self.n_clusters = n_clusters
        self.n_nonzero = n_nonzero
        self.random_state = random_state

    def _omp(self, dictionary, target, k):
        # greedily pick the columns that best reconstruct target
        residual = target.copy()
        idx, n = [], dictionary.shape[1]
        for _ in range(min(k, n)):
            corr = np.abs(dictionary.T @ residual)
            j = int(corr.argmax())
            if j not in idx:
                idx.append(j)
            A = dictionary[:, idx]
            coef, *_ = np.linalg.lstsq(A, target, rcond=None)
            residual = target - A @ coef
            if np.linalg.norm(residual) < 1e-9:
                break
        c = np.zeros(n)
        if idx:
            c[idx] = coef
        return c

    def _spectral(self, affinity):
        # normalised-cut embedding: bottom eigenvectors of the symmetric Laplacian
        import scipy.linalg
        from . import KMeans
        d = affinity.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L = np.eye(len(affinity)) - (dinv[:, None] * affinity * dinv[None, :])
        vals, vecs = scipy.linalg.eigh(L)
        emb = vecs[:, :self.n_clusters]               # smallest eigenvectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return KMeans(n_clusters=self.n_clusters,
                      random_state=self.random_state).fit_predict(emb)

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        C = np.zeros((n, n))
        for i in range(n):
            others = np.delete(np.arange(n), i)
            code = self._omp(X[others].T, X[i], self.n_nonzero)  # exclude self
            C[i, others] = code
        affinity = np.abs(C) + np.abs(C.T)            # symmetric self-expression graph
        self.labels_ = self._spectral(affinity)
        self.affinity_matrix_ = affinity
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


__all__ = ["SparseSubspaceClustering"]
