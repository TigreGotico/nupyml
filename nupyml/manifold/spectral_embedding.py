"""Laplacian eigenmaps: the bottom eigenvectors of the graph Laplacian."""
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class SpectralEmbedding(BaseEstimator):
    """Laplacian eigenmaps: the bottom eigenvectors of the graph Laplacian."""

    def __init__(self, n_components=2, affinity="nearest_neighbors",
                 gamma=None, n_neighbors=10, random_state=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _affinity_matrix(self, X):
        n = len(X)
        if self.affinity == "rbf":
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            return np.exp(-gamma * cdist(X, X, "sqeuclidean"))
        if self.affinity == "nearest_neighbors":
            tree = cKDTree(X)
            k = min(self.n_neighbors + 1, n)
            _, idx = tree.query(X, k=k)
            W = np.zeros((n, n))
            rows = np.repeat(np.arange(n), k - 1)
            W[rows, idx[:, 1:].ravel()] = 1.0
            return np.maximum(W, W.T)   # symmetrize the directed kNN graph
        raise ValueError(f"Unknown affinity: {self.affinity!r}")

    def fit_transform(self, X, y=None):
        X = check_array(X)
        n = len(X)
        W = self._affinity_matrix(X)
        np.fill_diagonal(W, 0.0)
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        vals, vecs = scipy.linalg.eigh(L)
        # drop the trivial constant eigenvector at index 0
        emb = vecs[:, 1:self.n_components + 1]
        emb = emb * d_inv_sqrt[:, None]
        self.embedding_ = emb
        self.affinity_matrix_ = W
        return emb

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


__all__ = ["SpectralEmbedding"]
