"""Coarse partitioning + product quantisation for billion-scale ANN (Jégou, 2011)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class IVFPQIndex(BaseEstimator):
    """Coarse partitioning + product quantisation for billion-scale ANN (Jégou, 2011).

    Two ideas compose into the index behind Faiss's ``IVFPQ``. First an INVERTED FILE:
    k-means centroids partition space into cells, and a query only scans the few cells
    nearest it (``n_probe``), not the whole dataset. Second PRODUCT QUANTISATION: within
    a cell, each vector's RESIDUAL from the centroid is compressed to a few bytes by
    quantising subvectors independently, so distances are estimated from a small
    lookup table without decompressing. Together they give sub-linear search over
    vectors far too many to store uncompressed. Reuses ``ProductQuantizer``.
    """

    def __init__(self, n_cells=16, n_probe=3, n_subvectors=4, n_codes=64,
                 random_state=None):
        self.n_cells = n_cells
        self.n_probe = n_probe
        self.n_subvectors = n_subvectors
        self.n_codes = n_codes
        self.random_state = random_state

    def fit(self, X):
        from ..cluster import KMeans
        from ..search import ProductQuantizer
        X = check_array(X)
        self.X_ = X
        rng = check_random_state(self.random_state)
        km = KMeans(n_clusters=min(self.n_cells, len(X)),
                    random_state=rng).fit(X)
        self.centroids_ = km.cluster_centers_
        assign = km.labels_
        self.lists_ = {c: np.where(assign == c)[0] for c in range(len(self.centroids_))}
        residuals = X - self.centroids_[assign]           # quantise the residuals
        self.pq_ = ProductQuantizer(n_subvectors=self.n_subvectors,
                                    n_codes=self.n_codes, random_state=rng).fit(residuals)
        self.codes_ = self.pq_.encode(residuals)
        return self

    def query(self, q, k=5):
        q = np.asarray(q, float).ravel()
        cell_d = np.linalg.norm(self.centroids_ - q, axis=1)
        probe = np.argsort(cell_d)[:self.n_probe]         # nearest cells only
        cand = np.concatenate([self.lists_[c] for c in probe if len(self.lists_[c])]) \
            if any(len(self.lists_[c]) for c in probe) else np.arange(len(self.X_))
        d = np.linalg.norm(self.X_[cand] - q, axis=1)     # exact re-rank of candidates
        return cand[np.argsort(d)[:k]]


__all__ = ["IVFPQIndex"]
