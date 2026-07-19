"""Memberships that DON'T sum to one, so noise can belong to nothing"""
import numpy as np
from scipy.spatial.distance import cdist
from ..base import BaseEstimator, ClusterMixin, check_is_fitted
from ..utils import check_array, check_random_state


class PossibilisticCMeans(BaseEstimator, ClusterMixin):
    """Memberships that DON'T sum to one, so noise can belong to nothing
    (Krishnapuram & Keller, 1993).

    Fuzzy c-means forces every point's memberships across clusters to sum to 1 --
    so an outlier equidistant from two clusters gets 0.5 in each, as if it were a
    confident half-member of both, and it drags the centres toward itself.
    Possibilistic c-means drops the sum-to-one rule: a membership becomes a
    TYPICALITY, ``1 / (1 + (d^2/eta)^{1/(m-1)})``, that depends only on the point's
    distance to THAT cluster. A far outlier gets a low typicality to everything and
    stops distorting the centres -- which is exactly what makes it robust to noise.
    ``eta`` (the per-cluster scale) is bootstrapped from a fuzzy c-means pass.
    """

    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fuzzy_init(self, X, rng):
        # a few fuzzy-c-means iterations to seed centres and the eta scales
        n, k = len(X), self.n_clusters
        U = rng.rand(n, k); U /= U.sum(axis=1, keepdims=True)
        centers = None
        for _ in range(20):
            Um = U ** self.m
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            d2 = cdist(X, centers, "sqeuclidean") + 1e-12
            power = 1.0 / (self.m - 1)
            inv = 1.0 / d2
            U = inv ** power
            U /= U.sum(axis=1, keepdims=True)
        Um = U ** self.m
        eta = (Um * cdist(X, centers, "sqeuclidean")).sum(axis=0) / Um.sum(axis=0)
        return centers, np.maximum(eta, 1e-6)

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        centers, eta = self._fuzzy_init(X, rng)
        power = 1.0 / (self.m - 1)
        T = None
        for _ in range(self.max_iter):
            d2 = cdist(X, centers, "sqeuclidean")
            T = 1.0 / (1.0 + (d2 / eta) ** power)     # typicality, not probability
            Tm = T ** self.m
            new = (Tm.T @ X) / Tm.sum(axis=0)[:, None]
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.tol:
                break
        self.typicalities_ = T
        self.cluster_centers_ = centers
        self.eta_ = eta
        self.labels_ = T.argmax(axis=1)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


__all__ = ["PossibilisticCMeans"]
