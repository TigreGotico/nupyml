"""k-nearest-neighbour density estimation.

Kernel density estimation fixes a bandwidth everywhere; k-NN density instead
fixes the COUNT and lets the volume adapt -- large where data is sparse, small
where it is dense -- so it follows varying density better than a fixed kernel.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array


class KNNDensity(BaseEstimator):
    """Density estimate from the distance to the k-th nearest neighbour.

    THE ESTIMATOR
    -------------
    Around a query point, grow a ball until it contains ``k`` training points; its
    radius ``r_k`` measures the local sparsity. The density estimate is::

        p(x) = k / (n * V_d * r_k^d)

    where ``V_d`` is the unit-ball volume in ``d`` dimensions. Where data is dense
    ``r_k`` is small (high density); where sparse, ``r_k`` is large (low density).
    Unlike a fixed-bandwidth KDE it ADAPTS its resolution to the local data, at the
    cost of not integrating to exactly one (a known quirk). ``score_samples``
    returns log-densities.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y=None):
        self.data_ = check_array(X)
        self.d_ = self.data_.shape[1]
        return self

    def _unit_ball_volume(self):
        from math import gamma, pi
        return pi ** (self.d_ / 2) / gamma(self.d_ / 2 + 1)

    def score_samples(self, X):
        check_is_fitted(self, "data_")
        X = check_array(X)
        from scipy.spatial.distance import cdist
        D = cdist(X, self.data_)
        r_k = np.sort(D, axis=1)[:, self.k - 1]        # distance to the k-th neighbour
        n = len(self.data_)
        vol = self._unit_ball_volume() * r_k ** self.d_ + 1e-300
        return np.log(self.k / (n * vol))

    def score(self, X):
        return float(self.score_samples(X).mean())


__all__ = ["KNNDensity"]
