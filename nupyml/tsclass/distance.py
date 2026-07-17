"""Elastic-distance nearest-neighbour classification -- the TSC gold standard.

For years, 1-NN with dynamic time warping was the method to beat in time-series
classification, and it is still the honest baseline every new method is measured
against. DTW aligns two series by stretching the time axis, so a pattern that
occurs early in one series and late in another still matches -- the invariance a
Euclidean distance completely lacks.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..utils import check_array, check_X_y
from ..preprocessing import LabelEncoder


class KNeighborsTimeSeriesClassifier(BaseEstimator, ClassifierMixin):
    """k-NN over whole series under an elastic distance (DTW by default).

    ``fit`` just stores the training series (this is a lazy learner); ``predict``
    scores a test series against every training series under the chosen distance
    and takes the majority label of the ``k`` closest. The Sakoe-Chiba
    ``window`` bounds how far the warping path may stray from the diagonal --
    which both speeds DTW up and stops pathological over-warping.
    """

    def __init__(self, n_neighbors=1, metric="dtw", window=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.window = window

    def _distance(self, a, b):
        from ..sequence import dtw_distance
        if self.metric == "dtw":
            return dtw_distance(a, b, window=self.window)
        if self.metric == "euclidean":
            return float(np.sqrt(np.sum((a - b) ** 2)))
        raise ValueError(f"unknown metric {self.metric!r}")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.X_ = X
        self.y_ = self._le.transform(y)
        return self

    def predict(self, X):
        check_is_fitted(self, "X_")
        X = check_array(X)
        preds = np.empty(len(X), dtype=int)
        k = min(self.n_neighbors, len(self.X_))
        for i, xt in enumerate(X):
            dists = np.array([self._distance(xt, xr) for xr in self.X_])
            nn = np.argsort(dists)[:k]
            preds[i] = np.bincount(self.y_[nn]).argmax()
        return self.classes_[preds]


__all__ = ["KNeighborsTimeSeriesClassifier"]
