"""Ensemble a base detector over random FEATURE subsets (Lazarevic, 2005)."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array, check_random_state
from .detectors import _Detector


class FeatureBaggingDetector(_Detector):
    """Ensemble a base detector over random FEATURE subsets (Lazarevic, 2005).

    In high dimensions the anomaly signal often lives in a few features and is
    drowned out by the rest (the curse of dimensionality for distance-based
    detectors). Feature bagging fits the base detector on many random feature
    SUBSPACES and averages the (rank-normalised) scores, so a subspace where the
    anomaly stands out can carry the vote even when the full-space detector is
    blind to it. The subspace analogue of bagging.
    """

    def __init__(self, base_estimator=None, n_estimators=10,
                 max_features=0.5, contamination=0.1, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        from .detectors import KNN
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        k = max(1, int(self.max_features * d))
        base = self.base_estimator if self.base_estimator is not None else KNN()
        self.subsets_, self.detectors_ = [], []
        for _ in range(self.n_estimators):
            cols = rng.choice(d, k, replace=False)
            det = clone(base).fit(X[:, cols])
            self.subsets_.append(cols)
            self.detectors_.append(det)
        self._set_threshold(self.decision_function(X))
        return self

    @staticmethod
    def _rank(s):
        order = s.argsort().argsort()                # 0..n-1 ranks
        return order / (len(s) - 1 + 1e-12)

    def decision_function(self, X):
        X = check_array(X)
        total = np.zeros(len(X))
        for cols, det in zip(self.subsets_, self.detectors_):
            total += self._rank(det.decision_function(X[:, cols]))
        return total / self.n_estimators


__all__ = ["FeatureBaggingDetector"]
