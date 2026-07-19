"""Beta calibration: a flexible 2-parameter map for binary probabilities"""
import numpy as np
from ..base import BaseEstimator


class BetaCalibration(BaseEstimator):
    """Beta calibration: a flexible 2-parameter map for binary probabilities
    (Kull et al., 2017).

    Platt scaling (a logistic map) is symmetric and often too rigid; isotonic is
    flexible but can overfit and is non-smooth. Beta calibration fits a logistic
    regression on ``log(p)`` and ``log(1-p)``::

        calibrated = sigmoid(a*log(p) - b*log(1-p) + c)

    which is the family of maps derived from Beta likelihoods -- smooth, and able
    to represent the asymmetric S-curves real classifiers produce, with only three
    parameters. A strong default between Platt and isotonic.
    """

    def fit(self, scores, y):
        from ..linear_model import LogisticRegression
        p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
        feats = np.column_stack([np.log(p), -np.log(1 - p)])
        self.lr_ = LogisticRegression(max_iter=500).fit(feats, np.asarray(y))
        return self

    def transform(self, scores):
        p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
        feats = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.lr_.predict_proba(feats)[:, 1]


__all__ = ["BetaCalibration"]
