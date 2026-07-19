"""The Theta method -- robust extrapolation that won the M3 competition"""
import numpy as np
from ..base import BaseEstimator


class Theta(BaseEstimator):
    """The Theta method -- robust extrapolation that won the M3 competition
    (Assimakopoulos & Nikolopoulos, 2000).

    Decompose the series into "theta lines" -- versions with the local curvature
    scaled by ``theta``. ``theta=0`` is the linear regression trend (long-term
    direction); ``theta=2`` doubles the curvature (short-term structure). Forecast
    each and combine (the classic recipe averages the ``theta=0`` line with a
    simple-exponential-smoothing forecast of the ``theta=2`` line). This simple
    blend of long-term trend and short-term smoothing was, astonishingly, the most
    accurate method in the M3 forecasting competition.
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        t = np.arange(n)
        # linear trend (the theta=0 line)
        A = np.column_stack([np.ones(n), t])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        # SES level of the theta=2 line (2*y - trend)
        theta2 = 2 * y - (A @ self.coef_)
        level = theta2[0]
        for v in theta2[1:]:
            level += self.alpha * (v - level)
        self.level_ = level
        self.n_ = n
        return self

    def predict(self, horizon):
        t = np.arange(self.n_, self.n_ + horizon)
        trend = self.coef_[0] + self.coef_[1] * t
        return 0.5 * trend + 0.5 * self.level_         # blend trend + SES level

    forecast = predict


__all__ = ["Theta"]
