"""A trend that BENDS at changepoints, plus Fourier seasonality (Taylor, 2018)."""
import numpy as np
from ..base import BaseEstimator, RegressorMixin


class ProphetForecaster(BaseEstimator):
    """A trend that BENDS at changepoints, plus Fourier seasonality (Taylor, 2018).

    Business time series have a trend whose SLOPE changes (a launch, a policy) and
    smooth seasonal cycles. Prophet models exactly that as a regression:
    a piecewise-LINEAR trend with automatic CHANGEPOINTS where the slope may shift,
    plus Fourier terms for seasonality -- all fit at once by least squares. Being a
    regression (not autoregression) it handles gaps and irregular sampling and
    gives interpretable trend/seasonal components. ``seasonality`` is the period,
    ``n_changepoints`` how many candidate slope changes.
    """

    def __init__(self, n_changepoints=10, seasonality=0, fourier_order=3):
        self.n_changepoints = n_changepoints
        self.seasonality = seasonality
        self.fourier_order = fourier_order

    def _design(self, t):
        cols = [np.ones_like(t), t]                     # intercept + linear trend
        for cp in self.changepoints_:                   # slope changes (hinges)
            cols.append(np.clip(t - cp, 0, None))
        if self.seasonality:
            for k in range(1, self.fourier_order + 1):
                cols.append(np.sin(2 * np.pi * k * t / self.seasonality))
                cols.append(np.cos(2 * np.pi * k * t / self.seasonality))
        return np.column_stack(cols)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.n_ = len(y)
        t = np.arange(self.n_, dtype=float)
        self.changepoints_ = np.linspace(0, self.n_, self.n_changepoints + 2)[1:-1]
        X = self._design(t)
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def forecast(self, steps=10):
        t = np.arange(self.n_, self.n_ + steps, dtype=float)
        return self._design(t) @ self.coef_

    def predict(self, t):
        return self._design(np.asarray(t, float)) @ self.coef_


__all__ = ["ProphetForecaster"]
