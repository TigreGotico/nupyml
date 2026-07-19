"""Complex and MULTIPLE seasonality via trig terms + Box-Cox (De Livera, 2011)."""
import numpy as np
from ..base import BaseEstimator


class TBATS(BaseEstimator):
    """Complex and MULTIPLE seasonality via trig terms + Box-Cox (De Livera, 2011).

    Holt-Winters needs one integer seasonal period. Real series can have several,
    possibly non-integer, cycles at once (daily + weekly, or a 365.25-day year).
    TBATS represents each season by a few TRIGONOMETRIC (Fourier) terms instead of a
    per-period state, so it handles high-frequency and multiple seasonalities
    cheaply, and a BOX-COX transform first stabilises a variance that grows with the
    level. Here: Box-Cox, then least-squares fit of a linear trend plus Fourier
    seasonal terms for each period, with the fit inverted for forecasting.
    """

    def __init__(self, periods=(12,), n_harmonics=3, box_cox_lambda=None):
        self.periods = list(periods)
        self.n_harmonics = n_harmonics
        self.box_cox_lambda = box_cox_lambda

    def _boxcox(self, y):
        lam = self.box_cox_lambda
        if lam is None or lam == 0:
            return np.log(y)
        return (y ** lam - 1) / lam

    def _inv_boxcox(self, z):
        lam = self.box_cox_lambda
        if lam is None or lam == 0:
            return np.exp(z)
        return (lam * z + 1) ** (1 / lam)

    def _design(self, t):
        cols = [np.ones_like(t), t]
        for p in self.periods:
            for k in range(1, self.n_harmonics + 1):
                cols.append(np.sin(2 * np.pi * k * t / p))
                cols.append(np.cos(2 * np.pi * k * t / p))
        return np.column_stack(cols)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.n_ = len(y)
        self._positive = np.all(y > 0)
        z = self._boxcox(y) if self._positive else y
        t = np.arange(self.n_, dtype=float)
        self.coef_, *_ = np.linalg.lstsq(self._design(t), z, rcond=None)
        return self

    def forecast(self, steps=10):
        t = np.arange(self.n_, self.n_ + steps, dtype=float)
        z = self._design(t) @ self.coef_
        return self._inv_boxcox(z) if self._positive else z


__all__ = ["TBATS"]
