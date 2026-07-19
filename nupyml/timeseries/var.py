"""Vector autoregression: several series that predict EACH OTHER (Sims, 1980)."""
import numpy as np
from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array


class VAR(BaseEstimator):
    """Vector autoregression: several series that predict EACH OTHER (Sims, 1980).

    A univariate AR explains a series by its own past. But interest rates, GDP and
    inflation move TOGETHER -- each one's future depends on the recent past of ALL
    of them. VAR(p) stacks that: every variable is regressed on ``p`` lags of every
    variable, a single multivariate OLS. It is the workhorse of macro-econometrics
    and the basis of Granger causality (does adding X's lags help predict Y?).
    Fits ``Y: (T, k)`` and forecasts all ``k`` series jointly.
    """

    def __init__(self, p=1):
        self.p = p

    def fit(self, Y):
        Y = check_array(Y)
        T, k = Y.shape
        rows, targets = [], []
        for t in range(self.p, T):
            lags = Y[t - self.p:t][::-1].ravel()       # p lags of all k series
            rows.append(np.concatenate([[1.0], lags]))
            targets.append(Y[t])
        X = np.array(rows); Z = np.array(targets)
        self.coef_, *_ = np.linalg.lstsq(X, Z, rcond=None)   # (1+p*k, k)
        self.k_ = k
        self._history = Y[-self.p:].copy()
        return self

    def forecast(self, steps=10):
        hist = list(self._history)
        out = []
        for _ in range(steps):
            lags = np.array(hist[-self.p:][::-1]).ravel()
            x = np.concatenate([[1.0], lags])
            pred = x @ self.coef_
            out.append(pred); hist.append(pred)
        return np.array(out)


__all__ = ["VAR"]
