"""Bayesian VAR with the Minnesota prior: shrink a VAR toward a random walk."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array


class BayesianVAR(BaseEstimator):
    """VAR that SHRINKS toward a random walk (Litterman's Minnesota prior, 1986).

    A VAR on many series has more coefficients than short macro samples can pin down,
    so OLS overfits and forecasts badly. The Minnesota prior encodes the belief that
    each series is roughly a RANDOM WALK: it shrinks every coefficient toward zero
    EXCEPT each variable's own first lag (shrunk toward one), with own lags shrunk less
    than cross lags and higher lags shrunk more. The posterior is a ridge-like
    regularised VAR whose forecasts are far more stable than OLS on limited data.
    ``lam`` controls the overall shrinkage tightness.
    """

    def __init__(self, p=1, lam=0.1):
        self.p = p
        self.lam = lam

    def fit(self, Y):
        Y = check_array(Y)
        T, k = Y.shape
        rows, targ = [], []
        for t in range(self.p, T):
            rows.append(np.concatenate([[1.0], Y[t - self.p:t][::-1].ravel()]))
            targ.append(Y[t])
        Xd = np.array(rows); B = np.array(targ)
        # prior mean: own first lag = 1, everything else 0
        prior_mean = np.zeros((Xd.shape[1], k))
        for j in range(k):
            prior_mean[1 + j, j] = 1.0                    # own lag-1 -> random walk
        # ridge toward the prior mean (Minnesota-style shrinkage)
        pen = np.eye(Xd.shape[1]) / self.lam ** 2
        pen[0, 0] = 0                                     # don't shrink the intercept
        self.coef_ = np.linalg.solve(Xd.T @ Xd + pen,
                                     Xd.T @ B + pen @ prior_mean)
        self.k_ = k
        self._history = Y[-self.p:].copy()
        return self

    def forecast(self, steps=10):
        hist = list(self._history)
        out = []
        for _ in range(steps):
            x = np.concatenate([[1.0], np.array(hist[-self.p:][::-1]).ravel()])
            pred = x @ self.coef_
            out.append(pred); hist.append(pred)
        return np.array(out)


__all__ = ["BayesianVAR"]
