"""Bayesian structural time series: a decomposable state-space forecaster."""
import numpy as np

from ..base import BaseEstimator


class BayesianStructuralTimeSeries(BaseEstimator):
    """Forecast by tracking interpretable COMPONENTS with a Kalman filter
    (Harvey; Scott & Varian).

    ARIMA models a series as one opaque autoregression. A structural time series
    instead writes it as a sum of components that each mean something -- a local
    linear TREND (a level and a slowly-changing slope) plus a periodic SEASONAL --
    and tracks them with a Kalman filter over the state ``[level, slope,
    seasonal...]``. Because the model is a proper state space, forecasts come with
    UNCERTAINTY that widens with the horizon, and you can read off the estimated
    trend and seasonal separately. Noise variances are set from the data's scale.
    ``season_length=1`` disables the seasonal component.
    """

    def __init__(self, season_length=1, level_var=None, slope_var=None,
                 seasonal_var=None, obs_var=None):
        self.season_length = season_length
        self.level_var = level_var
        self.slope_var = slope_var
        self.seasonal_var = seasonal_var
        self.obs_var = obs_var

    def _build(self, y):
        m = self.season_length
        scale = np.var(np.diff(y)) + 1e-6
        lv = self.level_var if self.level_var is not None else scale * 0.1
        sv = self.slope_var if self.slope_var is not None else scale * 0.01
        se = self.seasonal_var if self.seasonal_var is not None else scale * 0.1
        ov = self.obs_var if self.obs_var is not None else scale
        # state = [level, slope, s_0, ..., s_{m-2}]
        n_seas = (m - 1) if m > 1 else 0
        dim = 2 + n_seas
        T = np.zeros((dim, dim))
        T[0, 0] = 1; T[0, 1] = 1; T[1, 1] = 1              # local linear trend
        if n_seas:                                         # seasonal recursion
            T[2, 2:] = -1
            for i in range(1, n_seas):
                T[2 + i, 2 + i - 1] = 1
        Z = np.zeros(dim); Z[0] = 1
        if n_seas:
            Z[2] = 1
        Q = np.zeros((dim, dim)); Q[0, 0] = lv; Q[1, 1] = sv
        if n_seas:
            Q[2, 2] = se
        return T, Z, Q, ov, dim, n_seas

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        T, Z, Q, ov, dim, n_seas = self._build(y)
        a = np.zeros(dim); a[0] = y[0]
        P = np.eye(dim) * (np.var(y) + 1.0)
        levels, slopes, seas = [], [], []
        for t in range(len(y)):
            # predict
            a = T @ a
            P = T @ P @ T.T + Q
            # update with observation
            v = y[t] - Z @ a
            F = Z @ P @ Z + ov
            K = P @ Z / F
            a = a + K * v
            P = P - np.outer(K, Z @ P)
            levels.append(a[0]); slopes.append(a[1])
            seas.append(a[2] if n_seas else 0.0)
        self._T, self._Z, self._Q, self._ov = T, Z, Q, ov
        self._a, self._P = a, P
        self.trend_ = np.array(levels)
        self.slope_ = np.array(slopes)
        self.seasonal_ = np.array(seas)
        return self

    def forecast(self, steps=10, return_std=False):
        a, P = self._a.copy(), self._P.copy()
        means, variances = [], []
        for _ in range(steps):
            a = self._T @ a
            P = self._T @ P @ self._T.T + self._Q
            means.append(self._Z @ a)
            variances.append(self._Z @ P @ self._Z + self._ov)
        means = np.array(means)
        if not return_std:
            return means
        return means, np.sqrt(np.array(variances))


__all__ = ["BayesianStructuralTimeSeries"]
