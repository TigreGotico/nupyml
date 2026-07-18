"""Multivariate forecasting: several series that drive EACH OTHER.

Three coupled series where each one's next value depends on the recent past of ALL
of them (a vector-autoregressive process). Forecasting them jointly -- using the
cross-series structure -- should beat forecasting each series in isolation. The
history is a ``(T, k)`` array; the forecast is a ``(horizon, k)`` array. Scored by
RMSE over all series and steps, lower is better.
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "forecast"
GOAL = "Jointly forecast 3 coupled series; RMSE over all series (lower is better)."
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 1.2           # ceiling: process noise floors the RMSE


def load():
    rng = np.random.RandomState(0)
    T = 400
    A = np.array([[0.5, 0.2, 0.0],
                  [-0.2, 0.6, 0.1],
                  [0.1, -0.1, 0.5]])
    Y = np.zeros((T, 3)); Y[0] = [1, 0, -1]
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + 0.3 * rng.randn(3)
    horizon = 15
    return Y[:-horizon], Y[-horizon:]


def metric(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred).reshape(y_true.shape)
    return float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
