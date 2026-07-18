"""Hierarchical forecasting: forecast a total and its parts COHERENTLY.

Four bottom-level series evolve as a coupled vector-autoregression; the fifth
series is their total. The history is a ``(T, 5)`` array (total first, then the four
parts) and the forecast must be ``(horizon, 5)``. Forecasting each series in
isolation leaves them incoherent (the parts do not add to the total); reconciling
them (bottom-up / MinT) restores coherence without increasing error. Scored by RMSE
over all series, lower is better.
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "forecast"
GOAL = "Forecast a 4-part hierarchy + its total coherently; RMSE (lower is better)."
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 1.6           # ceiling: process noise floors the RMSE


def load():
    rng = np.random.RandomState(0)
    T = 300
    A = np.array([[0.5, 0.1, 0.0, 0.0],
                  [0.1, 0.5, 0.1, 0.0],
                  [0.0, 0.1, 0.5, 0.1],
                  [0.0, 0.0, 0.1, 0.5]])
    B = np.zeros((T, 4)); B[0] = [2, 1, -1, 0]
    for t in range(1, T):
        B[t] = A @ B[t - 1] + 0.3 * rng.randn(4)
    total = B.sum(axis=1, keepdims=True)
    Y = np.hstack([total, B])                          # (T, 5): total then 4 parts
    horizon = 15
    return Y[:-horizon], Y[-horizon:]


def metric(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).reshape(y_true.shape)
    return float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
