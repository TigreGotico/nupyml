"""Volatility forecasting: predict the VARIANCE of a returns series, not its level.

Asset returns are near-unpredictable in the mean but their VARIANCE clusters --
calm and turbulent regimes persist. The task: given a returns history, forecast the
variance of the next ``horizon`` steps. The truth we score against is the realised
squared return of the held-out period (a noisy but unbiased proxy for variance), so
a good forecaster (GARCH, EWMA) that tracks the volatility regime beats a static
one. RMSE over the horizon, lower is better.
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "forecast"
GOAL = "Forecast the variance of a returns series; RMSE vs realised r^2 (lower is better)."
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 3.6           # ceiling: r^2 noise floors the achievable RMSE


def load():
    rng = np.random.RandomState(0)
    n = 800
    omega, alpha, beta = 0.2, 0.15, 0.8
    var = np.empty(n); eps = np.empty(n); var[0] = 1.0; eps[0] = 0.0
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * rng.randn()
    horizon = 30
    # history = returns; target = realised squared returns of the future window
    return eps[:-horizon], eps[-horizon:] ** 2


def metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
