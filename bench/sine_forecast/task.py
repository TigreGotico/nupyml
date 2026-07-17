"""Univariate forecasting: extend a noisy seasonal series.

A submission sees only the history and the horizon, and must return the next
`horizon` values. Scored by RMSE (lower is better) against the held-out
continuation -- so a method that captures the trend AND the seasonality beats one
that captures only the level.
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "forecast"
GOAL = "Forecast the next 24 steps of a trended, seasonal, noisy series."
METRIC = "rmse"
HIGHER_IS_BETTER = False
# a loose QA ceiling: even the naive "repeat last value" reference (~5.3) clears
# it, so it catches gross breakage while the scoreboard shows the real ranking
# (Holt-Winters ~1.2 far ahead). A strong seasonal forecaster stays under ~1.5.
MIN_SCORE = 6.0


def _series():
    t = np.arange(240)
    # level + linear trend + period-12 seasonality + noise
    rng = np.random.RandomState(0)
    return 50 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 240)


def load():
    series = _series()
    horizon = 24
    return series[:-horizon], series[-horizon:]      # (history, held-out future)


def metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)
