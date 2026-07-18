"""Intermittent-demand forecasting: a series that is mostly ZERO with occasional
spikes.

Spare-parts demand, rare events -- most periods see nothing, a few see a burst.
Standard smoothers chase the spikes badly; methods built for intermittency
(Croston and friends) forecast the small average rate. Scored by RMSE over the
forecast horizon (lower is better).
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "forecast"
GOAL = "Forecast a sparse intermittent-demand series; RMSE (lower is better)."
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 2.1           # ceiling: intermittent variance forces RMSE ~2


def load():
    rng = np.random.RandomState(0)
    n = 260
    occurs = rng.rand(n) < 0.25                        # demand happens ~25% of periods
    sizes = rng.poisson(3, n) + 1
    series = np.where(occurs, sizes, 0).astype(float)
    horizon = 20
    return series[:-horizon], series[-horizon:]


def metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
