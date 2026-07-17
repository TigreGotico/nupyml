"""Baseline: ARIMA -- differences out the trend, models the rest."""
from nupyml.timeseries import ARIMA


def solve(y_history, horizon):
    return ARIMA(p=2, d=1, q=1).fit(y_history).forecast(horizon)
