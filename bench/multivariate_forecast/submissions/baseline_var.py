"""Baseline: VAR(1) -- uses the cross-series coupling."""
from nupyml.timeseries import VAR


def solve(y_history, horizon):
    return VAR(p=1).fit(y_history).forecast(horizon)
