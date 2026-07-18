"""Baseline: GARCH(1,1) -- models the conditional variance directly."""
from nupyml.timeseries import GARCH


def solve(y_history, horizon):
    return GARCH().fit(y_history).forecast(horizon)
