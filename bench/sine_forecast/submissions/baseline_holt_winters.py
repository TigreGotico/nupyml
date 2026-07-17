"""Baseline: Holt-Winters -- level + trend + seasonality, the right shape here."""
from nupyml.timeseries import HoltWinters


def solve(y_history, horizon):
    model = HoltWinters(season_length=12, trend=True, seasonal="additive").fit(
        y_history)
    return model.forecast(horizon)
