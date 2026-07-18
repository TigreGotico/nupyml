"""Croston's method -- the classic intermittent-demand forecaster.

Smooths demand sizes and inter-demand intervals separately and forecasts the
rate = size / interval, rather than chasing the sparse spikes.
"""
from nupyml.timeseries import Croston


def solve(y_history, horizon):
    return Croston(alpha=0.1).fit(y_history).predict(horizon)
