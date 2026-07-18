"""Forecast the average demand RATE -- the Croston-style intuition for intermittency.

Instead of chasing spikes, predict the mean of the history (the long-run rate),
which for intermittent demand minimises squared error far better than following
the last value.
"""
import numpy as np


def solve(y_history, horizon):
    return np.full(horizon, float(np.mean(y_history)))
