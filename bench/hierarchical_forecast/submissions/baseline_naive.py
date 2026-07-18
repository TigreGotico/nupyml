"""Baseline: repeat each series' last value."""
import numpy as np


def solve(y_history, horizon):
    return np.tile(y_history[-1], (horizon, 1))
