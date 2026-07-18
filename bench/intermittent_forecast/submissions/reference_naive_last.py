"""Repeat the last observed value -- terrible on intermittent data (the floor)."""
import numpy as np


def solve(y_history, horizon):
    return np.full(horizon, float(y_history[-1]))
