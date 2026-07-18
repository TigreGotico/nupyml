"""Moving average of the recent window."""
import numpy as np


def solve(y_history, horizon):
    w = min(30, len(y_history))
    return np.full(horizon, float(np.mean(y_history[-w:])))
