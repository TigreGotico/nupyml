"""Baseline: the unconditional (historical) variance -- ignores clustering."""
import numpy as np


def solve(y_history, horizon):
    return np.full(horizon, y_history.var())
