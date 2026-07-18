"""Baseline: EWMA / RiskMetrics -- exponentially-weighted variance, held forward."""
import numpy as np


def solve(y_history, horizon):
    lam = 0.94
    var = y_history[0] ** 2
    for r in y_history:
        var = lam * var + (1 - lam) * r ** 2
    return np.full(horizon, var)
