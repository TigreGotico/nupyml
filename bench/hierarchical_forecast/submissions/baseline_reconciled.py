"""Baseline: independent AR(1) per series, then MinT reconciliation.

Each series is forecast alone (incoherent), then reconciliation borrows strength
across the hierarchy -- the accurate total forecast corrects the noisier parts.
"""
import numpy as np
from nupyml.timeseries import AutoRegressive, reconcile_forecasts


def solve(y_history, horizon):
    base = np.column_stack([AutoRegressive(p=1).fit(y_history[:, j]).forecast(horizon)
                            for j in range(y_history.shape[1])])
    S = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    return np.array([reconcile_forecasts(base[t], S, method="mint")
                     for t in range(horizon)])
