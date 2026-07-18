"""Baseline: an independent AR(1) per series -- ignores cross-coupling."""
import numpy as np
from nupyml.timeseries import AutoRegressive


def solve(y_history, horizon):
    out = []
    for j in range(y_history.shape[1]):
        out.append(AutoRegressive(p=1).fit(y_history[:, j]).forecast(horizon))
    return np.column_stack(out)
