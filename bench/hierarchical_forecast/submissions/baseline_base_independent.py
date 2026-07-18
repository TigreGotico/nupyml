"""Baseline: independent AR(1) per series -- forecasts are incoherent."""
import numpy as np
from nupyml.timeseries import AutoRegressive


def solve(y_history, horizon):
    return np.column_stack([AutoRegressive(p=1).fit(y_history[:, j]).forecast(horizon)
                            for j in range(y_history.shape[1])])
