"""Baseline: repeat the last observed value -- the null forecaster.

The reference every real method must beat. It captures neither trend nor
season, so its RMSE is the "did you learn anything?" bar.
"""
import numpy as np


def solve(y_history, horizon):
    return np.full(horizon, y_history[-1])
