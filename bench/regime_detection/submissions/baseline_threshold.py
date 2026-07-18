"""Baseline: threshold the rolling mean at its median (regime by level)."""
import numpy as np


def solve(X):
    return (X[:, 1] > np.median(X[:, 1])).astype(int)
