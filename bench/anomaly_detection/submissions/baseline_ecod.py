"""ECOD: empirical-CDF outlier detection, parameter-free.

For each feature it estimates the empirical CDF and measures how far into the
TAIL a value sits (both ends), aggregating the tail probabilities across
features. No distance computations and no parameters -- a remarkably strong,
deterministic baseline.
"""
from nupyml.anomaly import ECOD


def solve(X):
    return ECOD().fit(X).decision_function(X)
