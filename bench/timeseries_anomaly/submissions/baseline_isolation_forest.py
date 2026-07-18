"""Baseline: isolation forest over the per-timestep features."""
from nupyml.outlier import IsolationForest


def solve(X):
    iso = IsolationForest(random_state=0).fit(X)
    return -iso.score_samples(X)                          # higher = more anomalous
