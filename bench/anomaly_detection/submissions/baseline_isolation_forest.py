"""Isolation Forest: anomalies are easy to ISOLATE with random splits.

Random axis-aligned splits isolate an outlier in very few cuts (short path from
the root), while inliers need many. The averaged path length becomes the score.
"""
import numpy as np

from nupyml.outlier import IsolationForest


def solve(X):
    iso = IsolationForest(n_estimators=100, random_state=0).fit(X)
    # score_samples is higher for inliers; negate so higher = more anomalous
    return -iso.score_samples(X)
