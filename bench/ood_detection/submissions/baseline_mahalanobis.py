"""Mahalanobis distance to the in-distribution manifold -- the classic OOD score."""
from nupyml.anomaly import MahalanobisDetector


def solve(X):
    return MahalanobisDetector().fit(X).decision_function(X)
