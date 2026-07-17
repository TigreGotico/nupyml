"""HBOS: histogram-based outlier score.

Build a histogram per feature; a point's score is the sum of the (log) inverse
densities of the bins it falls into. Assumes feature independence, which makes it
extremely fast and a strong baseline when anomalies are extreme on some axis.
"""
from nupyml.anomaly import HBOS


def solve(X):
    return HBOS().fit(X).decision_function(X)
