"""Baseline: k-means on (position, intensity) pixel features."""
from nupyml.cluster import KMeans


def solve(X):
    return KMeans(n_clusters=4, random_state=0).fit_predict(X)
