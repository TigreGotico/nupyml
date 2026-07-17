"""Baseline: k-means -- the natural fit for round, separated blobs."""
from nupyml.cluster import KMeans


def solve(X):
    return KMeans(n_clusters=5, n_init=10, random_state=0).fit(X).labels_
