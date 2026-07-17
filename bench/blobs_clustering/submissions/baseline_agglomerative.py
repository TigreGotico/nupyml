"""Baseline: agglomerative (Ward) clustering."""
from nupyml.cluster import AgglomerativeClustering


def solve(X):
    return AgglomerativeClustering(n_clusters=5, linkage="ward").fit(X).labels_
