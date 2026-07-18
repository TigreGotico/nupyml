"""Baseline: agglomerative (Ward) clustering of the pixels."""
from nupyml.cluster import AgglomerativeClustering


def solve(X):
    return AgglomerativeClustering(n_clusters=4).fit_predict(X)
