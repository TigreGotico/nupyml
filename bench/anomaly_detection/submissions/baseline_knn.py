"""k-NN outlier score: distance to the k-th nearest neighbour.

A point deep inside a dense cluster has a tiny k-NN distance; one in the sparse
space between clusters has a large one. The intuitive distance-based detector.
"""
from nupyml.anomaly import KNN


def solve(X):
    return KNN(n_neighbors=20).fit(X).decision_function(X)
