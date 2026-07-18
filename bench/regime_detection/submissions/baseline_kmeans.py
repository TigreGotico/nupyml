"""Baseline: k-means on the rolling-statistic features."""
from nupyml.cluster import KMeans
from nupyml.preprocessing import StandardScaler


def solve(X):
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=2, random_state=0).fit_predict(Xs)
