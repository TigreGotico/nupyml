"""davies_bouldin_score"""
import numpy as np
from ..utils import column_or_1d


def davies_bouldin_score(X, labels):
    from scipy.spatial.distance import cdist
    X = np.asarray(X, dtype=np.float64)
    labels = column_or_1d(labels)
    uniq = np.unique(labels)
    centroids = np.array([X[labels == c].mean(axis=0) for c in uniq])
    spreads = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
        for i, c in enumerate(uniq)])
    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)
    ratios = (spreads[:, None] + spreads[None, :]) / dists
    return float(np.max(ratios, axis=1).mean())


__all__ = ["davies_bouldin_score"]
