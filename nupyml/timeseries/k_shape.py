"""k-Shape: cluster series by shape, invariant to shift and scale."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


def _sbd(a, b):
    """Shape-based distance: 1 - max normalised cross-correlation over all shifts."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    cc = np.correlate(a, b, mode="full")
    ncc = cc / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return 1 - ncc.max()


class KShape(BaseEstimator):
    """Cluster series by SHAPE, ignoring shift and scale (Paparrizos & Gravano, 2015).

    Two heartbeats are the "same shape" even if one is shifted in time or scaled in
    amplitude -- but Euclidean k-means calls them different. k-Shape uses a SHAPE-based
    distance built on normalised cross-correlation (which maximises over all
    alignments), and updates each centroid by extracting the shape that best aligns
    with its members (the top eigenvector of their aligned covariance). So it groups
    series by their pattern, invariant to phase and amplitude -- the right notion for
    most time-series clustering. Series are z-normalised internally.
    """

    def __init__(self, n_clusters=2, max_iter=50, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        labels = rng.randint(0, self.n_clusters, n)
        Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
        for _ in range(self.max_iter):
            centroids = np.array([self._extract_shape(Xn[labels == k], Xn.shape[1])
                                  for k in range(self.n_clusters)])
            new = np.array([min(range(self.n_clusters),
                                key=lambda k: _sbd(Xn[i], centroids[k]))
                            for i in range(n)])
            if np.array_equal(new, labels):
                break
            labels = new
        self.labels_ = labels
        self.centroids_ = centroids
        return self

    def _extract_shape(self, members, length):
        if len(members) == 0:
            return np.zeros(length)
        # centroid = top eigenvector of the members' covariance (the common shape)
        M = members - members.mean(axis=1, keepdims=True)
        S = M.T @ M
        vals, vecs = np.linalg.eigh(S)
        c = vecs[:, -1]
        if (c @ members.mean(axis=0)) < 0:
            c = -c
        return (c - c.mean()) / (c.std() + 1e-9)

    def fit_predict(self, X):
        return self.fit(X).labels_


__all__ = ["KShape"]
