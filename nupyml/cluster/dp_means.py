"""k-means that INVENTS clusters as it needs them (Kulis & Jordan, 2012)."""
import numpy as np
from scipy.spatial.distance import cdist
from ..base import BaseEstimator, ClusterMixin, check_is_fitted
from ..utils import check_array, check_random_state


class DPMeans(BaseEstimator, ClusterMixin):
    """k-means that INVENTS clusters as it needs them (Kulis & Jordan, 2012).

    k-means makes you pick ``k`` up front. DP-means replaces that with a single
    distance penalty ``lambda``: assign each point to its nearest centre unless the
    nearest centre is farther than ``sqrt(lambda)``, in which case START A NEW
    CLUSTER centred on that point. It is the small-variance limit of a Dirichlet-
    process mixture -- the same nonparametric "let the data decide how many
    clusters" behaviour, but as a hard-assignment algorithm as cheap as Lloyd's.
    Larger ``lambda`` => fewer, coarser clusters.
    """

    def __init__(self, lam=1.0, max_iter=100, tol=1e-4, random_state=None):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        centers = X[[0]].copy()                       # start with one cluster
        labels = np.zeros(len(X), dtype=int)
        for _ in range(self.max_iter):
            d2 = cdist(X, centers, "sqeuclidean")
            labels = d2.argmin(axis=1)
            nearest = d2[np.arange(len(X)), labels]
            # any point too far from every centre founds a new cluster
            far = np.where(nearest > self.lam)[0]
            if len(far):
                centers = np.vstack([centers, X[far[nearest[far].argmax()]]])
                continue
            new = np.array([X[labels == k].mean(axis=0) for k in range(len(centers))])
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.tol:
                break
        # compact any emptied clusters
        uniq = np.unique(labels)
        remap = {old: i for i, old in enumerate(uniq)}
        self.labels_ = np.array([remap[l] for l in labels])
        self.cluster_centers_ = centers[uniq]
        self.n_clusters_ = len(uniq)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


__all__ = ["DPMeans"]
