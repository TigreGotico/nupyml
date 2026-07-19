"""Balanced iterative reducing and clustering using hierarchies."""
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist, pdist, squareform
from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class Birch(BaseEstimator, ClusterMixin, TransformerMixin):
    """Balanced iterative reducing and clustering using hierarchies."""

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        X = check_array(X)
        # single-pass CF construction: absorb into a subcluster if the merged
        # radius stays under threshold, else open a new one
        centroids = []
        counts = []
        sums = []
        for x in X:
            if centroids:
                d = np.linalg.norm(np.asarray(centroids) - x, axis=1)
                j = int(np.argmin(d))
                if d[j] <= self.threshold:
                    counts[j] += 1
                    sums[j] += x
                    centroids[j] = sums[j] / counts[j]
                    continue
            centroids.append(x.copy())
            counts.append(1)
            sums.append(x.copy())
        self.subcluster_centers_ = np.asarray(centroids)
        self.subcluster_counts_ = np.asarray(counts)
        # global clustering over the CF leaves
        if self.n_clusters is not None and len(centroids) > self.n_clusters:
            Z = sch.linkage(self.subcluster_centers_, method="ward")
            self.subcluster_labels_ = sch.fcluster(
                Z, t=self.n_clusters, criterion="maxclust") - 1
        else:
            self.subcluster_labels_ = np.arange(len(centroids))
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        check_is_fitted(self, "subcluster_centers_")
        X = check_array(X)
        nearest = cdist(X, self.subcluster_centers_).argmin(axis=1)
        return self.subcluster_labels_[nearest]

    def transform(self, X):
        check_is_fitted(self, "subcluster_centers_")
        return cdist(check_array(X), self.subcluster_centers_)


__all__ = ["Birch"]
