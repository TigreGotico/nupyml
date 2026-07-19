"""Hierarchical DBSCAN over the mutual-reachability minimum spanning tree."""
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform
from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class HDBSCAN(BaseEstimator, ClusterMixin):
    """Hierarchical DBSCAN over the mutual-reachability minimum spanning tree."""

    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        min_samples = self.min_samples or self.min_cluster_size
        k = min(min_samples, n)
        tree = cKDTree(X)
        dist, _ = tree.query(X, k=k)
        core = dist[:, -1] if k > 1 else np.zeros(n)
        # mutual reachability: max(core_i, core_j, d_ij)
        D = squareform(pdist(X))
        mutual = np.maximum(np.maximum(core[:, None], core[None, :]), D)
        # single-linkage over the mutual-reachability metric is the MST
        Z = sch.linkage(squareform(mutual, checks=False), method="single")
        # condense: cut the dendrogram so every cluster meets min_cluster_size
        best_labels = np.zeros(n, dtype=int)
        best_score = -np.inf
        for t in np.unique(Z[:, 2]):
            labels = sch.fcluster(Z, t=t, criterion="distance") - 1
            uniq, counts = np.unique(labels, return_counts=True)
            valid = uniq[counts >= self.min_cluster_size]
            if len(valid) < 2:
                continue
            # prefer the cut yielding the most points inside valid clusters
            covered = np.isin(labels, valid).sum()
            score = covered * len(valid)
            if score > best_score:
                best_score = score
                out = np.full(n, -1)
                for new, c in enumerate(valid):
                    out[labels == c] = new
                best_labels = out
        self.labels_ = best_labels
        self.linkage_matrix_ = Z
        return self


__all__ = ["HDBSCAN"]
