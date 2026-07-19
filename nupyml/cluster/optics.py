"""Ordering points to identify the clustering structure."""
import numpy as np
from scipy.spatial import cKDTree
from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class OPTICS(BaseEstimator, ClusterMixin):
    """Ordering points to identify the clustering structure."""

    def __init__(self, min_samples=5, max_eps=np.inf, xi=0.05,
                 cluster_method="xi", eps=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.xi = xi
        self.cluster_method = cluster_method
        self.eps = eps

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        tree = cKDTree(X)
        k = min(self.min_samples, n)
        core_dist, _ = tree.query(X, k=k)
        core_distances = core_dist[:, -1] if k > 1 else np.zeros(n)
        core_distances[core_distances > self.max_eps] = np.inf

        reachability = np.full(n, np.inf)
        processed = np.zeros(n, dtype=bool)
        ordering = []
        for start in range(n):
            if processed[start]:
                continue
            # expand the cluster order from this seed
            seeds = [start]
            while seeds:
                # pick the unprocessed point with the smallest reachability
                seeds = [s for s in seeds if not processed[s]]
                if not seeds:
                    break
                i = min(seeds, key=lambda s: reachability[s])
                seeds.remove(i)
                processed[i] = True
                ordering.append(i)
                if not np.isfinite(core_distances[i]):
                    continue
                neighbors = tree.query_ball_point(X[i], min(self.max_eps, 1e12))
                for j in neighbors:
                    if processed[j]:
                        continue
                    new_reach = max(core_distances[i],
                                    np.linalg.norm(X[i] - X[j]))
                    if new_reach < reachability[j]:
                        reachability[j] = new_reach
                    if j not in seeds:
                        seeds.append(j)
        self.ordering_ = np.asarray(ordering)
        self.reachability_ = reachability
        self.core_distances_ = core_distances
        self.labels_ = self._extract_labels(X)
        return self

    def _extract_labels(self, X):
        """DBSCAN-style extraction at ``eps`` from the reachability plot."""
        eps = self.eps if self.eps is not None else (
            np.percentile(self.reachability_[np.isfinite(self.reachability_)],
                          90) if np.isfinite(self.reachability_).any() else 1.0)
        labels = np.full(len(X), -1)
        cluster = -1
        for i in self.ordering_:
            if self.reachability_[i] > eps:
                if self.core_distances_[i] <= eps:
                    cluster += 1
                    labels[i] = cluster
                else:
                    labels[i] = -1
            else:
                labels[i] = cluster if cluster >= 0 else -1
        return labels


__all__ = ["OPTICS"]
