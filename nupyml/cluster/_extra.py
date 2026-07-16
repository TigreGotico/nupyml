"""Birch, OPTICS, AffinityPropagation, and HDBSCAN."""
import warnings

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class _CFNode:
    """Clustering-feature node: keeps (n, linear sum) per subcluster."""

    __slots__ = ("is_leaf", "subclusters", "children", "branching_factor",
                 "threshold")

    def __init__(self, threshold, branching_factor, is_leaf):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.subclusters = []      # list of [n, linear_sum]
        self.children = []


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


class AffinityPropagation(BaseEstimator, ClusterMixin):
    """Message passing between points to pick exemplars."""

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15,
                 preference=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        S = -cdist(X, X, "sqeuclidean")
        pref = self.preference if self.preference is not None else np.median(S)
        np.fill_diagonal(S, pref)
        A = np.zeros((n, n))
        R = np.zeros((n, n))
        d = self.damping
        last_exemplars = None
        stable = 0
        converged = False
        for it in range(self.max_iter):
            # responsibilities
            AS = A + S
            idx_max = AS.argmax(axis=1)
            first_max = AS[np.arange(n), idx_max]
            AS[np.arange(n), idx_max] = -np.inf
            second_max = AS.max(axis=1)
            R_new = S - first_max[:, None]
            R_new[np.arange(n), idx_max] = S[np.arange(n), idx_max] - second_max
            R = d * R + (1 - d) * R_new
            # availabilities
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, np.diag(R))
            A_new = Rp.sum(axis=0)[None, :] - Rp
            dA = np.diag(A_new).copy()
            A_new = np.minimum(A_new, 0)
            np.fill_diagonal(A_new, dA)
            A = d * A + (1 - d) * A_new
            exemplars = np.where(np.diag(A) + np.diag(R) > 0)[0]
            if last_exemplars is not None and np.array_equal(exemplars,
                                                             last_exemplars):
                stable += 1
                if stable >= self.convergence_iter:
                    converged = True
                    break
            else:
                stable = 0
            last_exemplars = exemplars
        if len(exemplars) == 0:   # degenerate: everything in one cluster
            exemplars = np.array([int(np.argmax(np.diag(S)))])
        self.converged_ = converged
        if not converged:
            warnings.warn(
                "Affinity propagation did not converge; the exemplars and "
                "labels may be degenerate. Try raising damping or max_iter, "
                "or lowering preference.", UserWarning, stacklevel=2)
        self.cluster_centers_indices_ = exemplars
        self.cluster_centers_ = X[exemplars]
        self.labels_ = cdist(X, self.cluster_centers_).argmin(axis=1)
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_).argmin(axis=1)


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


__all__ = ["Birch", "OPTICS", "AffinityPropagation", "HDBSCAN"]
