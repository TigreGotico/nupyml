"""Clustering algorithms."""
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _kmeans_plusplus(X, k, rng):
    n = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.randint(n)]
    closest_sq = cdist(X, centers[:1]).ravel() ** 2
    for i in range(1, k):
        probs = closest_sq / closest_sq.sum()
        centers[i] = X[rng.choice(n, p=probs)]
        d = cdist(X, centers[i:i + 1]).ravel() ** 2
        closest_sq = np.minimum(closest_sq, d)
    return centers


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _single_run(self, X, rng, w):
        k = self.n_clusters
        if self.init == "k-means++":
            centers = _kmeans_plusplus(X, k, rng)
        elif self.init == "random":
            centers = X[rng.choice(len(X), size=k, replace=False)]
        else:
            centers = np.asarray(self.init, dtype=np.float64).copy()
        for _ in range(self.max_iter):
            dist = cdist(X, centers)
            labels = dist.argmin(axis=1)
            new_centers = np.empty_like(centers)
            for c in range(k):
                mask = labels == c
                if mask.any() and w[mask].sum() > 0:
                    new_centers[c] = np.average(X[mask], axis=0, weights=w[mask])
                else:  # dead cluster: reseed at farthest point
                    new_centers[c] = X[dist.min(axis=1).argmax()]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break
        dist = cdist(X, centers)
        labels = dist.argmin(axis=1)
        inertia = float((w * dist[np.arange(len(X)), labels] ** 2).sum())
        return centers, labels, inertia

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        rng = check_random_state(self.random_state)
        runs = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(runs):
            centers, labels, inertia = self._single_run(X, rng, w)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_).argmin(axis=1)

    def transform(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_)


class MiniBatchKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, batch_size=256, max_iter=100,
                 random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        centers = _kmeans_plusplus(X, self.n_clusters, rng)
        counts = np.zeros(self.n_clusters)
        n = len(X)
        for _ in range(self.max_iter):
            batch = X[rng.randint(0, n, size=min(self.batch_size, n))]
            labels = cdist(batch, centers).argmin(axis=1)
            for c in np.unique(labels):
                pts = batch[labels == c]
                counts[c] += len(pts)
                eta = len(pts) / counts[c]
                centers[c] = (1 - eta) * centers[c] + eta * pts.mean(axis=0)
        self.cluster_centers_ = centers
        self.labels_ = cdist(X, centers).argmin(axis=1)
        return self

    def partial_fit(self, X, y=None):
        """Update centers from one mini-batch of data."""
        X = check_array(X)
        if not hasattr(self, "cluster_centers_"):
            rng = check_random_state(self.random_state)
            k = min(self.n_clusters, len(X))
            self.cluster_centers_ = _kmeans_plusplus(X, k, rng)
            self._counts = np.zeros(len(self.cluster_centers_))
        labels = cdist(X, self.cluster_centers_).argmin(axis=1)
        for c in np.unique(labels):
            pts = X[labels == c]
            self._counts[c] += len(pts)
            eta = len(pts) / self._counts[c]
            self.cluster_centers_[c] = ((1 - eta) * self.cluster_centers_[c]
                                        + eta * pts.mean(axis=0))
        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_).argmin(axis=1)


class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        tree = cKDTree(X)
        neighbors = tree.query_ball_point(X, self.eps)
        core = np.array([len(nb) >= self.min_samples for nb in neighbors])
        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not core[i]:
                continue
            # BFS expand cluster
            labels[i] = cluster
            queue = list(neighbors[i])
            while queue:
                j = queue.pop()
                if labels[j] == -1:
                    labels[j] = cluster
                    if core[j]:
                        queue.extend(q for q in neighbors[j] if labels[q] == -1)
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where(core)[0]
        return self


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        X = check_array(X)
        Z = sch.linkage(X, method=self.linkage)
        self.labels_ = sch.fcluster(Z, t=self.n_clusters, criterion="maxclust") - 1
        self.linkage_matrix_ = Z
        return self


class MeanShift(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=None, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X = check_array(X)
        bw = self.bandwidth or float(np.mean(cdist(X, X).std(axis=1)))
        tree = cKDTree(X)
        points = X.copy()
        for _ in range(self.max_iter):
            moved = 0.0
            for i in range(len(points)):
                nb = tree.query_ball_point(points[i], bw)
                if not nb:
                    continue
                new = X[nb].mean(axis=0)
                moved = max(moved, np.linalg.norm(new - points[i]))
                points[i] = new
            if moved < self.tol:
                break
        # merge modes closer than bandwidth/2
        centers = []
        labels = np.full(len(X), -1)
        for i, p in enumerate(points):
            for ci, c in enumerate(centers):
                if np.linalg.norm(p - c) < bw / 2:
                    labels[i] = ci
                    break
            else:
                centers.append(p)
                labels[i] = len(centers) - 1
        self.cluster_centers_ = np.array(centers)
        self.labels_ = labels
        return self


class SpectralClustering(BaseEstimator, ClusterMixin):
    """Normalized-cuts spectral clustering with an RBF or kNN affinity."""

    def __init__(self, n_clusters=8, affinity="rbf", gamma=1.0, n_neighbors=10,
                 random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        if self.affinity == "rbf":
            W = np.exp(-self.gamma * cdist(X, X) ** 2)
        elif self.affinity == "nearest_neighbors":
            tree = cKDTree(X)
            _, idx = tree.query(X, k=self.n_neighbors + 1)
            W = np.zeros((n, n))
            rows = np.repeat(np.arange(n), self.n_neighbors)
            W[rows, idx[:, 1:].ravel()] = 1.0
            W = np.maximum(W, W.T)
        else:
            raise ValueError(f"Unknown affinity: {self.affinity!r}")
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        # smallest eigenvectors of the normalized laplacian
        k = self.n_clusters
        if n <= 200:
            vals, vecs = np.linalg.eigh(L_sym)
            U = vecs[:, :k]
        else:
            vals, U = sp.linalg.eigsh(sp.csr_matrix(L_sym), k=k, which="SM")
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        km = KMeans(n_clusters=k, random_state=self.random_state).fit(U)
        self.labels_ = km.labels_
        self.affinity_matrix_ = W
        return self


from ._extra import Birch, OPTICS, AffinityPropagation, HDBSCAN  # noqa: E402

__all__ = ["KMeans", "MiniBatchKMeans", "DBSCAN", "AgglomerativeClustering",
           "MeanShift", "SpectralClustering", "Birch", "OPTICS",
           "AffinityPropagation", "HDBSCAN"]
