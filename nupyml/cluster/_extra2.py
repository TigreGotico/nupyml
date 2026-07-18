"""Clustering v3: nonparametric k, possibilistic memberships, and subspace
clustering.

Three different escapes from k-means's assumptions. DP-means DROPS the fixed ``k``
(a distance penalty spawns new clusters as needed). Possibilistic c-means DROPS
the "memberships sum to one" rule so outliers can belong to nothing. Sparse
subspace clustering DROPS the idea that a cluster is a blob -- it groups points
that lie in a common low-dimensional SUBSPACE, even when those subspaces cross.
"""
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


class PossibilisticCMeans(BaseEstimator, ClusterMixin):
    """Memberships that DON'T sum to one, so noise can belong to nothing
    (Krishnapuram & Keller, 1993).

    Fuzzy c-means forces every point's memberships across clusters to sum to 1 --
    so an outlier equidistant from two clusters gets 0.5 in each, as if it were a
    confident half-member of both, and it drags the centres toward itself.
    Possibilistic c-means drops the sum-to-one rule: a membership becomes a
    TYPICALITY, ``1 / (1 + (d^2/eta)^{1/(m-1)})``, that depends only on the point's
    distance to THAT cluster. A far outlier gets a low typicality to everything and
    stops distorting the centres -- which is exactly what makes it robust to noise.
    ``eta`` (the per-cluster scale) is bootstrapped from a fuzzy c-means pass.
    """

    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fuzzy_init(self, X, rng):
        # a few fuzzy-c-means iterations to seed centres and the eta scales
        n, k = len(X), self.n_clusters
        U = rng.rand(n, k); U /= U.sum(axis=1, keepdims=True)
        centers = None
        for _ in range(20):
            Um = U ** self.m
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            d2 = cdist(X, centers, "sqeuclidean") + 1e-12
            power = 1.0 / (self.m - 1)
            inv = 1.0 / d2
            U = inv ** power
            U /= U.sum(axis=1, keepdims=True)
        Um = U ** self.m
        eta = (Um * cdist(X, centers, "sqeuclidean")).sum(axis=0) / Um.sum(axis=0)
        return centers, np.maximum(eta, 1e-6)

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        centers, eta = self._fuzzy_init(X, rng)
        power = 1.0 / (self.m - 1)
        T = None
        for _ in range(self.max_iter):
            d2 = cdist(X, centers, "sqeuclidean")
            T = 1.0 / (1.0 + (d2 / eta) ** power)     # typicality, not probability
            Tm = T ** self.m
            new = (Tm.T @ X) / Tm.sum(axis=0)[:, None]
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.tol:
                break
        self.typicalities_ = T
        self.cluster_centers_ = centers
        self.eta_ = eta
        self.labels_ = T.argmax(axis=1)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class SparseSubspaceClustering(BaseEstimator, ClusterMixin):
    """Group points that share a low-dimensional SUBSPACE (Elhamifar & Vidal, 2013).

    When data lie on several intersecting planes -- faces under different lighting,
    motions of different objects -- distance-based clustering fails, because a
    point can be near others from a DIFFERENT plane. The self-expressive idea: any
    point in a subspace is a sparse linear combination of OTHER points from the
    SAME subspace. So reconstruct each point from all the others with a sparse
    code, read those codes as an affinity (who reconstructs whom), and spectral-
    cluster that. Points end up connected only to their own subspace. Here the
    sparse code is found greedily (orthogonal matching pursuit).
    """

    def __init__(self, n_clusters=2, n_nonzero=5, random_state=None):
        self.n_clusters = n_clusters
        self.n_nonzero = n_nonzero
        self.random_state = random_state

    def _omp(self, dictionary, target, k):
        # greedily pick the columns that best reconstruct target
        residual = target.copy()
        idx, n = [], dictionary.shape[1]
        for _ in range(min(k, n)):
            corr = np.abs(dictionary.T @ residual)
            j = int(corr.argmax())
            if j not in idx:
                idx.append(j)
            A = dictionary[:, idx]
            coef, *_ = np.linalg.lstsq(A, target, rcond=None)
            residual = target - A @ coef
            if np.linalg.norm(residual) < 1e-9:
                break
        c = np.zeros(n)
        if idx:
            c[idx] = coef
        return c

    def _spectral(self, affinity):
        # normalised-cut embedding: bottom eigenvectors of the symmetric Laplacian
        import scipy.linalg
        from . import KMeans
        d = affinity.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L = np.eye(len(affinity)) - (dinv[:, None] * affinity * dinv[None, :])
        vals, vecs = scipy.linalg.eigh(L)
        emb = vecs[:, :self.n_clusters]               # smallest eigenvectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return KMeans(n_clusters=self.n_clusters,
                      random_state=self.random_state).fit_predict(emb)

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        C = np.zeros((n, n))
        for i in range(n):
            others = np.delete(np.arange(n), i)
            code = self._omp(X[others].T, X[i], self.n_nonzero)  # exclude self
            C[i, others] = code
        affinity = np.abs(C) + np.abs(C.T)            # symmetric self-expression graph
        self.labels_ = self._spectral(affinity)
        self.affinity_matrix_ = affinity
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


__all__ = ["DPMeans", "PossibilisticCMeans", "SparseSubspaceClustering"]
