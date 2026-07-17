"""Manifold learning: non-linear dimensionality reduction.

THE ASSUMPTION
--------------
High-dimensional data usually does not fill its space. Images of a rotating
object live in a million pixel dimensions but are described by one number -- the
angle. The data lies on a low-dimensional MANIFOLD curved through the ambient
space, and the goal is to unroll it.

PCA cannot. It only rotates, so it finds the best flat subspace -- and if the
manifold is curved, like a swiss roll, no flat projection preserves it: PCA
flattens the roll and stacks distant sheets on top of each other.

WHAT "DISTANCE" MEANS ON A MANIFOLD
-----------------------------------
The key idea in ``Isomap``: two points on opposite sheets of a rolled sheet of
paper are close in STRAIGHT-LINE distance and far apart ALONG the paper. The
second is the honest one. Isomap builds a neighbour graph -- trusting straight
lines only locally, where the curve is negligible -- and takes shortest paths
through it as geodesic distances, then hands those to classical MDS.

THE FAMILY
----------
* ``MDS`` -- place points so pairwise distances are preserved as well as
  possible. Linear if given euclidean distances; the engine the others build on.
* ``Isomap`` -- MDS on geodesic distances. Unrolls global structure.
* ``LocallyLinearEmbedding`` -- never uses global distance at all. Each point is
  a weighted blend of its neighbours; find a low-dimensional layout keeping the
  same blends. Local geometry only.
* ``SpectralEmbedding`` -- Laplacian eigenmaps; see ``nupyml.cluster``'s
  spectral discussion for why the Laplacian's bottom eigenvectors encode
  connectivity.
* ``TSNE`` -- see below.

THE CAVEATS THAT MATTER
-----------------------
These are almost all TRANSDUCTIVE: they embed the points you give them and have
no ``transform`` for new data, because the embedding is defined by the whole
graph and not by a function. Add a point and you refit.

And the neighbour graph is a decision, not a detail. Too many neighbours
"short-circuits" the manifold -- an edge across the gap between two sheets makes
them adjacent and the unrolling collapses. Too few disconnects the graph, and
geodesic distance becomes infinite.
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, check_random_state


def _binary_search_perplexity(D_row, target_entropy, tol=1e-5, max_iter=50):
    """Find precision beta so the conditional distribution hits the target
    perplexity for one row of squared distances."""
    beta = 1.0
    beta_min, beta_max = -np.inf, np.inf
    for _ in range(max_iter):
        P = np.exp(-D_row * beta)
        sumP = P.sum()
        if sumP <= 0:
            sumP = 1e-12
        H = np.log(sumP) + beta * (D_row * P).sum() / sumP
        diff = H - target_entropy
        if abs(diff) < tol:
            break
        if diff > 0:
            beta_min = beta
            beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
        else:
            beta_max = beta
            beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
    return P / sumP


class TSNE(BaseEstimator):
    """Exact t-SNE with early exaggeration and momentum (O(n^2))."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 max_iter=1000, early_exaggeration=12.0, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        n = len(X)
        rng = check_random_state(self.random_state)
        D = squareform(pdist(X, "sqeuclidean"))
        target = np.log(self.perplexity)
        P = np.zeros((n, n))
        for i in range(n):
            row = np.delete(D[i], i)
            p = _binary_search_perplexity(row, target)
            P[i, np.arange(n) != i] = p
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)

        Y = rng.normal(scale=1e-4, size=(n, self.n_components))
        velocity = np.zeros_like(Y)
        exag_end = 250
        for it in range(self.max_iter):
            momentum = 0.5 if it < exag_end else 0.8
            Pe = P * self.early_exaggeration if it < exag_end else P
            dist_sq = squareform(pdist(Y, "sqeuclidean"))
            num = 1.0 / (1.0 + dist_sq)
            np.fill_diagonal(num, 0.0)
            Q = np.maximum(num / num.sum(), 1e-12)
            PQ = (Pe - Q) * num
            grad = 4 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)
            velocity = momentum * velocity - self.learning_rate * grad
            Y += velocity
            Y -= Y.mean(axis=0)
        dist_sq = squareform(pdist(Y, "sqeuclidean"))
        num = 1.0 / (1.0 + dist_sq)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        self.kl_divergence_ = float((P * np.log(P / Q)).sum())
        self.embedding_ = Y
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


class Isomap(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_neighbors=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        X = check_array(X)
        n = len(X)
        tree = cKDTree(X)
        dist, idx = tree.query(X, k=self.n_neighbors + 1)
        graph = sp.lil_matrix((n, n))
        for i in range(n):
            for d, j in zip(dist[i, 1:], idx[i, 1:]):
                graph[i, j] = d
        geodesic = sp.csgraph.shortest_path(graph.tocsr(), directed=False)
        if np.isinf(geodesic).any():
            raise ValueError("Isomap graph is not connected; increase n_neighbors")
        # classical MDS on geodesic distances
        D2 = geodesic ** 2
        J = np.eye(n) - np.full((n, n), 1.0 / n)
        B = -0.5 * J @ D2 @ J
        vals, vecs = scipy.linalg.eigh(B)
        order = np.argsort(-vals)[:self.n_components]
        self.embedding_ = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
        return self.embedding_

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        raise NotImplementedError("Isomap supports fit_transform only")


class MDS(BaseEstimator):
    """Metric MDS via SMACOF (or classical if metric='classical')."""

    def __init__(self, n_components=2, metric="smacof", max_iter=300, tol=1e-6,
                 random_state=None):
        self.n_components = n_components
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        n = len(X)
        D = squareform(pdist(X))
        if self.metric == "classical":
            J = np.eye(n) - np.full((n, n), 1.0 / n)
            B = -0.5 * J @ (D ** 2) @ J
            vals, vecs = scipy.linalg.eigh(B)
            order = np.argsort(-vals)[:self.n_components]
            self.embedding_ = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
            return self.embedding_
        rng = check_random_state(self.random_state)
        Y = rng.normal(size=(n, self.n_components))
        prev_stress = np.inf
        for _ in range(self.max_iter):
            d = squareform(pdist(Y))
            stress = ((D - d) ** 2).sum() / 2
            if abs(prev_stress - stress) < self.tol * max(prev_stress, 1.0):
                break
            prev_stress = stress
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(d > 1e-12, D / d, 0.0)
            B = -ratio
            np.fill_diagonal(B, 0.0)
            np.fill_diagonal(B, -B.sum(axis=1))
            Y = B @ Y / n
        self.embedding_ = Y
        self.stress_ = float(stress)
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


class LocallyLinearEmbedding(BaseEstimator):
    def __init__(self, n_components=2, n_neighbors=5, reg=1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg

    def fit_transform(self, X, y=None):
        X = check_array(X)
        n = len(X)
        k = self.n_neighbors
        tree = cKDTree(X)
        _, idx = tree.query(X, k=k + 1)
        idx = idx[:, 1:]
        W = np.zeros((n, n))
        for i in range(n):
            Z = X[idx[i]] - X[i]
            C = Z @ Z.T
            C += np.eye(k) * self.reg * np.trace(C) if np.trace(C) > 0 else \
                np.eye(k) * self.reg
            w = np.linalg.solve(C, np.ones(k))
            W[i, idx[i]] = w / w.sum()
        M = (np.eye(n) - W).T @ (np.eye(n) - W)
        vals, vecs = scipy.linalg.eigh(M)
        # skip the constant eigenvector (smallest eigenvalue ~ 0)
        self.embedding_ = vecs[:, 1:self.n_components + 1]
        return self.embedding_

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


from ._extra import SpectralEmbedding, TSNEBarnesHut  # noqa: E402
from ._umap_som import UMAP, SelfOrganizingMap  # noqa: E402

__all__ = ["TSNE", "Isomap", "MDS", "LocallyLinearEmbedding",
           "SpectralEmbedding", "TSNEBarnesHut", "UMAP", "SelfOrganizingMap"]
