"""UMAP and self-organizing maps: two more ways to see high-dimensional data.

Both sit beside t-SNE in the manifold module: nonlinear dimensionality reduction
for visualisation, each with a different idea about what "preserve the structure"
should mean.
"""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class UMAP(BaseEstimator, TransformerMixin):
    """Uniform Manifold Approximation and Projection (a faithful, compact version).

    THE IDEA, AND HOW IT DIFFERS FROM t-SNE
    ---------------------------------------
    Like t-SNE, UMAP builds a graph of local neighbourhoods in high dimensions and
    then lays it out in low dimensions so that neighbours stay neighbours. The
    difference that matters in practice:

    * t-SNE optimises a divergence that cares almost only about LOCAL structure,
      so its global layout -- the distances BETWEEN clusters -- is largely
      meaningless.
    * UMAP's objective (a cross-entropy over fuzzy edge probabilities) balances
      attraction of true neighbours against repulsion of everything else in a way
      that preserves more GLOBAL structure, so the relative placement of clusters
      carries some signal.

    It is also faster, because it optimises the layout by sampled attraction and
    repulsion (like a force-directed graph) rather than an all-pairs computation.

    THE HONEST CAVEAT
    -----------------
    UMAP's global structure is BETTER than t-SNE's, not TRUSTWORTHY. Cluster sizes
    and inter-cluster distances in ANY neighbour-embedding are shaped by the
    algorithm as much as by the data; read them as suggestive, never as
    measurements. This is the compact projected-gradient version -- the real UMAP
    adds a fuzzy-simplicial-set construction and careful edge weighting, but the
    attraction/repulsion heart is here.

    McInnes, Healy & Melville (2018).
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1,
                 n_epochs=200, learning_rate=1.0, random_state=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        k = min(self.n_neighbors, n - 1)

        # the high-dimensional neighbour graph: each point's k nearest, with edge
        # weights that decay from an adaptive local scale (so dense and sparse
        # regions are treated comparably -- UMAP's local-connectivity idea)
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        knn = np.argsort(D, axis=1)[:, :k]
        edges = []
        for i in range(n):
            rho = D[i, knn[i, 0]]                   # distance to nearest neighbour
            sigma = np.mean(D[i, knn[i]]) + 1e-9
            for j in knn[i]:
                w = np.exp(-(max(D[i, j] - rho, 0)) / sigma)
                edges.append((i, j, w))

        # low-dim init and sampled attraction/repulsion, force-directed style
        Y = rng.normal(0, 1e-4, size=(n, self.n_components))
        edges_arr = np.array([(i, j) for i, j, _ in edges])
        weights = np.array([w for _, _, w in edges])
        lr = self.learning_rate

        for epoch in range(self.n_epochs):
            alpha = lr * (1 - epoch / self.n_epochs)   # anneal the step size
            for (i, j), w in zip(edges_arr, weights):
                # ATTRACT true neighbours
                diff = Y[i] - Y[j]
                d2 = diff @ diff + 1e-9
                grad = -2 * w / (1 + d2) * diff
                Y[i] += alpha * np.clip(grad, -4, 4)
                Y[j] -= alpha * np.clip(grad, -4, 4)
                # REPEL a random non-neighbour (negative sampling, as in word2vec)
                r = rng.randint(n)
                diff_r = Y[i] - Y[r]
                d2r = diff_r @ diff_r + 1e-9
                grad_r = 2 / ((1 + d2r) * d2r) * diff_r
                Y[i] += alpha * np.clip(grad_r, -4, 4)

        self.embedding_ = Y
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class SelfOrganizingMap(BaseEstimator, TransformerMixin):
    """A Kohonen map: neurons on a grid that learn to tile the data.

    THE IDEA
    --------
    Lay out a grid of neurons, each holding a weight vector in the data's space.
    Present a data point, find the BEST-MATCHING neuron (closest weight), and pull
    it AND ITS GRID NEIGHBOURS toward the point. Repeat.

    The result is a map with a property nothing else here has: neurons that are
    adjacent ON THE GRID end up with similar weights, because they were always
    updated together. So the 2D grid becomes a TOPOLOGY-PRESERVING chart of the
    high-dimensional data -- nearby regions of the map correspond to nearby
    regions of the data. It is dimensionality reduction and clustering at once,
    onto a layout you can literally look at.

    THE TWO SCHEDULES
    -----------------
    Both the learning rate and the NEIGHBOURHOOD RADIUS shrink over training. Early
    on a broad neighbourhood drags whole regions of the grid into rough position
    (global ordering); later a narrow one lets individual neurons fine-tune
    (local detail). Starting narrow would freeze a tangled map; the shrinking
    radius is what lets it unfold smoothly -- the same coarse-to-fine idea as
    simulated annealing's temperature.

    Kohonen (1982).
    """

    def __init__(self, grid_shape=(10, 10), n_epochs=100, learning_rate=0.5,
                 sigma=None, random_state=None):
        self.grid_shape = grid_shape
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        rows, cols = self.grid_shape
        n_neurons = rows * cols
        self.weights_ = rng.uniform(X.min(), X.max(),
                                    size=(n_neurons, X.shape[1]))
        # grid coordinates, so "neighbour on the grid" has a meaning
        coords = np.array([(r, c) for r in range(rows) for c in range(cols)],
                          dtype=float)
        sigma0 = self.sigma or max(rows, cols) / 2.0
        lr0 = self.learning_rate

        for epoch in range(self.n_epochs):
            # both schedules decay: broad-then-narrow neighbourhood, high-then-low
            # rate -- global ordering first, local detail after
            frac = epoch / self.n_epochs
            sigma = sigma0 * (1 - frac) + 0.5 * frac
            lr = lr0 * (1 - frac) + 0.01 * frac
            for x in X[rng.permutation(len(X))]:
                # best-matching unit: the neuron whose weight is nearest x
                bmu = np.argmin(np.sum((self.weights_ - x) ** 2, axis=1))
                # pull the BMU and its grid-neighbours toward x, weighted by a
                # gaussian of grid distance -- this is what couples grid adjacency
                # to weight similarity
                grid_dist2 = np.sum((coords - coords[bmu]) ** 2, axis=1)
                influence = np.exp(-grid_dist2 / (2 * sigma ** 2))
                self.weights_ += lr * influence[:, None] * (x - self.weights_)

        self._coords = coords
        return self

    def transform(self, X):
        """Map each point to the grid coordinates of its best-matching neuron."""
        check_is_fitted(self, "weights_")
        X = check_array(X)
        bmus = np.argmin(cdist(X, self.weights_), axis=1)
        return self._coords[bmus]

    def winner(self, x):
        return int(np.argmin(np.sum((self.weights_ - np.asarray(x)) ** 2, axis=1)))


__all__ = ["UMAP", "SelfOrganizingMap"]
