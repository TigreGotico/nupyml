"""t-SNE with a quadtree-approximated repulsion (O(n log n) per step)."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class _QuadTree:
    """Quadtree over 2-D points; each cell caches its center of mass."""

    __slots__ = ("center", "size", "com", "count", "children", "leaf_point")

    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.com = np.zeros(2)
        self.count = 0
        self.children = None
        self.leaf_point = None

    def _quadrant(self, p):
        return (1 if p[0] > self.center[0] else 0) + \
               (2 if p[1] > self.center[1] else 0)

    def _subdivide(self):
        half = self.size / 2
        quarter = half / 2
        self.children = []
        for q in range(4):
            dx = quarter if q & 1 else -quarter
            dy = quarter if q & 2 else -quarter
            self.children.append(
                _QuadTree(self.center + np.array([dx, dy]), half))

    def insert(self, p, depth=0):
        self.com = (self.com * self.count + p) / (self.count + 1)
        self.count += 1
        if self.count == 1:
            self.leaf_point = p
            return
        if depth > 40:      # coincident points: stop splitting
            return
        if self.children is None:
            self._subdivide()
            if self.leaf_point is not None:
                old = self.leaf_point
                self.leaf_point = None
                self.children[self._quadrant(old)].insert(old, depth + 1)
        self.children[self._quadrant(p)].insert(p, depth + 1)

    def compute_forces(self, p, theta, out):
        """Accumulate the repulsive term; returns the partition-function part."""
        if self.count == 0:
            return 0.0
        diff = p - self.com
        dist_sq = float(diff @ diff)
        if self.children is None or (self.size ** 2 / max(dist_sq, 1e-12)
                                     < theta ** 2):
            if dist_sq < 1e-12:
                return 0.0
            q = 1.0 / (1.0 + dist_sq)
            out += self.count * q * q * diff
            return self.count * q
        z = 0.0
        for child in self.children:
            z += child.compute_forces(p, theta, out)
        return z


def _barnes_hut_repulsion(Y, theta):
    """Repulsive gradient of the t-SNE objective in O(n log n)."""
    n = len(Y)
    center = (Y.max(axis=0) + Y.min(axis=0)) / 2
    size = float(np.max(Y.max(axis=0) - Y.min(axis=0))) + 1e-5
    tree = _QuadTree(center, size)
    for p in Y:
        tree.insert(p)
    forces = np.zeros_like(Y)
    Z = 0.0
    for i in range(n):
        out = np.zeros(2)
        Z += tree.compute_forces(Y[i], theta, out)
        forces[i] = out
    return forces / max(Z, 1e-12)


def _joint_probabilities_nn(X, perplexity, n_neighbors):
    """Sparse P from each point's nearest neighbors only."""
    from . import _binary_search_perplexity
    n = len(X)
    k = min(n_neighbors, n - 1)
    tree = cKDTree(X)
    dist, idx = tree.query(X, k=k + 1)
    dist, idx = dist[:, 1:] ** 2, idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    vals = np.concatenate([_binary_search_perplexity(dist[i], np.log(perplexity))
                           for i in range(n)])
    P = sp.csr_matrix((vals, (rows, idx.ravel())), shape=(n, n))
    P = (P + P.T) / (2 * n)
    P.data = np.maximum(P.data, 1e-12)
    return P


class TSNEBarnesHut(BaseEstimator):
    """t-SNE with a quadtree-approximated repulsion (O(n log n) per step)."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 max_iter=1000, early_exaggeration=12.0, angle=0.5,
                 n_neighbors=None, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.angle = angle
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        if self.n_components != 2:
            raise ValueError("the Barnes-Hut approximation is 2-D only")
        rng = check_random_state(self.random_state)
        n = len(X)
        k = self.n_neighbors or min(n - 1, int(3 * self.perplexity))
        P = _joint_probabilities_nn(X, self.perplexity, k)
        P_coo = P.tocoo()
        rows, cols, pvals = P_coo.row, P_coo.col, P_coo.data

        Y = rng.normal(scale=1e-4, size=(n, 2))
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        exag_end = 250
        for it in range(self.max_iter):
            momentum = 0.5 if it < exag_end else 0.8
            scale = self.early_exaggeration if it < exag_end else 1.0
            # attraction only over the sparse neighbor pairs
            diff = Y[rows] - Y[cols]
            num = 1.0 / (1.0 + (diff ** 2).sum(axis=1))
            attr = np.zeros_like(Y)
            np.add.at(attr, rows, (scale * pvals * num)[:, None] * diff)
            rep = _barnes_hut_repulsion(Y, self.angle)
            grad = 4 * (attr - rep)
            # per-dimension adaptive gains (Jacobs 1988)
            inc = np.sign(grad) != np.sign(velocity)
            gains = np.where(inc, gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            velocity = momentum * velocity - self.learning_rate * gains * grad
            Y += velocity
            Y -= Y.mean(axis=0)
        self.embedding_ = Y
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


__all__ = ["TSNEBarnesHut"]
