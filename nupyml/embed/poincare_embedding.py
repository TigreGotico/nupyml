"""Embed a HIERARCHY in HYPERBOLIC space (Nickel & Kiela, 2017)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class PoincareEmbedding(BaseEstimator):
    """Embed a HIERARCHY in HYPERBOLIC space (Nickel & Kiela, 2017).

    Trees do not fit in Euclidean space: the number of nodes grows exponentially
    with depth, but Euclidean volume grows only polynomially, so a tree gets
    badly distorted. Hyperbolic space's volume grows EXPONENTIALLY, matching the
    tree, so a hierarchy embeds with tiny distortion in just 2 dimensions. Points
    live in the Poincare BALL; distance uses the hyperbolic metric, and general
    concepts naturally settle near the ORIGIN while specific ones push toward the
    boundary. Trained by Riemannian SGD on related/unrelated pairs.

    ``fit`` takes hierarchy edges ``(parent, child)``; ``distance`` gives the
    hyperbolic distance.
    """

    def __init__(self, dim=2, epochs=100, lr=0.1, n_negative=5, random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def _dist(self, u, v):
        sq = np.sum((u - v) ** 2)
        nu = np.sum(u ** 2); nv = np.sum(v ** 2)
        return np.arccosh(1 + 2 * sq / ((1 - nu) * (1 - nv) + 1e-9))

    def _dist_grad(self, u, v):
        """Euclidean gradient of the Poincare distance d(u,v) w.r.t. u."""
        a = 1 - np.sum(u ** 2)
        b = 1 - np.sum(v ** 2)
        uv = np.sum((u - v) ** 2)
        gamma = 1 + 2 * uv / (a * b + 1e-9)
        c = 4 / (b * np.sqrt(gamma ** 2 - 1) + 1e-9)
        return c * ((np.sum(v ** 2) - 2 * (u @ v) + 1) / (a ** 2 + 1e-9) * u - v / (a + 1e-9))

    def fit(self, edges):
        rng = check_random_state(self.random_state)
        nodes = sorted({n for e in edges for n in e})
        self.idx_ = {n: k for k, n in enumerate(nodes)}
        self.nodes_ = nodes
        N = len(nodes)
        self.emb_ = rng.uniform(-0.05, 0.05, (N, self.dim))
        E = [(self.idx_[a], self.idx_[b]) for a, b in edges]
        for _ in range(self.epochs):
            rng.shuffle(E)
            for u, v in E:
                negs = [rng.randint(N) for _ in range(self.n_negative)]
                self._step(u, v, negs)
        self.embedding_ = self.emb_
        return self

    def _riemannian(self, node, egrad):
        scale = ((1 - np.sum(self.emb_[node] ** 2)) ** 2) / 4   # inverse metric
        self.emb_[node] = self._project(self.emb_[node] - self.lr * scale * egrad)

    def _step(self, u, v, negs):
        # softmax over -distance to {positive v} + negatives; pull v in, push negs out
        cands = [v] + [n for n in negs if n != u]
        d = np.array([self._dist(self.emb_[u], self.emb_[c]) for c in cands])
        expd = np.exp(-d); p = expd / expd.sum()
        grad_u = np.zeros(self.dim)
        for rank, c in enumerate(cands):
            coeff = -(p[rank] - (1.0 if rank == 0 else 0.0))   # dLoss/d(-d) chain
            gu = self._dist_grad(self.emb_[u], self.emb_[c])
            gc = self._dist_grad(self.emb_[c], self.emb_[u])
            grad_u += coeff * gu
            self._riemannian(c, coeff * gc)            # move the endpoint too
        self._riemannian(u, grad_u)

    def _project(self, x):
        norm = np.linalg.norm(x)
        return x / norm * (1 - 1e-5) if norm >= 1 else x

    def distance(self, a, b):
        return float(self._dist(self.emb_[self.idx_[a]], self.emb_[self.idx_[b]]))

    def norm(self, a):
        return float(np.linalg.norm(self.emb_[self.idx_[a]]))


__all__ = ["PoincareEmbedding"]
