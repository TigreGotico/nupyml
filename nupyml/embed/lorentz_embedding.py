"""Lorentz (hyperboloid) embeddings: hyperbolic space in its numerically stable model."""
import numpy as np

from ..base import BaseEstimator
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state


class LorentzEmbedding(BaseEstimator):
    """Embed a HIERARCHY in hyperbolic space, the STABLE way (Nickel & Kiela, 2018).

    The Poincare ball crowds every point toward a unit boundary, where distances blow up
    and gradients vanish -- training is fragile. The Lorentz (hyperboloid) model is the
    same hyperbolic geometry in different coordinates: a point lives on the upper sheet
    of ``-x0^2 + |z|^2 = -1``, so it is fully described by its spatial part ``z`` with
    ``x0 = sqrt(1 + |z|^2)`` DERIVED. That parametrisation has no boundary to fall off,
    the distance ``arccosh(<x, y>_Lorentz)`` is well-conditioned, and ordinary gradient
    steps on ``z`` stay on the manifold automatically -- so it trains where the ball
    diverges. ``fit`` takes hierarchy edges ``(parent, child)``; ``distance`` is the
    hyperbolic distance.
    """

    def __init__(self, dim=2, epochs=100, lr=0.5, n_negative=5, random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    @staticmethod
    def _lorentz_dist(zu, zv):
        # arccosh(x0_u x0_v - z_u . z_v), computed from the spatial parts alone
        x0u = (1.0 + (zu * zu).sum(axis=-1)).sqrt()
        x0v = (1.0 + (zv * zv).sum(axis=-1)).sqrt()
        g = x0u * x0v - (zu * zv).sum(axis=-1)
        g = g.clip(1.0 + 1e-7, 1e9)
        return (g + (g * g - 1.0).sqrt()).log()             # arccosh(g)

    def fit(self, edges):
        rng = check_random_state(self.random_state)
        nodes = sorted({n for e in edges for n in e})
        self.idx_ = {n: k for k, n in enumerate(nodes)}
        self.nodes_ = nodes
        N = len(nodes)
        Z = Tensor(rng.uniform(-0.05, 0.05, (N, self.dim)), requires_grad=True)
        E = [(self.idx_[a], self.idx_[b]) for a, b in edges]
        for _ in range(self.epochs):
            rng.shuffle(E)
            for u, v in E:
                negs = [n for n in (rng.randint(N) for _ in range(self.n_negative))
                        if n != u and n != v]
                if not negs:
                    continue
                cands = [v] + negs
                zu = Z[np.array([u])]
                zc = Z[np.array(cands)]
                d = self._lorentz_dist(zu, zc)              # (len(cands),)
                logp = F.log_softmax((-d).reshape((1, len(cands))), axis=1)
                loss = -logp[np.array([0]), np.array([0])]  # pull the positive closer
                Z.zero_grad()
                loss.backward()
                Z.data -= self.lr * Z.grad                  # Euclidean SGD stays on-manifold
        self.embedding_ = Z.data
        return self

    def distance(self, a, b):
        zu = Tensor(self.embedding_[self.idx_[a]])
        zv = Tensor(self.embedding_[self.idx_[b]])
        return float(self._lorentz_dist(zu, zv).data)


__all__ = ["LorentzEmbedding"]
