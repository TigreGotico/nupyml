"""Large-scale Information Network Embedding (Tang et al., 2015)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


class LINE(BaseEstimator):
    """Large-scale Information Network Embedding (Tang et al., 2015).

    THE OBJECTIVE
    -------------
    Learn a vector per node so that CONNECTED nodes have similar embeddings --
    first-order proximity. LINE maximises, over the observed edges, the
    log-probability ``sigma(u_i · u_j)`` of the edge existing, against
    NEGATIVE-SAMPLED non-edges (random node pairs pushed apart). It is essentially
    node2vec's skip-gram objective applied directly to edges instead of to
    random-walk windows -- simpler, and designed to scale to millions of edges by
    sampling edges and negatives rather than materialising anything dense.

    (Second-order proximity -- sharing NEIGHBOURS -- uses a separate context
    embedding; this implements the first-order variant.)
    """

    def __init__(self, n_components=16, n_epochs=50, lr=0.025, n_negative=5,
                 random_state=None):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def fit(self, A):
        rng = check_random_state(self.random_state)
        B = _binary(A)
        n = B.shape[0]
        edges = np.transpose(np.nonzero(np.triu(B, 1)))
        self.embedding_ = rng.normal(0, 0.1, (n, self.n_components))
        U = self.embedding_
        for _ in range(self.n_epochs):
            rng.shuffle(edges)
            for i, j in edges:
                self._step(U, i, j, 1.0)             # positive edge: pull together
                for _ in range(self.n_negative):
                    k = rng.randint(n)               # negative: push apart
                    self._step(U, i, k, 0.0)
        return self

    def _step(self, U, i, j, label):
        score = 1.0 / (1.0 + np.exp(-np.clip(U[i] @ U[j], -30, 30)))
        g = self.lr * (label - score)
        ui = U[i].copy()
        U[i] += g * U[j]
        U[j] += g * ui

    def fit_transform(self, A):
        return self.fit(A).embedding_


__all__ = ["LINE"]
