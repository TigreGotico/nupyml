"""DeepWalk/node2vec are secretly MATRIX FACTORISATION (Qiu et al., 2018)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class NetMF(BaseEstimator):
    """DeepWalk/node2vec are secretly MATRIX FACTORISATION (Qiu et al., 2018).

    Node2vec looks like a neural method, but Qiu et al. proved that skip-gram over
    random walks is implicitly factorising a specific matrix built from powers of
    the normalised adjacency. NetMF forms that matrix EXPLICITLY -- a log-transformed,
    window-weighted sum of transition-matrix powers -- and factorises it with a
    truncated SVD. No sampling, no SGD, and often better embeddings, because it uses
    the exact target the random-walk method only approximates. ``window`` is the
    skip-gram context size, ``negative`` its negative-sample count.
    """

    def __init__(self, dim=16, window=3, negative=1.0):
        self.dim = dim
        self.window = window
        self.negative = negative

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        deg = A.sum(axis=1)
        vol = A.sum()
        Dinv = np.diag(1.0 / np.maximum(deg, 1e-12))
        P = Dinv @ A                                     # transition matrix
        S = np.zeros((n, n))
        Pr = np.eye(n)
        for _ in range(self.window):                     # sum of transition powers
            Pr = Pr @ P
            S += Pr
        S = S / self.window
        M = (vol / self.negative) * S @ Dinv
        M = np.log(np.maximum(M, 1.0))                   # the implicit skip-gram matrix
        U, s, _ = np.linalg.svd(M)
        self.embedding_ = U[:, :self.dim] * np.sqrt(s[:self.dim])
        return self

    def get_vector(self, node):
        return self.embedding_[node]


__all__ = ["NetMF"]
