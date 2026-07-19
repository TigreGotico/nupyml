"""Preserve k-STEP proximity, one order at a time (Cao et al., 2015)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class GraRep(BaseEstimator):
    """Preserve k-STEP proximity, one order at a time (Cao et al., 2015).

    node2vec blends all walk lengths into one objective. GraRep separates them: for
    each order ``k = 1..K`` it forms the ``k``-step transition matrix ``P^k`` (where a
    random walker is after k hops), turns it into a shifted-log co-occurrence matrix
    (the same target skip-gram implicitly factorises), factorises THAT with an SVD to
    get a ``k``-order embedding, and CONCATENATES the orders. So the final embedding
    explicitly carries direct-neighbour, 2-hop, 3-hop, ... structure in separate
    blocks, which captures long-range community structure a single order misses.
    """

    def __init__(self, dim=16, max_order=3, beta=None):
        self.dim = dim
        self.max_order = max_order
        self.beta = beta

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        d = A.sum(axis=1)
        P = A / np.maximum(d[:, None], 1e-12)             # transition matrix
        beta = self.beta if self.beta is not None else 1.0 / n
        per_order = self.dim // self.max_order
        blocks = []
        Pk = np.eye(n)
        for _ in range(self.max_order):
            Pk = Pk @ P
            M = np.log(np.maximum(Pk / beta, 1.0))        # shifted-log co-occurrence
            U, s, _ = np.linalg.svd(M)
            blocks.append(U[:, :per_order] * np.sqrt(s[:per_order]))
        self.embedding_ = np.hstack(blocks)
        return self

    def get_vector(self, node):
        return self.embedding_[node]


__all__ = ["GraRep"]
