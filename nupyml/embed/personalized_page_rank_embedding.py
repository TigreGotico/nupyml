"""Embed a node by WHERE a walk from it lands (Tsitsulin et al., 2018)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class PersonalizedPageRankEmbedding(BaseEstimator):
    """Embed a node by WHERE a walk from it lands (Tsitsulin et al., 2018).

    A node's role is captured by the set of nodes a random walk STARTING there tends
    to visit -- its personalised PageRank vector (a walk that restarts at the node
    with probability ``alpha``). Two nodes with similar PPR vectors sit in similar
    parts of the graph. Stacking every node's PPR vector gives an ``n x n`` proximity
    matrix, and reducing it with an SVD yields compact embeddings that preserve this
    diffusion-based similarity -- closely related to spectral embeddings but weighted
    by the restart-controlled diffusion. ``alpha`` is the restart probability.
    """

    def __init__(self, dim=16, alpha=0.15):
        self.dim = dim
        self.alpha = alpha

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        P = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        # PPR matrix: row i is the personalised PageRank vector of node i
        ppr = self.alpha * np.linalg.inv(np.eye(n) - (1 - self.alpha) * P)
        U, s, _ = np.linalg.svd(ppr)
        self.embedding_ = U[:, :self.dim] * np.sqrt(s[:self.dim])
        return self

    def get_vector(self, node):
        return self.embedding_[node]


__all__ = ["PersonalizedPageRankEmbedding"]
