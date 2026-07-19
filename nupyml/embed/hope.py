"""High-Order Proximity preserved Embedding, asymmetry and all (Ou et al., 2016)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class HOPE(BaseEstimator):
    """High-Order Proximity preserved Embedding, asymmetry and all (Ou et al., 2016).

    Many graphs are DIRECTED, and directed proximity is asymmetric -- a page links to
    another without being linked back. HOPE preserves a chosen high-order proximity
    measure (Katz: paths of every length, geometrically discounted) by giving each
    node a SOURCE and a TARGET embedding whose inner product approximates that
    proximity. Because the Katz matrix has a low-rank sparse form, a single
    generalised SVD yields both embeddings efficiently, capturing directional,
    multi-step structure that symmetric methods cannot. ``beta`` discounts longer
    paths.
    """

    def __init__(self, dim=16, beta=0.01):
        self.dim = dim
        self.beta = beta

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        # Katz proximity S = (I - beta A)^{-1} beta A -- paths of all lengths
        S = np.linalg.inv(np.eye(n) - self.beta * A) @ (self.beta * A)
        U, s, Vt = np.linalg.svd(S)
        scale = np.sqrt(s[:self.dim])
        self.source_ = U[:, :self.dim] * scale            # asymmetric source/target
        self.target_ = Vt[:self.dim].T * scale
        self.embedding_ = np.hstack([self.source_, self.target_])
        return self

    def get_vector(self, node):
        return self.embedding_[node]

    def proximity(self, i, j):
        return self.source_[i] @ self.target_[j]


__all__ = ["HOPE"]
