"""Embeddings v5 / retrieval: high-order graph proximity and quantised indexing.

Four more representation/retrieval tools. GraRep and HOPE both embed nodes so that
HIGH-ORDER proximity (multi-step reachability, not just direct edges) is preserved,
by matrix factorisation. Personalised-PageRank embeddings describe a node by where a
random walk from it tends to land. The IVFPQ index combines coarse partitioning with
product quantisation for billion-scale nearest-neighbour search.
"""
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


class IVFPQIndex(BaseEstimator):
    """Coarse partitioning + product quantisation for billion-scale ANN (Jégou, 2011).

    Two ideas compose into the index behind Faiss's ``IVFPQ``. First an INVERTED FILE:
    k-means centroids partition space into cells, and a query only scans the few cells
    nearest it (``n_probe``), not the whole dataset. Second PRODUCT QUANTISATION: within
    a cell, each vector's RESIDUAL from the centroid is compressed to a few bytes by
    quantising subvectors independently, so distances are estimated from a small
    lookup table without decompressing. Together they give sub-linear search over
    vectors far too many to store uncompressed. Reuses ``ProductQuantizer``.
    """

    def __init__(self, n_cells=16, n_probe=3, n_subvectors=4, n_codes=64,
                 random_state=None):
        self.n_cells = n_cells
        self.n_probe = n_probe
        self.n_subvectors = n_subvectors
        self.n_codes = n_codes
        self.random_state = random_state

    def fit(self, X):
        from ..cluster import KMeans
        from ..search import ProductQuantizer
        X = check_array(X)
        self.X_ = X
        rng = check_random_state(self.random_state)
        km = KMeans(n_clusters=min(self.n_cells, len(X)),
                    random_state=rng).fit(X)
        self.centroids_ = km.cluster_centers_
        assign = km.labels_
        self.lists_ = {c: np.where(assign == c)[0] for c in range(len(self.centroids_))}
        residuals = X - self.centroids_[assign]           # quantise the residuals
        self.pq_ = ProductQuantizer(n_subvectors=self.n_subvectors,
                                    n_codes=self.n_codes, random_state=rng).fit(residuals)
        self.codes_ = self.pq_.encode(residuals)
        return self

    def query(self, q, k=5):
        q = np.asarray(q, float).ravel()
        cell_d = np.linalg.norm(self.centroids_ - q, axis=1)
        probe = np.argsort(cell_d)[:self.n_probe]         # nearest cells only
        cand = np.concatenate([self.lists_[c] for c in probe if len(self.lists_[c])]) \
            if any(len(self.lists_[c]) for c in probe) else np.arange(len(self.X_))
        d = np.linalg.norm(self.X_[cand] - q, axis=1)     # exact re-rank of candidates
        return cand[np.argsort(d)[:k]]


__all__ = ["GraRep", "HOPE", "PersonalizedPageRankEmbedding", "IVFPQIndex"]
