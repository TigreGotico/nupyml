"""Embeddings v4 / retrieval: spectral structural embeddings, the matrix-
factorisation view of random walks, hashed embeddings, and a navigable ANN index.

Four more representation/retrieval tools. GraphWave embeds a node's structural ROLE
from how heat diffuses around it. NetMF shows that DeepWalk/node2vec are implicitly
factorising a known matrix, and does it directly. Hash embeddings vectorise an
unbounded vocabulary into a fixed pool. HNSW is a graph index for fast approximate
nearest-neighbour search.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class GraphWave(BaseEstimator):
    """Embed a node's structural ROLE from heat diffusion (Donnat et al., 2018).

    Two nodes play the same structural role (both hubs, both bridge points) even if
    they are far apart -- proximity embeddings miss this. GraphWave treats each node
    as a heat source and lets heat DIFFUSE through the graph (a spectral graph
    wavelet); the resulting pattern of coefficients around a node is a fingerprint of
    its local structure. Summarising that pattern's distribution via its empirical
    CHARACTERISTIC FUNCTION (sampled at a few points) gives an embedding where
    structurally-equivalent nodes coincide, with no random walks or labels. ``scales``
    are the diffusion times.
    """

    def __init__(self, scales=(1.0,), n_samples=10, sample_max=2.0):
        self.scales = list(scales)
        self.n_samples = n_samples
        self.sample_max = sample_max

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        deg = A.sum(axis=1)
        L = np.diag(deg) - A                             # unnormalised Laplacian
        vals, vecs = np.linalg.eigh(L)
        ts = np.linspace(0, self.sample_max, self.n_samples)
        embeddings = []
        for s in self.scales:
            # spectral graph wavelet operator: U exp(-s Lambda) U^T
            Psi = (vecs * np.exp(-s * vals)) @ vecs.T
            emb = np.zeros((n, 2 * self.n_samples))
            for node in range(n):
                coeffs = Psi[:, node]                    # wavelet centred at node
                # empirical characteristic function phi(t) = mean exp(i t coeff)
                for k, t in enumerate(ts):
                    emb[node, 2 * k] = np.mean(np.cos(t * coeffs))
                    emb[node, 2 * k + 1] = np.mean(np.sin(t * coeffs))
            embeddings.append(emb)
        self.embedding_ = np.hstack(embeddings)
        return self

    def get_vector(self, node):
        return self.embedding_[node]


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


class HashEmbedding(BaseEstimator):
    """Embed an UNBOUNDED vocabulary into a fixed pool (Tito Svenstrup, 2017).

    A normal embedding table needs one row per token and a vocabulary you must fix
    in advance. Hash embeddings drop both: they keep a small shared POOL of vectors
    and map each token, via several hash functions, to a few pool slots, combining
    them by per-token importance weights (also hashed). Any token -- including ones
    never seen -- gets a vector, memory is fixed regardless of vocabulary size, and
    hash collisions are absorbed because each token uses a DIFFERENT combination of
    slots. ``pool_size`` rows serve an arbitrary number of tokens.
    """

    def __init__(self, pool_size=64, dim=16, n_hashes=2, random_state=0):
        self.pool_size = pool_size
        self.dim = dim
        self.n_hashes = n_hashes
        self.random_state = random_state

    def fit(self, tokens=None):
        rng = check_random_state(self.random_state)
        self.pool_ = rng.randn(self.pool_size, self.dim) * 0.1
        return self

    def _hashes(self, token):
        return [hash((h, str(token))) % self.pool_size for h in range(self.n_hashes)]

    def _weights(self, token):
        w = np.array([((hash((self.n_hashes + h, str(token))) % 1000) / 1000.0)
                      for h in range(self.n_hashes)])
        return w / (w.sum() + 1e-9)

    def get_vector(self, token):
        slots = self._hashes(token)
        w = self._weights(token)
        return sum(wi * self.pool_[s] for wi, s in zip(w, slots))

    def transform(self, tokens):
        if not hasattr(self, "pool_"):
            self.fit()
        return np.array([self.get_vector(t) for t in tokens])


class HNSW(BaseEstimator):
    """A navigable graph index for fast approximate nearest neighbours
    (Malkov & Yashunin, 2018).

    Brute-force nearest-neighbour search scans every point. HNSW builds a
    multi-LAYER proximity graph: the top layers are sparse "express lanes" of a few
    long-range links, the bottom layer connects every point to its near neighbours.
    A query enters at the top, greedily hops toward closer nodes, then descends layer
    by layer -- so it reaches a query's neighbourhood in roughly LOGARITHMIC hops
    instead of scanning everything. It is the index behind most modern vector
    databases. ``M`` neighbours per node, ``ef`` search breadth.
    """

    def __init__(self, M=8, ef=32, random_state=None):
        self.M = M
        self.ef = ef
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        self.X_ = X
        rng = check_random_state(self.random_state)
        n = len(X)
        mL = 1.0 / np.log(self.M)
        self.levels_ = (-np.log(rng.rand(n)) * mL).astype(int)
        max_level = self.levels_.max()
        self.graph_ = [dict() for _ in range(max_level + 1)]   # per-layer adjacency
        self.entry_ = int(np.argmax(self.levels_))
        order = rng.permutation(n)
        inserted = []
        for i in order:
            for lvl in range(self.levels_[i] + 1):
                if not inserted:
                    self.graph_[lvl][i] = []
                    continue
                cand = [j for j in inserted if self.levels_[j] >= lvl]
                if not cand:
                    self.graph_[lvl][i] = []
                    continue
                d = np.linalg.norm(X[cand] - X[i], axis=1)
                nn = [cand[k] for k in np.argsort(d)[:self.M]]
                self.graph_[lvl][i] = nn
                for j in nn:                             # bidirectional links
                    self.graph_[lvl].setdefault(j, [])
                    if i not in self.graph_[lvl][j]:
                        self.graph_[lvl][j].append(i)
            inserted.append(i)
        return self

    def _search_layer(self, q, entry, layer, ef):
        visited = {entry}
        cand = [(np.linalg.norm(self.X_[entry] - q), entry)]
        best = list(cand)
        while cand:
            cand.sort()
            dist, node = cand.pop(0)
            if dist > max(b[0] for b in best) and len(best) >= ef:
                break
            for nb in self.graph_[layer].get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    dn = np.linalg.norm(self.X_[nb] - q)
                    cand.append((dn, nb)); best.append((dn, nb))
                    best.sort(); best = best[:ef]
        return best

    def query(self, q, k=5):
        q = np.asarray(q, float).ravel()
        entry = self.entry_
        for layer in range(len(self.graph_) - 1, 0, -1):     # descend express lanes
            entry = self._search_layer(q, entry, layer, 1)[0][1]
        best = self._search_layer(q, entry, 0, self.ef)
        return np.array([n for _, n in sorted(best)[:k]])


__all__ = ["GraphWave", "NetMF", "HashEmbedding", "HNSW"]
