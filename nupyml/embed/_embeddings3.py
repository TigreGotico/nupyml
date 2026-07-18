"""Embeddings v3: approximate nearest neighbours, the hashing trick, and two graph
embeddings that go beyond proximity.

``LSHIndex`` and ``feature_hashing`` are about SCALE -- finding neighbours and
vectorising features without ever materialising the full space. ``struc2vec`` and
``metapath2vec`` are about MEANING: struc2vec embeds nodes by their structural ROLE
(a hub is like a hub, wherever it sits), and metapath2vec embeds a HETEROGENEOUS
graph by walking typed paths. Both reuse ``Word2Vec`` for the skip-gram step.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from . import Word2Vec


class LSHIndex(BaseEstimator):
    """Find near neighbours WITHOUT comparing to everything (Indyk & Motwani, 1998).

    Exact nearest-neighbour search compares a query to every point -- hopeless at
    scale. Locality-sensitive hashing uses hash functions that, unlike normal ones,
    put SIMILAR points in the same bucket ON PURPOSE. For cosine similarity the hash
    is the sign of a random projection: nearby directions almost always agree.
    Concatenate several such bits into a key (few collisions), keep several
    independent tables (high recall), and a query only ever compares against the
    handful of points sharing a bucket. Sub-linear query time for a small, tunable
    hit to recall.
    """

    def __init__(self, n_bits=12, n_tables=8, random_state=None):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self.X_ = X
        d = X.shape[1]
        self.planes_ = [rng.randn(d, self.n_bits) for _ in range(self.n_tables)]
        self.tables_ = []
        for planes in self.planes_:
            keys = (X @ planes > 0)                       # sign of random projections
            table = {}
            for i, k in enumerate(map(tuple, keys)):
                table.setdefault(k, []).append(i)
            self.tables_.append(table)
        return self

    def _candidates(self, q):
        cand = set()
        for planes, table in zip(self.planes_, self.tables_):
            key = tuple(q @ planes > 0)
            cand.update(table.get(key, ()))              # only same-bucket points
        return np.fromiter(cand, int) if cand else np.arange(len(self.X_))

    def query(self, q, k=5):
        q = np.asarray(q, float).ravel()
        cand = self._candidates(q)
        sub = self.X_[cand]
        sims = sub @ q / (np.linalg.norm(sub, axis=1) * np.linalg.norm(q) + 1e-12)
        order = np.argsort(sims)[::-1][:k]
        return cand[order]


def feature_hashing(records, n_features=256, seed=0):
    """Vectorise arbitrary features into a fixed size by HASHING (Weinberger, 2009).

    A bag of raw features (words, categories, id strings) can have millions of
    distinct keys, and a normal one-hot needs a growing vocabulary you must store
    and keep in sync. The hashing trick drops the vocabulary: hash each feature key
    to one of ``n_features`` columns and add its value there, with a second hash
    giving a SIGN so that random collisions tend to cancel rather than accumulate.
    Fixed memory, no fitting, streaming-friendly -- at the cost of occasional
    collisions. ``records`` is a list of dicts or of token lists.
    """
    out = np.zeros((len(records), n_features))
    for i, rec in enumerate(records):
        items = rec.items() if isinstance(rec, dict) else ((tok, 1.0) for tok in rec)
        for key, val in items:
            h = hash((seed, str(key)))
            col = h % n_features
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0    # sign hash cancels collisions
            out[i, col] += sign * val
    return out


def _walks_to_embeddings(walks, n_dim, window, epochs, rng_state):
    w2v = Word2Vec(n_dim=n_dim, window=window, n_epochs=epochs,
                   random_state=rng_state)
    w2v.fit([[str(n) for n in walk] for walk in walks])
    return w2v


class struc2vec(BaseEstimator):
    """Embed nodes by their structural ROLE, not their location (Ribeiro, 2017).

    node2vec/DeepWalk place a node near the neighbours it co-occurs with, so two
    hubs at opposite ends of a graph -- structurally identical but never adjacent --
    get unrelated embeddings. struc2vec fixes this: it measures STRUCTURAL
    similarity by comparing nodes' sorted degree sequences at growing hop distances,
    builds a similarity graph on that, and runs biased walks + skip-gram over IT --
    so nodes with the same role land together regardless of where they sit. This is
    a compact single-layer version using degree-sequence distance.
    """

    def __init__(self, n_dim=16, n_walks=20, walk_length=20, window=5, n_hops=2,
                 epochs=5, random_state=None):
        self.n_dim = n_dim
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.window = window
        self.n_hops = n_hops
        self.epochs = epochs
        self.random_state = random_state

    def _degree_signature(self, A):
        n = len(A)
        deg = A.sum(axis=1)
        # for each node, the sorted degree sequence of its k-hop neighbourhood
        sigs = []
        reach = A.astype(bool)
        rings = [reach.copy()]
        power = reach.copy()
        for _ in range(self.n_hops - 1):
            power = (power @ reach) > 0
            rings.append(power)
        for u in range(n):
            sig = []
            for ring in rings:
                nb = np.where(ring[u])[0]
                sig.append(tuple(sorted(deg[nb].astype(int))) if len(nb) else ())
            sigs.append(sig)
        return sigs, deg

    def _struct_dist(self, s1, s2):
        d = 0.0
        for a, b in zip(s1, s2):                          # compare per-hop degree seqs
            la, lb = list(a), list(b)
            m = max(len(la), len(lb), 1)
            la += [0] * (m - len(la)); lb += [0] * (m - len(lb))
            d += np.mean(np.abs(np.array(sorted(la)) - np.array(sorted(lb))))
        return d

    def fit(self, adjacency):
        A = check_array(adjacency)
        rng = check_random_state(self.random_state)
        n = len(A)
        sigs, _ = self._degree_signature(A)
        # structural similarity graph: weight = exp(-structural distance)
        W = np.zeros((n, n))
        for u in range(n):
            for v in range(u + 1, n):
                w = np.exp(-self._struct_dist(sigs[u], sigs[v]))
                W[u, v] = W[v, u] = w
        P = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        walks = []
        for _ in range(self.n_walks):
            for start in range(n):
                walk = [start]
                for _ in range(self.walk_length - 1):
                    walk.append(rng.choice(n, p=P[walk[-1]]))
                walks.append(walk)
        self.model_ = _walks_to_embeddings(walks, self.n_dim, self.window,
                                           self.epochs, rng)
        self.node_ids_ = list(range(n))
        return self

    def get_vector(self, node):
        return self.model_.get_vector(str(node))


class metapath2vec(BaseEstimator):
    """Embed a HETEROGENEOUS graph by walking typed paths (Dong et al., 2017).

    Real graphs have several node TYPES -- authors, papers, venues -- and a plain
    random walk mixes them incoherently. metapath2vec constrains the walk to follow
    a META-PATH, a repeating type schema like Author-Paper-Author, so each walk
    stays semantically meaningful, then feeds the walks to skip-gram. The result
    embeds different types into one space where a meaningful relation (co-authorship
    via shared papers) becomes proximity. ``node_types`` maps node -> type;
    ``metapath`` is the type sequence to repeat.
    """

    def __init__(self, metapath, n_dim=16, n_walks=20, walk_length=40, window=5,
                 epochs=5, random_state=None):
        self.metapath = metapath
        self.n_dim = n_dim
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.window = window
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, adjacency, node_types):
        A = check_array(adjacency)
        rng = check_random_state(self.random_state)
        n = len(A)
        types = np.asarray(node_types)
        neighbors = [np.where(A[u] > 0)[0] for u in range(n)]
        walks = []
        mp = self.metapath
        for _ in range(self.n_walks):
            for start in range(n):
                if types[start] != mp[0]:
                    continue
                walk = [start]
                step = 1
                for _ in range(self.walk_length - 1):
                    want = mp[step % len(mp)]             # next required type
                    cur = walk[-1]
                    cands = [v for v in neighbors[cur] if types[v] == want]
                    if not cands:
                        break
                    walk.append(int(rng.choice(cands)))
                    step += 1
                if len(walk) > 1:
                    walks.append(walk)
        self.model_ = _walks_to_embeddings(walks, self.n_dim, self.window,
                                           self.epochs, rng)
        return self

    def get_vector(self, node):
        return self.model_.get_vector(str(node))


__all__ = ["LSHIndex", "feature_hashing", "struc2vec", "metapath2vec"]
