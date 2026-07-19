"""Embed nodes by their structural ROLE, not their location (Ribeiro, 2017)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from . import Word2Vec


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


__all__ = ["struc2vec"]
