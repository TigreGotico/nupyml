"""Embed a HETEROGENEOUS graph by walking typed paths (Dong et al., 2017)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from . import Word2Vec


def _walks_to_embeddings(walks, n_dim, window, epochs, rng_state):
    w2v = Word2Vec(n_dim=n_dim, window=window, n_epochs=epochs,
                   random_state=rng_state)
    w2v.fit([[str(n) for n in walk] for walk in walks])
    return w2v


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


__all__ = ["metapath2vec"]
