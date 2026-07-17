"""Node embeddings: turning graph nodes into vectors via random walks.

THE BIG IDEA
------------
word2vec learns word vectors by making a word predict its neighbours in a
sentence. DeepWalk and node2vec borrow the trick wholesale: generate many RANDOM
WALKS over the graph and treat each walk as a "sentence" of node-tokens, then run
word2vec. Nodes that keep co-occurring on walks -- because they are close and
well-connected -- end up with similar vectors. So graph structure becomes
geometry, and any vector-based ML (clustering, kNN, a classifier) then applies to
nodes.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


def _random_walk(adj, start, length, rng, p=1.0, q=1.0, prev=None):
    """One walk. With p=q=1 it is a uniform (DeepWalk) walk; otherwise the walk is
    BIASED (node2vec) by the return parameter p and in-out parameter q."""
    walk = [start]
    for _ in range(length - 1):
        cur = walk[-1]
        neigh = adj[cur]
        if len(neigh) == 0:
            break
        if prev is None or (p == 1.0 and q == 1.0):
            nxt = rng.choice(neigh)
        else:
            # node2vec bias: weight each candidate by whether it returns to prev
            # (1/p), stays local (1), or explores outward (1/q)
            weights = []
            prev_neigh = set(adj[prev].tolist())
            for nb in neigh:
                if nb == prev:
                    weights.append(1.0 / p)          # return
                elif nb in prev_neigh:
                    weights.append(1.0)              # stay in the neighbourhood
                else:
                    weights.append(1.0 / q)          # explore
            weights = np.array(weights)
            nxt = rng.choice(neigh, p=weights / weights.sum())
        prev = cur
        walk.append(nxt)
    return walk


class DeepWalk(BaseEstimator):
    """Node embeddings from UNIFORM random walks + skip-gram.

    THE ALGORITHM
    -------------
    From each node, generate ``n_walks`` uniform random walks of length
    ``walk_length``. Feed the walks -- sequences of node ids -- to word2vec as if
    they were sentences. The skip-gram objective makes a node predict the nodes
    that appear near it on walks, so co-occurring (structurally close) nodes get
    similar vectors.

    Uniform walks mean DeepWalk captures a blend of local and global proximity
    with no knobs beyond walk length and window. It was the first to show that the
    word2vec machinery transfers directly to graphs, which is the insight
    node2vec then made tunable.

    Perozzi, Al-Rfou & Skiena (2014).
    """

    def __init__(self, n_dim=64, walk_length=40, n_walks=10, window=5,
                 n_epochs=5, random_state=None):
        self.n_dim = n_dim
        self.walk_length = walk_length
        self.n_walks = n_walks
        self.window = window
        self.n_epochs = n_epochs
        self.random_state = random_state
        self._p = self._q = 1.0

    def fit(self, A):
        from ..embed import Word2Vec
        A = np.asarray(A, float)
        rng = check_random_state(self.random_state)
        adj = [np.where(A[i] > 0)[0] for i in range(len(A))]

        # the corpus of walks, node ids stringified as word2vec tokens
        walks = []
        for _ in range(self.n_walks):
            for start in rng.permutation(len(A)):
                walk = _random_walk(adj, start, self.walk_length, rng,
                                    self._p, self._q, prev=None)
                walks.append([str(v) for v in walk])

        self._w2v = Word2Vec(n_dim=self.n_dim, window=self.window,
                             n_epochs=self.n_epochs, min_count=1,
                             random_state=rng).fit(walks)
        # assemble a node -> vector matrix
        self.embeddings_ = np.zeros((len(A), self.n_dim))
        for i in range(len(A)):
            if str(i) in self._w2v.vocab_:
                self.embeddings_[i] = self._w2v.get_vector(str(i))
        return self

    def similarity(self, i, j):
        a, b = self.embeddings_[i], self.embeddings_[j]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Node2Vec(DeepWalk):
    """DeepWalk with BIASED walks -- tune local vs global exploration.

    THE TWO KNOBS
    -------------
    node2vec adds a return parameter ``p`` and an in-out parameter ``q`` that bias
    each step of the walk:

    * small ``q`` (large 1/q) pushes the walk OUTWARD, exploring far -- a
      breadth-first flavour that captures STRUCTURAL roles ("this node is a hub,
      like that distant node is a hub");
    * small ``p`` keeps the walk LOCAL, revisiting -- a depth-first flavour that
      captures COMMUNITY ("these nodes belong to the same tightly-knit group").

    So the same random-walk-plus-word2vec recipe as DeepWalk, but the two knobs
    let you dial what "similar" MEANS -- same community versus same structural
    role. That tunability is node2vec's whole contribution, and why it usually
    beats plain DeepWalk when you know which kind of similarity the task wants.

    Grover & Leskovec (2016).
    """

    def __init__(self, n_dim=64, walk_length=40, n_walks=10, window=5,
                 n_epochs=5, p=1.0, q=1.0, random_state=None):
        super().__init__(n_dim, walk_length, n_walks, window, n_epochs,
                         random_state)
        self._p = p
        self._q = q


__all__ = ["DeepWalk", "Node2Vec"]
