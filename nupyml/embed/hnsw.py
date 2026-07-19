"""A navigable graph index for fast approximate nearest neighbours"""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


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


__all__ = ["HNSW"]
