"""Find near neighbours WITHOUT comparing to everything (Indyk & Motwani, 1998)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


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


__all__ = ["LSHIndex"]
