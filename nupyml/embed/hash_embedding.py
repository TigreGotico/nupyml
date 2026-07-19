"""Embed an UNBOUNDED vocabulary into a fixed pool (Tito Svenstrup, 2017)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


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


__all__ = ["HashEmbedding"]
