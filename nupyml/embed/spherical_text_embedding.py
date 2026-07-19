"""Spherical text embeddings: directional word vectors on the unit sphere (von Mises-Fisher)."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class SphericalTextEmbedding(BaseEstimator):
    """Train word vectors on the SPHERE, where cosine is the native metric (Meng et al., 2019).

    Word2vec trains in Euclidean space with a dot-product objective, but everyone then
    compares vectors with COSINE similarity -- a mismatch between training and use. This
    trains embeddings directly on the unit sphere: every vector is kept L2-normalised and
    the model, under a von Mises-Fisher view (the Gaussian's analogue on a sphere), pulls
    a word's DIRECTION toward its context's direction and pushes negatives away. Because
    training and inference share the spherical geometry, the resulting cosine similarities
    are better calibrated for clustering and retrieval. Vectors are renormalised after
    each step. ``fit`` takes tokenised sentences; ``similarity`` gives cosine similarity.
    """

    def __init__(self, dim=64, window=5, epochs=5, lr=0.1, n_negative=5, min_count=1,
                 random_state=None):
        self.dim = dim
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.min_count = min_count
        self.random_state = random_state

    def _normalize(self, M):
        return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        counts = {}
        for s in sentences:
            for w in s:
                counts[w] = counts.get(w, 0) + 1
        vocab = sorted(w for w, c in counts.items() if c >= self.min_count)
        self.vocab_ = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        self.W_ = self._normalize(rng.randn(V, self.dim))
        self.C_ = self._normalize(rng.randn(V, self.dim))
        freq = np.array([counts[w] for w in vocab], float) ** 0.75
        noise = freq / freq.sum()
        for _ in range(self.epochs):
            for s in sentences:
                ids = [self.vocab_[w] for w in s if w in self.vocab_]
                for i, t in enumerate(ids):
                    lo = max(0, i - self.window)
                    for j in range(lo, min(len(ids), i + self.window + 1)):
                        if j == i:
                            continue
                        c = ids[j]
                        negs = rng.choice(V, self.n_negative, p=noise)
                        # vMF-style: raise cos(target, context), lower cos(negs, context)
                        self.W_[t] += self.lr * self.C_[c]
                        self.C_[c] += self.lr * self.W_[t]
                        for nn in negs:
                            self.W_[t] -= self.lr * self.C_[nn] \
                                * max(0.0, self.W_[t] @ self.C_[nn])
                        self.W_[t] = self._normalize(self.W_[t])   # back onto the sphere
                        self.C_[c] = self._normalize(self.C_[c])
        self.embedding_ = self.W_
        return self

    def similarity(self, a, b):
        return float(self.W_[self.vocab_[a]] @ self.W_[self.vocab_[b]])   # unit -> cosine


__all__ = ["SphericalTextEmbedding"]
