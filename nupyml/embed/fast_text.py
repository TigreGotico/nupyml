"""Word vectors from character n-grams -- OOV-robust (Bojanowski et al., 2017)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FastText(BaseEstimator):
    """Word vectors from character n-grams -- OOV-robust (Bojanowski et al., 2017).

    word2vec has one vector per word and is helpless on a word it never saw.
    FastText represents each word as the sum of its CHARACTER n-gram vectors (plus
    a whole-word vector), and trains those subword vectors with skip-gram. So an
    unseen or misspelled word still gets a sensible vector from its subwords, and
    morphologically related words ("run", "running", "runner") share n-grams and
    end up close -- a large win for rare words and rich morphology. Subwords are
    hashed into ``bucket`` buckets to bound memory.
    """

    def __init__(self, dim=32, min_n=3, max_n=5, bucket=2000, epochs=20, lr=0.025,
                 n_negative=5, window=2, random_state=None):
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.window = window
        self.random_state = random_state

    def _subwords(self, word):
        w = "<" + word + ">"
        grams = [w]
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(w) - n + 1):
                grams.append(w[i:i + n])
        return [hash(g) % self.bucket for g in grams]

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        vocab = sorted({w for s in sentences for w in s})
        self.vidx_ = {w: k for k, w in enumerate(vocab)}
        self.vocab_ = vocab
        self.sub_ = rng.normal(0, 0.1, (self.bucket, self.dim))   # subword vectors
        self.ctx_ = rng.normal(0, 0.1, (len(vocab), self.dim))    # context vectors
        freq = np.ones(len(vocab))
        neg_p = freq / freq.sum()
        for _ in range(self.epochs):
            for sent in sentences:
                for i, w in enumerate(sent):
                    v = self._word_vec(w)
                    lo, hi = max(0, i - self.window), min(len(sent), i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        c = self.vidx_[sent[j]]
                        negs = rng.choice(len(vocab), self.n_negative, p=neg_p)
                        self._step(w, v, c, negs)
        return self

    def _word_vec(self, word):
        return self.sub_[self._subwords(word)].sum(axis=0)

    def _step(self, word, v, c, negs):
        subs = self._subwords(word)
        grad = (_sigmoid(v @ self.ctx_[c]) - 1.0) * self.ctx_[c]
        self.ctx_[c] -= self.lr * (_sigmoid(v @ self.ctx_[c]) - 1.0) * v
        for k in negs:
            gk = _sigmoid(v @ self.ctx_[k])
            grad += gk * self.ctx_[k]
            self.ctx_[k] -= self.lr * gk * v
        for s in subs:                                 # spread gradient over subwords
            self.sub_[s] -= self.lr * grad / len(subs)

    def get_vector(self, word):
        """Works for ANY word (in-vocab or not) via its character n-grams."""
        return self._word_vec(word)


__all__ = ["FastText"]
