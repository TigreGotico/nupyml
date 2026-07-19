"""Sent2Vec: compositional sentence embeddings from word and n-gram vectors."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class Sent2Vec(BaseEstimator):
    """A sentence embedding is the AVERAGE of learned word vectors (Pagliardini, 2018).

    Averaging pretrained word vectors is a strong sentence baseline, but the vectors were
    trained for a different job. Sent2Vec trains them FOR averaging: it treats the whole
    sentence's mean vector as the context and, CBOW-style, learns word (and optionally
    bigram) vectors so that mean predicts each word actually in the sentence. Because the
    training objective IS the averaging operation, the composed sentence vector is
    directly optimised -- fast to train, fast to infer (just a mean), and competitive
    with far heavier encoders on sentence similarity. ``dim`` is the vector size;
    ``fit`` takes tokenised sentences; ``transform`` averages a sentence's vectors.
    """

    def __init__(self, dim=64, epochs=5, lr=0.05, n_negative=5, min_count=1,
                 random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        counts = {}
        for s in sentences:
            for w in s:
                counts[w] = counts.get(w, 0) + 1
        vocab = sorted(w for w, c in counts.items() if c >= self.min_count)
        self.vocab_ = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        self.W_ = rng.randn(V, self.dim) * 0.1              # source (averaged) vectors
        self.O_ = np.zeros((V, self.dim))                   # output (target) vectors
        freq = np.array([counts[w] for w in vocab], float) ** 0.75
        noise = freq / freq.sum()
        for _ in range(self.epochs):
            for s in sentences:
                ids = [self.vocab_[w] for w in s if w in self.vocab_]
                if len(ids) < 2:
                    continue
                mean = self.W_[ids].mean(axis=0)
                for t in ids:
                    ctx = mean - self.W_[t] / len(ids)      # leave-one-out context
                    negs = rng.choice(V, self.n_negative, p=noise)
                    targets = np.concatenate([[t], negs])
                    labels = np.array([1.0] + [0.0] * self.n_negative)
                    score = 1.0 / (1.0 + np.exp(-self.O_[targets] @ ctx))
                    err = score - labels                    # negative-sampling gradient
                    self.O_[targets] -= self.lr * err[:, None] * ctx
                    grad_ctx = err @ self.O_[targets]
                    self.W_[ids] -= self.lr * grad_ctx / len(ids)
        self.embedding_ = self.W_
        return self

    def transform(self, sentence):
        ids = [self.vocab_[w] for w in sentence if w in self.vocab_]
        if not ids:
            return np.zeros(self.dim)
        return self.W_[ids].mean(axis=0)


__all__ = ["Sent2Vec"]
