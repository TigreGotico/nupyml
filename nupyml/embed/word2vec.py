"""Word2Vec (skip-gram with negative sampling) and GloVe.

Both take a corpus as a list of token lists (already tokenized sentences) and
learn a dense vector per word.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class Word2Vec(BaseEstimator):
    """Skip-gram with negative sampling: predict a word's neighbours.

    THE TASK
    --------
    Slide a window over the text. For each CENTRE word, the training signal is:
    its actual context words should score HIGH, and random words should score LOW.
    That is it -- no reconstruction, no labels, just "these words really co-occur,
    those do not".

    WHY NEGATIVE SAMPLING
    ---------------------
    The proper skip-gram objective is a softmax over the ENTIRE vocabulary at
    every step -- millions of words, hopelessly slow. Negative sampling replaces
    it with a cheap binary problem: distinguish each true context word from a
    handful of ``negative`` random words drawn from the vocabulary. A
    vocabulary-sized softmax becomes a few logistic-regression updates per step,
    and that single approximation is what made word2vec fast enough to train on
    billions of words.

    The negatives are drawn from a UNIGRAM^0.75 distribution -- frequency raised
    to the 3/4 power. That exponent (an empirical choice that stuck) damps the
    dominance of ultra-common words like "the" as negatives while still sampling
    them more than rare words, and it measurably improves the vectors.

    TWO VECTORS PER WORD
    --------------------
    Each word has an "input" vector (when it is the centre) and an "output" vector
    (when it is context). The learned embedding is the input vector; the output
    vectors are scaffolding, discarded after training -- the same throw-away-half
    pattern as the SimCLR projection head.

    Mikolov et al. (2013).
    """

    def __init__(self, n_dim=50, window=2, negative=5, learning_rate=0.025,
                 n_epochs=5, min_count=1, random_state=None):
        self.n_dim = n_dim
        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.min_count = min_count
        self.random_state = random_state

    def _build_vocab(self, sentences):
        counts = {}
        for sent in sentences:
            for w in sent:
                counts[w] = counts.get(w, 0) + 1
        self.vocab_ = {w: i for i, w in enumerate(
            w for w, c in counts.items() if c >= self.min_count)}
        self.index_to_word_ = {i: w for w, i in self.vocab_.items()}
        self.counts_ = np.array([counts[self.index_to_word_[i]]
                                 for i in range(len(self.vocab_))], dtype=float)
        # the unigram^0.75 sampling distribution for negatives
        self._neg_dist = self.counts_ ** 0.75
        self._neg_dist /= self._neg_dist.sum()

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        self._build_vocab(sentences)
        V = len(self.vocab_)
        # input (centre) and output (context) vectors -- see the class docstring
        self.W_in = (rng.uniform(-0.5, 0.5, size=(V, self.n_dim)) / self.n_dim)
        self.W_out = np.zeros((V, self.n_dim))
        lr = self.learning_rate

        indexed = [[self.vocab_[w] for w in sent if w in self.vocab_]
                   for sent in sentences]
        for epoch in range(self.n_epochs):
            for sent in indexed:
                for pos, centre in enumerate(sent):
                    lo = max(0, pos - self.window)
                    hi = min(len(sent), pos + self.window + 1)
                    for ctx_pos in range(lo, hi):
                        if ctx_pos == pos:
                            continue
                        context = sent[ctx_pos]
                        self._update(centre, context, rng, lr)
        return self

    def _update(self, centre, context, rng, lr):
        """One skip-gram-negative-sampling step for a (centre, context) pair.

        The positive context is pushed toward the centre; ``negative`` random
        words are pushed away. Each is a single logistic update -- cheap, and the
        whole reason the method scales.
        """
        v = self.W_in[centre]
        # the positive pair plus `negative` sampled negatives, with their labels
        targets = [(context, 1.0)]
        negs = rng.choice(len(self.vocab_), size=self.negative, p=self._neg_dist)
        targets += [(int(n), 0.0) for n in negs]

        grad_v = np.zeros_like(v)
        for target, label in targets:
            u = self.W_out[target]
            score = 1.0 / (1.0 + np.exp(-np.clip(v @ u, -30, 30)))
            g = (score - label) * lr          # logistic gradient
            grad_v += g * u
            self.W_out[target] -= g * v
        self.W_in[centre] -= grad_v

    def get_vector(self, word):
        return self.W_in[self.vocab_[word]]

    def most_similar(self, word, topn=5):
        """Nearest words by cosine similarity -- the qualitative test of meaning."""
        v = self.get_vector(word)
        norms = np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(v) + 1e-12
        sims = (self.W_in @ v) / norms
        order = np.argsort(sims)[::-1]
        out = [(self.index_to_word_[i], float(sims[i])) for i in order
               if self.index_to_word_[i] != word][:topn]
        return out

    def analogy(self, a, b, c, topn=3):
        """a is to b as c is to ? -- vector arithmetic on meaning.

        Compute ``vec(b) - vec(a) + vec(c)`` and return the nearest words. This is
        the famous ``king - man + woman ~ queen``: the offset ``b - a`` captures a
        RELATION (here, gender), and adding it to ``c`` applies that relation.
        """
        target = (self.get_vector(b) - self.get_vector(a) + self.get_vector(c))
        norms = np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(target) + 1e-12
        sims = (self.W_in @ target) / norms
        seen = {a, b, c}
        order = np.argsort(sims)[::-1]
        return [(self.index_to_word_[i], float(sims[i])) for i in order
                if self.index_to_word_[i] not in seen][:topn]


class GloVe(BaseEstimator):
    """Global Vectors: factor the log co-occurrence matrix.

    THE COUNT-BASED ROUTE
    ---------------------
    Where word2vec streams local windows, GloVe first tallies the GLOBAL
    co-occurrence matrix ``X_ij`` (how often word j appears near word i across the
    whole corpus), then learns vectors so that::

        w_i . w_j + b_i + b_j  ~  log(X_ij)

    The insight is that RATIOS of co-occurrence probabilities carry meaning (ice
    co-occurs with "solid" far more than steam does), and fitting log-counts with
    a dot product makes those ratios linear in the vector space -- which is what
    gives the analogy arithmetic.

    THE WEIGHTING FUNCTION
    ----------------------
    Rare co-occurrences are noisy and common ones ("the", "of") are
    uninformative, so each squared error is weighted by ``f(X_ij)`` that rises
    with the count but CAPS at a maximum. Without it the fit is dominated by a
    handful of hyper-frequent pairs and learns little; that weighting is the piece
    that makes the global factorisation work as well as the local predictor.

    Pennington, Socher & Manning (2014).
    """

    def __init__(self, n_dim=50, window=5, x_max=100, alpha=0.75,
                 learning_rate=0.05, n_epochs=50, min_count=1, random_state=None):
        self.n_dim = n_dim
        self.window = window
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        counts = {}
        for sent in sentences:
            for w in sent:
                counts[w] = counts.get(w, 0) + 1
        self.vocab_ = {w: i for i, w in enumerate(
            w for w, c in counts.items() if c >= self.min_count)}
        self.index_to_word_ = {i: w for w, i in self.vocab_.items()}
        V = len(self.vocab_)

        # build the co-occurrence matrix, weighting closer words more (1/distance)
        cooc = {}
        for sent in sentences:
            idx = [self.vocab_[w] for w in sent if w in self.vocab_]
            for pos, i in enumerate(idx):
                lo = max(0, pos - self.window)
                for jpos in range(lo, pos):
                    j = idx[jpos]
                    d = pos - jpos
                    cooc[(i, j)] = cooc.get((i, j), 0) + 1.0 / d
                    cooc[(j, i)] = cooc.get((j, i), 0) + 1.0 / d

        self.W_ = rng.uniform(-0.5, 0.5, size=(V, self.n_dim)) / self.n_dim
        self.W_ctx_ = rng.uniform(-0.5, 0.5, size=(V, self.n_dim)) / self.n_dim
        self.b_ = np.zeros(V)
        self.b_ctx_ = np.zeros(V)
        pairs = list(cooc.items())
        lr = self.learning_rate

        for epoch in range(self.n_epochs):
            rng.shuffle(pairs)
            for (i, j), x in pairs:
                # the capped weighting: rare pairs count less, common ones do not
                # dominate
                weight = (x / self.x_max) ** self.alpha if x < self.x_max else 1.0
                diff = (self.W_[i] @ self.W_ctx_[j] + self.b_[i]
                        + self.b_ctx_[j] - np.log(x))
                g = weight * diff * lr
                wi = self.W_[i].copy()
                self.W_[i] -= g * self.W_ctx_[j]
                self.W_ctx_[j] -= g * wi
                self.b_[i] -= g
                self.b_ctx_[j] -= g
        # the final embedding sums both vectors, which is more robust than either
        self.embeddings_ = self.W_ + self.W_ctx_
        return self

    def get_vector(self, word):
        return self.embeddings_[self.vocab_[word]]

    def most_similar(self, word, topn=5):
        v = self.get_vector(word)
        norms = np.linalg.norm(self.embeddings_, axis=1) * np.linalg.norm(v) + 1e-12
        sims = (self.embeddings_ @ v) / norms
        order = np.argsort(sims)[::-1]
        return [(self.index_to_word_[i], float(sims[i])) for i in order
                if self.index_to_word_[i] != word][:topn]


__all__ = ["Word2Vec", "GloVe"]
