"""The n-gram smoothing that counts CONTEXTS, not occurrences (Kneser & Ney)."""
import re
from collections import defaultdict, Counter
import numpy as np
from ..base import BaseEstimator


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


class KneserNeyLM(BaseEstimator):
    """The n-gram smoothing that counts CONTEXTS, not occurrences (Kneser & Ney).

    Add-one smoothing wastes probability uniformly; Kneser-Ney asks a sharper
    question. The unigram fallback is not "how often is 'Francisco'?" (often -- it
    always follows 'San') but "how many DIFFERENT words does it follow?" (almost
    none). That CONTINUATION probability is what should back off a rare bigram, and
    it is why Kneser-Ney is the strongest n-gram smoothing. Absolute discounting
    subtracts a fixed ``discount`` from each seen count and redistributes the freed
    mass by continuation probability. Interpolated bigram model here.
    """

    def __init__(self, discount=0.75):
        self.discount = discount

    def fit(self, sentences):
        self.bigrams_ = defaultdict(Counter)
        self.unigrams_ = Counter()
        self.continuations_ = defaultdict(set)      # word -> set of preceding words
        self.followers_ = defaultdict(set)          # word -> set of following words
        for sent in sentences:
            toks = sent if isinstance(sent, list) else _tokenize(sent)
            for w1, w2 in zip(toks, toks[1:]):
                self.bigrams_[w1][w2] += 1
                self.followers_[w1].add(w2)
                self.continuations_[w2].add(w1)
            self.unigrams_.update(toks)
        self.vocab_ = set(self.unigrams_)
        self.n_bigram_types_ = sum(len(v) for v in self.bigrams_.values())
        return self

    def _p_continuation(self, w):
        # P_cont(w) = (# distinct words preceding w) / (# distinct bigram types)
        return len(self.continuations_.get(w, ())) / max(self.n_bigram_types_, 1)

    def prob(self, w1, w2):
        d = self.discount
        c12 = self.bigrams_.get(w1, {}).get(w2, 0)
        c1 = self.unigrams_.get(w1, 0)
        if c1 == 0:
            return self._p_continuation(w2) or 1e-10
        first = max(c12 - d, 0) / c1
        lam = d * len(self.followers_.get(w1, ())) / c1     # back-off weight
        return first + lam * self._p_continuation(w2)

    def perplexity(self, sentences):
        logp, n = 0.0, 0
        for sent in sentences:
            toks = sent if isinstance(sent, list) else _tokenize(sent)
            for w1, w2 in zip(toks, toks[1:]):
                logp += np.log(max(self.prob(w1, w2), 1e-12)); n += 1
        return float(np.exp(-logp / max(n, 1)))


__all__ = ["KneserNeyLM"]
