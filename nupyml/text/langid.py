"""Language identification from character n-gram profiles."""
from collections import defaultdict

import numpy as np

from ..base import BaseEstimator


class LanguageDetector(BaseEstimator):
    """Identify a language from CHARACTER n-grams (Cavnar & Trenkle, 1994).

    Every language has a fingerprint in its letter statistics -- English is full of
    "the" and "ing", Portuguese of "ao" and "que". This learns a character n-gram
    profile per language from example text and classifies a new string by the language
    whose profile it matches best (log-probability under a smoothed n-gram model). No
    dictionaries, robust on short strings, the classic approach to language ID.
    ``n`` is the character n-gram order.
    """

    def __init__(self, n=3):
        self.n = n

    def _grams(self, text):
        t = "  " + text.lower() + " "
        return [t[i:i + self.n] for i in range(len(t) - self.n + 1)]

    def fit(self, texts, labels):
        self.profiles_ = {}
        self.vocab_ = set()
        by_lang = defaultdict(lambda: defaultdict(int))
        for t, lab in zip(texts, labels):
            for g in self._grams(t):
                by_lang[lab][g] += 1; self.vocab_.add(g)
        V = len(self.vocab_)
        for lab, counts in by_lang.items():
            total = sum(counts.values())
            self.profiles_[lab] = (counts, total, V)        # add-one smoothed at query
        self.classes_ = sorted(self.profiles_)
        return self

    def predict(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        out = []
        for t in texts:
            grams = self._grams(t)
            best, best_ll = None, -np.inf
            for lab, (counts, total, V) in self.profiles_.items():
                ll = sum(np.log((counts.get(g, 0) + 1) / (total + V)) for g in grams)
                if ll > best_ll:
                    best_ll, best = ll, lab
            out.append(best)
        return out[0] if single else out


__all__ = ["LanguageDetector"]
