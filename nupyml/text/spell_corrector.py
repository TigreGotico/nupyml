"""Fix typos with a noisy-channel edit model (Norvig, 2007)."""
import re
from collections import Counter, defaultdict


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


class SpellCorrector:
    """Fix typos with a noisy-channel edit model (Norvig, 2007).

    A misspelling is a correct word corrupted by a few edits. To fix it, generate
    every word within one or two EDITS (insert, delete, replace, transpose) of the
    input, keep those that are REAL words (in a frequency dictionary), and return the
    most probable one -- most frequent wins, since a common word is a more likely
    intended target than a rare one. It is the whole of a working spell-checker in a
    page of code, and a clean illustration of the noisy-channel model. Trained from a
    word-frequency corpus.
    """

    def __init__(self):
        self.freq_ = Counter()

    def fit(self, corpus):
        for line in corpus:
            self.freq_.update(_tokenize(line))
        return self

    def _edits1(self, word):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
        inserts = [a + c + b for a, b in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _known(self, words):
        return {w for w in words if w in self.freq_}

    def correct(self, word):
        word = word.lower()
        candidates = (self._known([word]) or self._known(self._edits1(word))
                      or self._known({e2 for e1 in self._edits1(word)
                                      for e2 in self._edits1(e1)}) or {word})
        return max(candidates, key=lambda w: self.freq_[w])   # most probable


__all__ = ["SpellCorrector"]
