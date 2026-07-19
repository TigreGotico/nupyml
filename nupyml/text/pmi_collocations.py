"""Find word pairs that go together MORE than chance (Church & Hanks, 1990)."""
import re
from collections import Counter, defaultdict
import numpy as np


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


def pmi_collocations(corpus, min_count=2, top_k=10):
    """Find word pairs that go together MORE than chance (Church & Hanks, 1990).

    "New York" is a unit; "the cat" is not, even though "the" is far more frequent.
    Raw bigram counts confuse the two. Pointwise mutual information corrects for how
    common each word is on its own: ``PMI = log2 P(w1,w2) / (P(w1) P(w2))`` -- how
    much MORE the pair occurs than if the words were independent. High PMI pinpoints
    genuine collocations (names, idioms, technical terms). A minimum count filters the
    rare-pair PMI spikes. Returns the top pairs with their PMI.
    """
    unigrams = Counter()
    bigrams = Counter()
    for line in corpus:
        toks = _tokenize(line)
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    total_u = sum(unigrams.values())
    total_b = sum(bigrams.values())
    scores = []
    for (w1, w2), c in bigrams.items():
        if c < min_count:
            continue
        p12 = c / total_b
        p1 = unigrams[w1] / total_u
        p2 = unigrams[w2] / total_u
        scores.append(((w1, w2), np.log2(p12 / (p1 * p2))))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


__all__ = ["pmi_collocations"]
