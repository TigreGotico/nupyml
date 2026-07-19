"""Modified n-gram PRECISION with a brevity penalty (Papineni, 2002)."""
import re
from collections import defaultdict, Counter
import numpy as np


def _tokenize(text):
    return _WORD.findall(text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


_WORD = re.compile(r"[A-Za-z']+")


def bleu(reference, candidate, max_n=4):
    """Modified n-gram PRECISION with a brevity penalty (Papineni, 2002).

    BLEU scores a machine translation against a reference by how many of its
    n-grams (for n=1..4) appear in the reference -- but CLIPPED, so repeating a
    correct word does not inflate the score -- combined as a geometric mean. A
    brevity penalty stops a system from gaming precision by emitting only a few
    high-confidence words. Precision-oriented; 1.0 is a perfect match.
    """
    ref, cand = _tokenize(reference), _tokenize(candidate)
    if not cand:
        return 0.0
    max_n = min(max_n, len(cand))                        # short sentences use fewer n
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(cand, n)
        ref_ng = _ngrams(ref, n)
        overlap = sum(min(c, ref_ng.get(g, 0)) for g, c in cand_ng.items())
        total = max(sum(cand_ng.values()), 1)
        precisions.append(overlap / total)
    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = np.exp(np.mean([np.log(p) for p in precisions]))
    bp = 1.0 if len(cand) > len(ref) else np.exp(1 - len(ref) / max(len(cand), 1))
    return float(bp * geo)


__all__ = ["bleu"]
