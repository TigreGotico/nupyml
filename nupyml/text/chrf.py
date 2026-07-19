"""Character n-gram F-score -- robust to morphology (Popović, 2015)."""
import re
from collections import Counter, defaultdict
import numpy as np


def _ngrams_str(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(reference, candidate, max_n=6, beta=2.0):
    """Character n-gram F-score -- robust to morphology (Popović, 2015).

    Word-level BLEU misses near-matches: "colour" vs "color" scores zero overlap.
    chrF works on CHARACTER n-grams instead, so shared stems and inflections still
    match, and it correlates better with human judgement especially for
    morphologically rich languages and for short segments. It is the mean F-score
    over character n-grams of length 1..``max_n``, with recall weighted ``beta`` times
    precision. No tokenisation needed. 1.0 is a perfect match.
    """
    ref = re.sub(r"\s+", "", reference.lower())
    cand = re.sub(r"\s+", "", candidate.lower())
    fs = []
    for n in range(1, max_n + 1):
        rg, cg = _ngrams_str(ref, n), _ngrams_str(cand, n)
        overlap = sum((rg & cg).values())
        if sum(cg.values()) == 0 or sum(rg.values()) == 0:
            continue
        p = overlap / sum(cg.values())
        r = overlap / sum(rg.values())
        if p + r == 0:
            fs.append(0.0)
        else:
            fs.append((1 + beta ** 2) * p * r / (beta ** 2 * p + r))
    return float(np.mean(fs)) if fs else 0.0


__all__ = ["chrf"]
