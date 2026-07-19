"""Document distance as optimal transport over word embeddings (Kusner, 2015)."""
import re
from collections import defaultdict, Counter
import numpy as np


def _tokenize(text):
    return _WORD.findall(text.lower())


def _get(embeddings, token):
    if callable(embeddings):
        try:
            return np.asarray(embeddings(token), float)
        except Exception:
            return None
    return np.asarray(embeddings[token], float) if token in embeddings else None


def _emd(a, b, C):
    # exact earth mover's distance via a small transportation LP (greedy NW + ...)
    from scipy.optimize import linprog
    n, m = C.shape
    Aeq = np.zeros((n + m, n * m))
    for i in range(n):
        Aeq[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        Aeq[n + j, j::m] = 1
    res = linprog(C.ravel(), A_eq=Aeq, b_eq=np.concatenate([a, b]),
                  bounds=[(0, None)] * (n * m), method="highs")
    return res.fun


_WORD = re.compile(r"[A-Za-z']+")


def word_movers_distance(doc1, doc2, embeddings, reg=0.0):
    """Document distance as optimal transport over word embeddings (Kusner, 2015).

    Two sentences can share no words yet mean the same thing. Word Mover's Distance
    measures how far the words of one document must "travel" in EMBEDDING space to
    match the other -- an optimal-transport problem where the ground cost is the
    euclidean distance between word vectors. It captures that "Obama speaks to the
    press" is close to "the president greets reporters" despite zero word overlap.
    ``embeddings`` maps a token to a vector (dict or callable); ``reg>0`` uses the
    faster entropic (Sinkhorn) approximation.
    """
    from scipy.spatial.distance import cdist
    from ..optimal_transport import sinkhorn

    def bow(doc):
        toks = [t for t in _tokenize(doc) if _get(embeddings, t) is not None]
        c = Counter(toks)
        words = list(c)
        weights = np.array([c[w] for w in words], float)
        return words, weights / weights.sum()

    w1, a = bow(doc1)
    w2, b = bow(doc2)
    if not w1 or not w2:
        return np.inf
    V1 = np.array([_get(embeddings, w) for w in w1])
    V2 = np.array([_get(embeddings, w) for w in w2])
    C = cdist(V1, V2)
    if reg > 0:
        plan, cost = sinkhorn(a, b, C, reg=reg)
        return float(cost)
    return float(_emd(a, b, C))


__all__ = ["word_movers_distance"]
