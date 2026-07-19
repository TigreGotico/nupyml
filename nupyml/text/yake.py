"""YAKE: unsupervised keyphrase extraction from statistical word features."""
import re
from collections import defaultdict

import numpy as np


def _tf_sum(gram, tf):
    return sum(tf[w.lower()] for w in gram)


def yake_keywords(text, top_k=10, ngram=3, window=2):
    """Rank keyphrases from STATISTICAL features, unsupervised (Campos et al., 2018).

    RAKE and TextRank need co-occurrence graphs or phrase heuristics; YAKE scores each
    word from cheap statistics -- casing, position, frequency, and context diversity --
    so a distinctive, well-placed word scores LOW (better). Candidate n-grams inherit
    their words' scores, deduplicated. It is fast, language-agnostic, and needs no
    corpus. Returns ``[(phrase, score), ...]`` ascending (lower is more keyword-like).
    """
    sents = re.split(r"[.!?\n]+", text)
    tokens = [re.findall(r"[A-Za-z][A-Za-z'-]+", s) for s in sents]
    flat = [w for s in tokens for w in s]
    N = len(flat) or 1
    lower = [w.lower() for w in flat]
    tf = defaultdict(int)
    for w in lower:
        tf[w] += 1
    positions = defaultdict(list)
    for i, w in enumerate(lower):
        positions[w].append(i)
    left = defaultdict(set); right = defaultdict(set)
    for i, w in enumerate(lower):
        for j in range(max(0, i - window), i):
            left[w].add(lower[j])
        for j in range(i + 1, min(N, i + window + 1)):
            right[w].add(lower[j])
    mean_tf = np.mean(list(tf.values()))
    std_tf = np.std(list(tf.values())) + 1e-9
    score = {}
    for w in tf:
        casing = sum(1 for o in flat if o.lower() == w and o[:1].isupper()) / tf[w]
        pos = np.log(3 + np.median(positions[w]))
        freq = tf[w] / (mean_tf + std_tf)
        rel = 1 + (len(left[w]) + len(right[w])) / (2 * tf[w] + 1e-9)   # context spread
        score[w] = (rel * pos) / (casing + freq / rel + 1e-9)
    # candidate n-grams: lower combined score is better
    cands = {}
    for s in tokens:
        for n in range(1, ngram + 1):
            for i in range(len(s) - n + 1):
                gram = s[i:i + n]
                key = " ".join(w.lower() for w in gram)
                sc = np.prod([score[w.lower()] for w in gram]) / (_tf_sum(gram, tf) + 1)
                cands[key] = min(cands.get(key, np.inf), sc)
    return sorted(cands.items(), key=lambda kv: kv[1])[:top_k]


__all__ = ["yake_keywords"]
