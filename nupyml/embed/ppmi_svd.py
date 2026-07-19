"""Count-based word vectors: PPMI matrix truncated by SVD."""
import numpy as np
from collections import Counter


def ppmi_svd(sentences, dim=50, window=2, smoothing=0.75):
    """Count-based word vectors: PPMI matrix truncated by SVD.

    Before neural word2vec there were COUNT vectors, and Levy & Goldberg (2014)
    showed word2vec is implicitly factorising one. Build the word-context
    co-occurrence counts within a window; convert to POSITIVE POINTWISE MUTUAL
    INFORMATION ``max(log( P(w,c) / (P(w)P(c)) ), 0)`` -- which measures how much
    more often two words co-occur than chance, clamped at zero; then take the top
    ``dim`` singular vectors. The result rivals word2vec, is deterministic, and
    needs no training loop. ``smoothing`` raises context counts to a power (the
    context-distribution smoothing that improves rare-word vectors).

    Returns (vocab list, vectors dict word -> vector).
    """
    counts = Counter()
    ctx_counts = Counter()
    word_counts = Counter()
    total = 0
    for sent in sentences:
        for i, w in enumerate(sent):
            word_counts[w] += 1
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if j != i:
                    counts[(w, sent[j])] += 1
                    ctx_counts[sent[j]] += 1
                    total += 1
    vocab = sorted(word_counts)
    idx = {w: k for k, w in enumerate(vocab)}
    V = len(vocab)
    ctx_sm = {c: ctx_counts[c] ** smoothing for c in vocab}
    Z = sum(ctx_sm.values())
    M = np.zeros((V, V))
    for (w, c), n in counts.items():
        p_wc = n / total
        p_w = word_counts[w] / total
        p_c = ctx_sm[c] / Z
        M[idx[w], idx[c]] = max(np.log(p_wc / (p_w * p_c) + 1e-12), 0.0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(s[:dim])
    return vocab, {w: emb[idx[w]] for w in vocab}


__all__ = ["ppmi_svd"]
