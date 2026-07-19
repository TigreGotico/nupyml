"""Score with a RECALL bias and a fragmentation penalty (Banerjee & Lavie, 2005)."""
import re
from collections import Counter, defaultdict


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


def meteor(reference, candidate, alpha=0.9, gamma=0.5, beta=3.0):
    """Score with a RECALL bias and a fragmentation penalty (Banerjee & Lavie, 2005).

    BLEU is precision-oriented and brittle on single sentences. METEOR aligns unigrams
    between candidate and reference, computes an F-mean that WEIGHTS RECALL heavily
    (``alpha``), then applies a FRAGMENTATION penalty: the fewer contiguous CHUNKS the
    matched words form, the better, so word order is rewarded without demanding exact
    n-gram matches. It correlates with human judgement markedly better than BLEU at
    the sentence level. Exact-match version here (no stem/synonym tables).
    """
    ref, cand = _tokenize(reference), _tokenize(candidate)
    if not cand or not ref:
        return 0.0
    ref_count = Counter(ref)
    matched = 0
    match_flags = []
    for w in cand:
        if ref_count[w] > 0:
            ref_count[w] -= 1; matched += 1; match_flags.append(True)
        else:
            match_flags.append(False)
    if matched == 0:
        return 0.0
    P = matched / len(cand); R = matched / len(ref)
    fmean = P * R / (alpha * P + (1 - alpha) * R)
    # count chunks: maximal runs of matched words
    chunks = 0; prev = False
    for f in match_flags:
        if f and not prev:
            chunks += 1
        prev = f
    penalty = gamma * (chunks / matched) ** beta
    return float(fmean * (1 - penalty))


__all__ = ["meteor"]
