"""Recall-oriented n-gram overlap for summarisation (Lin, 2004)."""
import re
from collections import defaultdict, Counter


def _tokenize(text):
    return _WORD.findall(text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


_WORD = re.compile(r"[A-Za-z']+")


def rouge_n(reference, candidate, n=1):
    """Recall-oriented n-gram overlap for summarisation (Lin, 2004)."""
    ref, cand = _tokenize(reference), _tokenize(candidate)
    ref_ng, cand_ng = _ngrams(ref, n), _ngrams(cand, n)
    overlap = sum(min(c, cand_ng.get(g, 0)) for g, c in ref_ng.items())
    return overlap / max(sum(ref_ng.values()), 1)        # recall


__all__ = ["rouge_n"]
