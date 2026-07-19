"""LCS-based F-measure -- rewards in-order overlap (Lin, 2004)."""
import re
import numpy as np


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


def rouge_l(reference, candidate, beta=1.2):
    """LCS-based F-measure -- rewards in-order overlap (Lin, 2004)."""
    ref, cand = _tokenize(reference), _tokenize(candidate)
    m, k = len(ref), len(cand)
    dp = np.zeros((m + 1, k + 1))
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if ref[i - 1] == cand[j - 1] \
                else max(dp[i - 1, j], dp[i, j - 1])
    lcs = dp[m, k]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / max(k, 1), lcs / max(m, 1)
    return float((1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec + 1e-12))


__all__ = ["rouge_l"]
