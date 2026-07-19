"""Robust z-score via the median absolute deviation: |s - median| / MAD > k."""
import numpy as np


def threshold_mad(scores, k=3.0):
    """Robust z-score via the median absolute deviation: |s - median| / MAD > k."""
    scores = np.asarray(scores)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-12
    return (scores - med) / (1.4826 * mad) > k       # 1.4826 -> std-consistent


__all__ = ["threshold_mad"]
