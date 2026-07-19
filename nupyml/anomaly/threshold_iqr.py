"""Tukey's rule: outliers are scores above Q3 + k*IQR."""
import numpy as np


def threshold_iqr(scores, k=1.5):
    """Tukey's rule: outliers are scores above Q3 + k*IQR."""
    scores = np.asarray(scores)
    q1, q3 = np.percentile(scores, [25, 75])
    return scores > q3 + k * (q3 - q1)


__all__ = ["threshold_iqr"]
