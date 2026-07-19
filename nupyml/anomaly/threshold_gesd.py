"""Generalized ESD test: iteratively remove the most extreme score while a"""
import numpy as np
from .feature_bagging_detector import FeatureBaggingDetector
from .half_space_trees import HalfSpaceTrees
from .loda import LODA
from .mahalanobis_detector import MahalanobisDetector
from .pca_reconstruction_detector import PCAReconstructionDetector
from .threshold_iqr import threshold_iqr
from .threshold_mad import threshold_mad


def threshold_gesd(scores, alpha=0.05, max_outliers=None):
    """Generalized ESD test: iteratively remove the most extreme score while a
    Grubbs-style statistic exceeds its critical value (a principled, parametric
    cut when scores are roughly normal). Returns a boolean outlier mask."""
    from scipy import stats
    scores = np.asarray(scores, float)
    n = len(scores)
    max_outliers = max_outliers or max(1, n // 10)
    mask = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    active = list(idx)
    for i in range(1, max_outliers + 1):
        s = scores[active]
        mean, std = s.mean(), s.std(ddof=1) + 1e-12
        R = np.abs(s - mean) / std
        j = int(np.argmax(R))
        n_i = len(active)
        p = 1 - alpha / (2 * n_i)
        t = stats.t.ppf(p, n_i - 2)
        crit = (n_i - 1) * t / np.sqrt((n_i - 2 + t ** 2) * n_i)
        if R[j] > crit:
            mask[active[j]] = True
            active.pop(j)
        else:
            break
    return mask


__all__ = ["threshold_gesd"]
