"""homogeneity_score"""
import numpy as np
from ..utils import column_or_1d
from .mutual_info_score import mutual_info_score


def _contingency(labels_true, labels_pred):
    labels_true = column_or_1d(labels_true)
    labels_pred = column_or_1d(labels_pred)
    _, ti = np.unique(labels_true, return_inverse=True)
    _, pi = np.unique(labels_pred, return_inverse=True)
    C = np.zeros((ti.max() + 1, pi.max() + 1))
    np.add.at(C, (ti, pi), 1)
    return C


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def homogeneity_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    h_c = _entropy(C.sum(axis=1))
    if h_c == 0:
        return 1.0
    mi = mutual_info_score(labels_true, labels_pred)
    return float(mi / h_c)


__all__ = ["homogeneity_score"]
