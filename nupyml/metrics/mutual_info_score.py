"""mutual_info_score"""
import numpy as np
from ..utils import column_or_1d


def _contingency(labels_true, labels_pred):
    labels_true = column_or_1d(labels_true)
    labels_pred = column_or_1d(labels_pred)
    _, ti = np.unique(labels_true, return_inverse=True)
    _, pi = np.unique(labels_pred, return_inverse=True)
    C = np.zeros((ti.max() + 1, pi.max() + 1))
    np.add.at(C, (ti, pi), 1)
    return C


def mutual_info_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    n = C.sum()
    outer = np.outer(C.sum(axis=1), C.sum(axis=0))
    nz = C > 0
    return float((C[nz] / n * (np.log(C[nz] * n) - np.log(outer[nz]))).sum())


__all__ = ["mutual_info_score"]
