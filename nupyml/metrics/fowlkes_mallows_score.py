"""fowlkes_mallows_score"""
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


def fowlkes_mallows_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    n = C.sum()
    tk = (C ** 2).sum() - n
    pk = (C.sum(axis=0) ** 2).sum() - n
    qk = (C.sum(axis=1) ** 2).sum() - n
    if pk == 0 or qk == 0:
        return 0.0
    return float(tk / np.sqrt(pk * qk))


__all__ = ["fowlkes_mallows_score"]
