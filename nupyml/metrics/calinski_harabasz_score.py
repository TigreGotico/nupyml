"""calinski_harabasz_score"""
import numpy as np
from ..utils import column_or_1d


def calinski_harabasz_score(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = column_or_1d(labels)
    uniq = np.unique(labels)
    k, n = len(uniq), len(X)
    overall = X.mean(axis=0)
    between, within = 0.0, 0.0
    for c in uniq:
        Xc = X[labels == c]
        mean_c = Xc.mean(axis=0)
        between += len(Xc) * ((mean_c - overall) ** 2).sum()
        within += ((Xc - mean_c) ** 2).sum()
    if within == 0:
        return float("inf")
    return float(between / within * (n - k) / (k - 1))


__all__ = ["calinski_harabasz_score"]
