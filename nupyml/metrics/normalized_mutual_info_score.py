"""Shared information between two labellings, scaled to [0, 1]."""
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


def normalized_mutual_info_score(labels_true, labels_pred):
    """Shared information between two labellings, scaled to [0, 1].

    Mutual information asks how much knowing one labelling tells you about the
    other. It is invariant to permutations of the label names -- which is what
    you want when comparing clusterings, where "cluster 0" is arbitrary.

    Raw MI grows with the number of clusters, so normalising by the entropies
    makes it comparable. It does NOT correct for chance, though; for that see
    ``adjusted_rand_score``.
    """
    mi = mutual_info_score(labels_true, labels_pred)
    C = _contingency(labels_true, labels_pred)
    h1, h2 = _entropy(C.sum(axis=1)), _entropy(C.sum(axis=0))
    denom = (h1 + h2) / 2
    return float(mi / denom) if denom > 0 else 1.0


__all__ = ["normalized_mutual_info_score"]
