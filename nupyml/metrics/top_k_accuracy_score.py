"""top_k_accuracy_score"""
import numpy as np
from ..utils import column_or_1d


def top_k_accuracy_score(y_true, y_score, k=2, labels=None):
    y_true = column_or_1d(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    idx = np.searchsorted(labels, y_true)
    topk = np.argsort(-y_score, axis=1)[:, :k]
    return float(np.mean([i in row for i, row in zip(idx, topk)]))


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------


__all__ = ["top_k_accuracy_score"]
