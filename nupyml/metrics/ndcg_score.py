"""ndcg_score"""
import numpy as np
from .dcg_score import dcg_score


def ndcg_score(y_true, y_score, k=None):
    y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    ideal = dcg_score(y_true, y_true, k=k)
    if ideal == 0:
        return 0.0
    return dcg_score(y_true, y_score, k=k) / ideal


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


__all__ = ["ndcg_score"]
