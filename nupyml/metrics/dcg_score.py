"""dcg_score"""
import numpy as np


def dcg_score(y_true, y_score, k=None):
    y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    y_score = np.atleast_2d(np.asarray(y_score, dtype=np.float64))
    order = np.argsort(-y_score, axis=1)
    gains = np.take_along_axis(y_true, order, axis=1)
    if k is not None:
        gains = gains[:, :k]
    discounts = 1.0 / np.log2(np.arange(gains.shape[1]) + 2)
    return float((gains @ discounts).mean())


__all__ = ["dcg_score"]
