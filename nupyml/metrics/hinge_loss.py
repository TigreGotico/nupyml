"""hinge_loss"""
import numpy as np
from ..utils import column_or_1d


def hinge_loss(y_true, pred_decision):
    y_true = column_or_1d(y_true)
    pred_decision = column_or_1d(pred_decision).astype(np.float64)
    classes = np.unique(y_true)
    t = np.where(y_true == classes[-1], 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, 1.0 - t * pred_decision)))


__all__ = ["hinge_loss"]
