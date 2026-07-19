"""max_error"""
import numpy as np


def max_error(y_true, y_pred):
    return float(np.max(np.abs(np.asarray(y_true, dtype=np.float64)
                               - np.asarray(y_pred, dtype=np.float64))))


__all__ = ["max_error"]
