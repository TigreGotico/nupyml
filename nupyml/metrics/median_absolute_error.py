"""median_absolute_error"""
import numpy as np


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64))))


__all__ = ["median_absolute_error"]
