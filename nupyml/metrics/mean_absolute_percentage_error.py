"""mean_absolute_percentage_error"""
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


__all__ = ["mean_absolute_percentage_error"]
