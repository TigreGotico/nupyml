"""mean_gamma_deviance"""
import numpy as np


def mean_gamma_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-12)
    return float(2 * np.mean(np.log(y_pred / y_true) + y_true / y_pred - 1))


__all__ = ["mean_gamma_deviance"]
