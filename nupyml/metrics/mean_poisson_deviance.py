"""mean_poisson_deviance"""
import numpy as np


def mean_poisson_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2 * np.mean(term - y_true + y_pred))


__all__ = ["mean_poisson_deviance"]
