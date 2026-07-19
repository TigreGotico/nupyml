"""explained_variance_score"""
import numpy as np


def explained_variance_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_res = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0 if var_res > 0 else 1.0
    return float(1 - var_res / var_y)


__all__ = ["explained_variance_score"]
