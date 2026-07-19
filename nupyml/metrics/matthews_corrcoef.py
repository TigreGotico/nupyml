"""Correlation between predictions and truth, in [-1, +1]."""
import numpy as np


def matthews_corrcoef(y_true, y_pred):
    """Correlation between predictions and truth, in [-1, +1].

    The most honest single number for imbalanced binary problems, because it
    uses all four cells of the confusion matrix. Accuracy and F1 can both look
    excellent while a whole class is ignored; MCC cannot -- it is only high when
    the model does well on both classes. 0 is chance, negative is
    anti-correlated.
    """
    from . import confusion_matrix
    C = confusion_matrix(y_true, y_pred).astype(np.float64)
    t = C.sum(axis=1)   # true counts
    p = C.sum(axis=0)   # predicted counts
    n = C.sum()
    cov_ytp = np.trace(C) * n - t @ p
    cov_yy = n ** 2 - t @ t
    cov_pp = n ** 2 - p @ p
    denom = np.sqrt(cov_yy * cov_pp)
    return float(cov_ytp / denom) if denom > 0 else 0.0


__all__ = ["matthews_corrcoef"]
