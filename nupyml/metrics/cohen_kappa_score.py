"""cohen_kappa_score"""
import numpy as np


def cohen_kappa_score(y1, y2):
    from . import confusion_matrix
    C = confusion_matrix(y1, y2).astype(np.float64)
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(axis=0) @ C.sum(axis=1)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


__all__ = ["cohen_kappa_score"]
