"""balanced_accuracy_score"""
import numpy as np


def balanced_accuracy_score(y_true, y_pred):
    from . import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(per_class.mean())


__all__ = ["balanced_accuracy_score"]
