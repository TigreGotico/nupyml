"""Mean squared error of predicted probabilities. LOWER is better."""
import numpy as np
from ..utils import column_or_1d


def brier_score_loss(y_true, y_proba, pos_label=None):
    """Mean squared error of predicted probabilities. LOWER is better.

    Unlike log loss, it is bounded and forgiving of confident mistakes -- a
    prediction of 0.0 for a true positive costs 1.0, not infinity. That makes it
    the more stable choice for comparing calibration, and it decomposes neatly
    into calibration and refinement terms.
    """
    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba).astype(np.float64)
    classes = np.unique(y_true)
    if pos_label is None:
        pos_label = classes[-1]
    t = (y_true == pos_label).astype(np.float64)
    return float(np.mean((y_proba - t) ** 2))


__all__ = ["brier_score_loss"]
