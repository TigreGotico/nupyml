"""Area under the precision-recall curve. The right AUC under imbalance."""
import numpy as np
from .precision_recall_curve import precision_recall_curve


def average_precision_score(y_true, y_score):
    """Area under the precision-recall curve. The right AUC under imbalance.

    ROC-AUC uses the false-positive rate, whose denominator is the number of
    negatives. When negatives vastly outnumber positives, even a large number of
    false alarms barely moves that rate, so ROC-AUC stays high while the model
    is unusable in practice. Precision's denominator is the number of things
    FLAGGED, which reacts immediately -- so this metric tells the truth on the
    rare-positive problems where it matters.

    Computed as a step-wise sum rather than trapezoid: interpolating a PR curve
    is over-optimistic, because the curve is not linear between operating
    points.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # step-wise integral: sum (R_n - R_{n+1}) * P_n  (recall is decreasing here)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


__all__ = ["average_precision_score"]
