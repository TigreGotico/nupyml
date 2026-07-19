"""Are the predicted probabilities honest?"""
import numpy as np
from ..utils import column_or_1d


def calibration_curve(y_true, y_proba, n_bins=5, strategy="uniform"):
    """Are the predicted probabilities honest?

    Bin the predictions, and in each bin compare the mean predicted probability
    with the observed frequency. A perfectly calibrated model lies on the
    diagonal: of the cases it called 70% likely, 70% happen.

    Deviations have a shape worth recognising. An S-curve below the diagonal at
    the top means overconfidence -- the classic naive Bayes or boosted-tree
    signature. ``strategy="quantile"`` puts equal COUNTS in each bin rather than
    equal widths, which avoids near-empty bins when predictions cluster at the
    extremes.
    """
    y_true = column_or_1d(y_true).astype(np.float64)
    y_proba = column_or_1d(y_proba).astype(np.float64)
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bins = np.unique(np.percentile(y_proba, np.linspace(0, 100, n_bins + 1)))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    ids = np.clip(np.searchsorted(bins[1:-1], y_proba, side="right"), 0,
                  len(bins) - 2)
    prob_true, prob_pred = [], []
    for b in range(len(bins) - 1):
        mask = ids == b
        if mask.any():
            prob_true.append(y_true[mask].mean())
            prob_pred.append(y_proba[mask].mean())
    return np.array(prob_true), np.array(prob_pred)


__all__ = ["calibration_curve"]
