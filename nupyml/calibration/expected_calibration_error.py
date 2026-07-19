"""ECE: the gap between confidence and accuracy, averaged over bins."""
import numpy as np


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: the gap between confidence and accuracy, averaged over bins.

    Bin predictions by their confidence (max probability); in a well-calibrated
    model, the average confidence in each bin equals the accuracy in that bin. ECE
    is the average absolute confidence-minus-accuracy gap, weighted by bin size --
    a single number for "how much should I trust the probabilities". ``probs`` is
    the (n, n_classes) probability matrix (or (n,) for binary positive-class prob).
    """
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true)
    if probs.ndim == 1:
        conf = np.maximum(probs, 1 - probs)
        pred = (probs >= 0.5).astype(int)
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


__all__ = ["expected_calibration_error"]
