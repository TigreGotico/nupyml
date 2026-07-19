"""Calibrate binary probabilities by BINNING and using each bin's accuracy."""
import numpy as np
from ..base import BaseEstimator


class HistogramBinning(BaseEstimator):
    """Calibrate binary probabilities by BINNING and using each bin's accuracy.

    The most direct non-parametric calibrator: partition predicted probabilities
    into bins, and map every prediction in a bin to the OBSERVED positive rate of
    that bin. Assumption-free and easy, but blocky (piecewise-constant) and
    hungry for data per bin -- the trade against smooth methods like isotonic or
    beta. ``fit`` on scores + binary labels; ``transform`` maps new scores.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, scores, y):
        scores = np.asarray(scores, float)
        y = np.asarray(y)
        self.edges_ = np.linspace(0, 1, self.n_bins + 1)
        self.bin_prob_ = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            lo, hi = self.edges_[b], self.edges_[b + 1]
            mask = (scores >= lo) & (scores < hi) if b < self.n_bins - 1 else \
                (scores >= lo) & (scores <= hi)
            self.bin_prob_[b] = y[mask].mean() if mask.sum() > 0 else (lo + hi) / 2
        return self

    def transform(self, scores):
        scores = np.asarray(scores, float)
        idx = np.clip(np.searchsorted(self.edges_, scores) - 1, 0, self.n_bins - 1)
        return self.bin_prob_[idx]


__all__ = ["HistogramBinning"]
