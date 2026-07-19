"""Croston that tracks a demand PROBABILITY, not an interval (Teunter, 2011)."""
import numpy as np
from ..base import BaseEstimator


class CrostonTSB(BaseEstimator):
    """Croston that tracks a demand PROBABILITY, not an interval (Teunter, 2011).

    Croston never updates its forecast during a run of zeros, so it cannot react to
    demand that is DYING OUT (obsolescence). TSB replaces the inter-demand interval
    with a demand PROBABILITY that is updated EVERY period -- decayed on a zero,
    bumped on a sale -- and forecasts ``probability * size``. Because it updates on
    zeros too, it correctly lets the forecast fade for a discontinued item, which is
    exactly where Croston fails. ``alpha`` smooths sizes, ``beta`` the probability.
    """

    def __init__(self, alpha=0.1, beta=0.05):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        nz = y > 0
        p = nz.mean()
        size = y[nz].mean() if nz.any() else 0.0
        for t in range(len(y)):
            if y[t] > 0:
                p += self.beta * (1 - p)
                size += self.alpha * (y[t] - size)
            else:
                p += self.beta * (0 - p)                    # decays during zeros
        self.prob_, self.size_ = p, size
        self.rate_ = p * size
        return self

    def predict(self, steps=1):
        return np.full(steps, self.rate_)


__all__ = ["CrostonTSB"]
