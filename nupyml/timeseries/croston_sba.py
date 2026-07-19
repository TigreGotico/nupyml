"""Croston with the Syntetos-Boylan bias correction (Syntetos & Boylan, 2005)."""
import numpy as np
from ..base import BaseEstimator


class CrostonSBA(BaseEstimator):
    """Croston with the Syntetos-Boylan bias correction (Syntetos & Boylan, 2005).

    Croston's method forecasts intermittent demand as ``size / interval``, but that
    ratio is BIASED UPWARD (the expectation of a ratio is not the ratio of
    expectations). SBA multiplies the forecast by ``1 - alpha/2`` -- a small,
    principled correction that removes most of the bias and, on the standard
    intermittent-demand benchmarks, beats plain Croston. ``alpha`` smooths both the
    demand sizes and the gaps between them.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        nz = np.where(y > 0)[0]
        if len(nz) == 0:
            self.rate_ = 0.0
            return self
        size = y[nz[0]]; interval = nz[0] + 1
        last = nz[0]
        for i in nz[1:]:
            gap = i - last
            size += self.alpha * (y[i] - size)
            interval += self.alpha * (gap - interval)
            last = i
        self.size_, self.interval_ = size, max(interval, 1e-6)
        self.rate_ = (1 - self.alpha / 2) * size / self.interval_   # SBA correction
        return self

    def predict(self, steps=1):
        return np.full(steps, self.rate_)


__all__ = ["CrostonSBA"]
