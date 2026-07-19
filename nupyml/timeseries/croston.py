"""Forecasting for INTERMITTENT demand (Croston, 1972)."""
import numpy as np
from ..base import BaseEstimator


class Croston(BaseEstimator):
    """Forecasting for INTERMITTENT demand (Croston, 1972).

    Exponential smoothing applied to a mostly-zero series chases the occasional
    spike and decays wrongly between them. Croston's insight: smooth the demand
    SIZES and the INTERVALS between demands SEPARATELY, and forecast the rate as
    ``size / interval``. So a part sold in bursts of 4 every ~5 periods forecasts a
    steady 0.8/period, not a decaying echo of the last spike. The standard method
    for spare-parts and slow-moving inventory.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float)
        nz = np.where(y > 0)[0]
        if len(nz) == 0:
            self.rate_ = 0.0
            return self
        sizes = y[nz]
        intervals = np.diff(np.concatenate([[-1], nz]))   # gaps between demands
        z = sizes[0]; x = intervals[0]
        for s, iv in zip(sizes[1:], intervals[1:]):
            z += self.alpha * (s - z)                      # smooth demand size
            x += self.alpha * (iv - x)                     # smooth interval
        self.rate_ = z / x                                 # demand per period
        return self

    def predict(self, horizon):
        return np.full(horizon, self.rate_)

    forecast = predict


__all__ = ["Croston"]
