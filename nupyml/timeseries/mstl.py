"""Multiple seasonal-trend decomposition: peel off several cycles (Bandara, 2021)."""
import numpy as np
from ..base import BaseEstimator, RegressorMixin
from .smoothing import STL


class MSTL(BaseEstimator):
    """Multiple seasonal-trend decomposition: peel off several cycles (Bandara, 2021).

    STL splits a series into ONE trend, ONE season and a remainder. But real series
    often carry SEVERAL seasonalities at once -- daily AND weekly, weekly AND yearly.
    MSTL applies STL iteratively, extracting each seasonal period in turn and
    subtracting it before finding the next, leaving a single trend and remainder.
    The result is one seasonal component per period, each interpretable on its own.
    ``periods`` is the list of season lengths (largest effect first works best).
    """

    def __init__(self, periods, iterations=2):
        self.periods = list(periods)
        self.iterations = iterations

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.seasonal_ = {p: np.zeros_like(y) for p in self.periods}
        deseasonalised = y.copy()
        for _ in range(self.iterations):
            for p in self.periods:
                # add back this period's current estimate, re-extract with STL
                deseasonalised = deseasonalised + self.seasonal_[p]
                stl = STL(season_length=p).fit(deseasonalised)
                self.seasonal_[p] = stl.seasonal_
                deseasonalised = deseasonalised - self.seasonal_[p]
        stl = STL(season_length=self.periods[0]).fit(deseasonalised)
        self.trend_ = stl.trend_
        self.remainder_ = y - self.trend_ - sum(self.seasonal_.values())
        return self


__all__ = ["MSTL"]
