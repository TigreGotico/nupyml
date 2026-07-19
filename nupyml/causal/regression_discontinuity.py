"""Identify an effect at a sharp CUTOFF rule (Thistlethwaite & Campbell, 1960)."""
import numpy as np
from ..base import BaseEstimator, clone


class RegressionDiscontinuity(BaseEstimator):
    """Identify an effect at a sharp CUTOFF rule (Thistlethwaite & Campbell, 1960).

    When treatment is assigned by a threshold -- a scholarship for scores >= 80, a
    subsidy for income <= a line -- units just above and just below the cutoff are
    essentially identical except for treatment. Regression discontinuity fits a
    LOCAL LINEAR regression on each side of the cutoff (within a bandwidth) and reads
    the treatment effect off the JUMP in the fitted line at the boundary. It needs no
    randomisation, only that everything else varies smoothly through the cutoff.
    ``running`` is the assignment variable, ``cutoff`` the threshold.
    """

    def __init__(self, cutoff=0.0, bandwidth=None):
        self.cutoff = cutoff
        self.bandwidth = bandwidth

    def fit(self, running, y):
        r = np.asarray(running, float).ravel() - self.cutoff
        y = np.asarray(y, float).ravel()
        h = self.bandwidth if self.bandwidth is not None else np.std(r)
        keep = np.abs(r) <= h
        r, y = r[keep], y[keep]
        left = r < 0
        # local linear fit on each side; the intercept gap at 0 is the effect
        al, bl = self._fit_line(r[left], y[left])
        ar, br = self._fit_line(r[~left], y[~left])
        self.effect_ = ar - al
        self.left_intercept_, self.right_intercept_ = al, ar
        return self

    @staticmethod
    def _fit_line(x, y):
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef[0], coef[1]

    def effect(self):
        return self.effect_


__all__ = ["RegressionDiscontinuity"]
