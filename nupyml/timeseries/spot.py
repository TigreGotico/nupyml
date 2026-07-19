"""SPOT: extreme-value anomaly detection by fitting the tail (peaks-over-threshold)."""
import numpy as np

from ..base import BaseEstimator


class SPOT(BaseEstimator):
    """Flag anomalies by fitting the TAIL, not the whole distribution (Siffer, 2017).

    Setting an anomaly threshold by hand fails when the data's scale drifts or is
    unknown. SPOT (Streaming Peaks-Over-Threshold) uses EXTREME VALUE THEORY: it fits a
    Generalised Pareto Distribution to the PEAKS above an initial high quantile, then
    sets the alarm threshold at the value whose exceedance probability is a chosen risk
    ``q``. So the threshold is derived from the tail's own shape and a target false-
    alarm RATE, adapting automatically to the data's scale -- principled anomaly
    detection with a statistical guarantee. ``q`` is the desired anomaly probability.
    """

    def __init__(self, q=1e-3, init_quantile=0.95):
        self.q = q
        self.init_quantile = init_quantile

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.t_ = np.quantile(y, self.init_quantile)      # peak-selection threshold
        peaks = y[y > self.t_] - self.t_
        gamma, sigma = self._fit_gpd(peaks)
        n, Nt = len(y), len(peaks)
        # threshold z_q whose exceedance probability equals q (GPD quantile)
        if abs(gamma) > 1e-6:
            self.threshold_ = self.t_ + (sigma / gamma) * ((self.q * n / Nt) ** (-gamma) - 1)
        else:
            self.threshold_ = self.t_ - sigma * np.log(self.q * n / Nt)
        self.gamma_, self.sigma_ = gamma, sigma
        return self

    @staticmethod
    def _fit_gpd(peaks):
        # method-of-moments estimate of the GPD shape and scale
        m, v = peaks.mean(), peaks.var() + 1e-9
        gamma = 0.5 * (1 - m ** 2 / v)
        sigma = 0.5 * m * (m ** 2 / v + 1)
        return gamma, max(sigma, 1e-6)

    def predict(self, y):
        return (np.asarray(y, float) > self.threshold_).astype(int)


__all__ = ["SPOT"]
