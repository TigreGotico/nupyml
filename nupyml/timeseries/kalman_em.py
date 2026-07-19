"""Learn a state-space model's NOISE levels from data (Shumway & Stoffer, 1982)."""
import numpy as np
from ..base import BaseEstimator


class KalmanEM(BaseEstimator):
    """Learn a state-space model's NOISE levels from data (Shumway & Stoffer, 1982).

    A Kalman filter needs to know the process and observation noise variances, but
    you rarely do. Kalman-EM estimates them: the E-step runs the filter and RTS
    smoother to get the expected states, the M-step re-estimates the variances from
    those expectations, and iterating converges to the maximum-likelihood noise
    levels -- so the model calibrates its OWN uncertainty. Fitted here for a local-
    level (random-walk-plus-noise) model; ``signal_to_noise_`` is the learned ratio.
    """

    def __init__(self, max_iter=50, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        n = len(y)
        q = np.var(np.diff(y)) / 2 + 1e-6                   # process var (init)
        r = np.var(y) / 2 + 1e-6                            # obs var (init)
        prev = -np.inf
        for _ in range(self.max_iter):
            # forward filter
            a = np.zeros(n); P = np.zeros(n)
            a[0] = y[0]; P[0] = r
            for t in range(1, n):
                ap = a[t - 1]; Pp = P[t - 1] + q
                K = Pp / (Pp + r)
                a[t] = ap + K * (y[t] - ap); P[t] = (1 - K) * Pp
            # RTS smoother (store the gains J for the lag-one covariance)
            asm = a.copy(); Psm = P.copy(); Js = np.zeros(n)
            for t in range(n - 2, -1, -1):
                Pp = P[t] + q
                J = P[t] / Pp; Js[t] = J
                asm[t] = a[t] + J * (asm[t + 1] - a[t])
                Psm[t] = P[t] + J * J * (Psm[t + 1] - Pp)
            # M-step: full EM variances including the smoothed-state uncertainty
            # E[(mu_t - mu_{t-1})^2] = (dmu)^2 + Psm[t] + Psm[t-1] - 2 cov(mu_t,mu_{t-1})
            cross = Js[:-1] * Psm[1:]
            q = np.mean(np.diff(asm) ** 2 + Psm[1:] + Psm[:-1] - 2 * cross) + 1e-9
            r = np.mean((y - asm) ** 2 + Psm)
            ll = -0.5 * np.sum((y - asm) ** 2) / r
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.process_var_, self.obs_var_ = q, r
        self.signal_to_noise_ = q / r
        self.level_ = asm
        return self

    def forecast(self, steps=10):
        return np.full(steps, self.level_[-1])


__all__ = ["KalmanEM"]
