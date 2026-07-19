"""Additive hazards regression -- effects that CHANGE over time."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array


class AalenAdditiveHazards(BaseEstimator):
    """Additive hazards regression -- effects that CHANGE over time.

    THE CONTRAST WITH COX
    ---------------------
    Cox assumes each covariate multiplies the hazard by a constant factor for all
    time (proportional hazards). Aalen's model instead ADDS time-varying
    contributions::

        hazard(t | x) = b_0(t) + b_1(t) x_1 + ... + b_p(t) x_p

    so a covariate's effect can grow, fade, or reverse as time passes -- exactly
    what proportional hazards forbids. You recover the CUMULATIVE regression
    functions ``B_j(t)``, whose SLOPE at time t is the instantaneous effect; a
    flat stretch means the covariate stops mattering there.

    FIT
    ---
    At each event time, a weighted least-squares step on the at-risk subjects
    gives the increment ``dB``; accumulating those increments over event times is
    the estimator. No likelihood, no proportionality -- just a sequence of linear
    regressions down the risk set.

    Aalen (1989).
    """

    def fit(self, X, durations, events):
        X = check_array(X)
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        n, p = X.shape
        Xd = np.column_stack([np.ones(n), X])           # intercept = baseline
        order = np.argsort(durations)
        self.times_ = []
        increments = []
        B = np.zeros(p + 1)
        for i in order:
            if events[i] != 1:
                continue
            at_risk = durations >= durations[i]         # still-alive subjects
            Xr = Xd[at_risk]
            dN = np.zeros(at_risk.sum())
            # the subject having the event is the one at (its position in Xr)
            risk_idx = np.where(at_risk)[0]
            dN[np.where(risk_idx == i)[0][0]] = 1.0
            # least-squares increment: (X'X)^-1 X' dN
            G = Xr.T @ Xr + 1e-6 * np.eye(p + 1)
            dB = np.linalg.solve(G, Xr.T @ dN)
            B = B + dB
            self.times_.append(durations[i])
            increments.append(B.copy())
        self.times_ = np.array(self.times_)
        self.cumulative_coef_ = np.array(increments) if increments else np.zeros((0, p + 1))
        return self

    def predict_cumulative_hazard(self, X, t=None):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        if len(self.times_) == 0:
            return np.zeros(len(X))
        if t is None:
            t = self.times_[-1]
        k = np.searchsorted(self.times_, t, side="right") - 1
        k = max(0, k)
        return Xd @ self.cumulative_coef_[k]


__all__ = ["AalenAdditiveHazards"]
