"""Parametric accelerated failure time with a Weibull baseline."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array


class WeibullAFT(BaseEstimator):
    """Parametric accelerated failure time with a Weibull baseline.

    THE AFT VIEW
    ------------
    Instead of modelling the hazard (Cox), model log-time directly::

        log T = x·beta + sigma * W,   W ~ standard Gumbel (Weibull baseline)

    so covariates ACCELERATE or decelerate the time to failure by a constant
    factor ``exp(x·beta)`` -- a description many people find more intuitive than a
    hazard ratio ("this treatment doubles expected survival time"). Being fully
    parametric, it extrapolates smoothly beyond the observed follow-up, where Cox
    and Kaplan-Meier simply stop.

    FIT
    ---
    Maximum likelihood over ``beta`` and ``log sigma``, with censored subjects
    contributing their survival ``S(t)`` and events their density ``f(t)``.
    Optimised with L-BFGS.
    """

    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def fit(self, X, durations, events):
        from scipy.optimize import minimize
        X = check_array(X)
        t = np.asarray(durations, float)
        e = np.asarray(events, int)
        logt = np.log(t + 1e-8)
        n, p = X.shape
        Xd = np.column_stack([np.ones(n), X])

        def nll(params):
            beta = params[:p + 1]
            sigma = np.exp(params[p + 1])
            z = (logt - Xd @ beta) / sigma
            log_S = -np.exp(z)                          # log survival
            log_f = z - np.log(sigma) - np.log(t + 1e-8) + log_S   # log density
            return -np.sum(e * log_f + (1 - e) * log_S)

        init = np.concatenate([np.zeros(p + 1), [0.0]])
        res = minimize(nll, init, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x[:p + 1]
        self.sigma_ = float(np.exp(res.x[p + 1]))
        return self

    def predict_median(self, X):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        # median of the Weibull AFT: T = exp(x·beta) * (ln 2)^sigma
        return np.exp(Xd @ self.coef_) * (np.log(2) ** self.sigma_)

    def predict_survival(self, X, t):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        z = (np.log(t + 1e-8) - Xd @ self.coef_) / self.sigma_
        return np.exp(-np.exp(z))


# --- time-dependent evaluation --------------------------------------------


__all__ = ["WeibullAFT"]
