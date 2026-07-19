"""Model the VARIANCE, not the mean: volatility clustering (Bollerslev, 1986)."""
import numpy as np
from scipy.optimize import minimize
from ..base import BaseEstimator, RegressorMixin


class GARCH(BaseEstimator):
    """Model the VARIANCE, not the mean: volatility clustering (Bollerslev, 1986).

    ARIMA assumes constant-variance noise, but real financial and many physical
    series show VOLATILITY CLUSTERING -- calm stretches and turbulent stretches.
    GARCH(1,1) makes the conditional variance itself autoregressive::

        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    so a big shock (``eps^2``) RAISES the predicted variance of the next step, and
    it decays back at rate ``beta``. Fit by maximum likelihood; ``forecast`` returns
    the predicted variance path (which converges to the unconditional variance
    ``omega / (1 - alpha - beta)``).
    """

    def __init__(self):
        pass

    def _nll(self, params, eps):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        var = np.empty(len(eps))
        var[0] = eps.var()
        for t in range(1, len(eps)):
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        return 0.5 * np.sum(np.log(2 * np.pi * var) + eps ** 2 / var)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.mean_ = y.mean()
        eps = y - self.mean_
        v = eps.var()
        res = minimize(self._nll, [v * 0.1, 0.1, 0.8], args=(eps,),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        self.omega_, self.alpha_, self.beta_ = res.x
        # reconstruct the final conditional variance
        var = v
        for t in range(1, len(eps)):
            var = self.omega_ + self.alpha_ * eps[t - 1] ** 2 + self.beta_ * var
        self._last_var = var
        self._last_eps2 = eps[-1] ** 2
        return self

    def forecast(self, steps=10):
        uncond = self.omega_ / (1 - self.alpha_ - self.beta_)
        persistence = self.alpha_ + self.beta_
        # one-step-ahead variance from the last observation
        var1 = self.omega_ + self.alpha_ * self._last_eps2 + self.beta_ * self._last_var
        # h-step variance mean-reverts to the unconditional level at rate (a+b)
        return np.array([uncond + persistence ** h * (var1 - uncond)
                         for h in range(steps)])


__all__ = ["GARCH"]
