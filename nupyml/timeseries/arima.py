"""ARIMA: a series explained by its own past.

THE THREE LETTERS
-----------------
* **AR(p) -- AutoRegressive.** The value is a weighted sum of the last ``p``
  values: ``x_t = sum phi_i x_{t-i} + noise``. "Tomorrow resembles the last few
  days." Momentum and mean-reversion both live here.
* **MA(q) -- Moving Average.** The value is a weighted sum of the last ``q``
  ERRORS: ``x_t = sum theta_i e_{t-i} + noise``. "A shock echoes for a few steps
  before fading." Note this is errors, not values -- a different and often
  confusing thing from the moving average of a smoother.
* **I(d) -- Integrated.** DIFFERENCE the series ``d`` times before modelling, and
  cumulatively sum the forecasts back afterwards. This is the part that handles
  trends, and understanding why is the point of the module.

WHY DIFFERENCING: STATIONARITY IS THE WHOLE GAME
------------------------------------------------
AR and MA both assume the series is STATIONARY -- its mean, variance and
autocorrelation do not change over time. A trending series violates this
outright: its mean is climbing, so "the average level" is not a fixed thing to
regress toward, and the model's coefficients would be estimated against a moving
target.

Differencing fixes it. A series that trends linearly has a CONSTANT first
difference; one step of differencing turns the trend into a level, and the level
is stationary. Two steps handle a quadratic trend, and so on. So ``d`` is not a
tuning knob -- it is however many differences it takes to make the series
stationary, no more (over-differencing injects artificial negative correlation
and inflates variance).

The classic diagnostic: difference until the series looks mean-reverting and its
autocorrelation decays, then stop.

WHY IT LOST, AND WHY IT IS STILL HERE
-------------------------------------
For short univariate series with clear structure, a well-specified ARIMA still
beats a neural network -- it encodes the right assumptions and has few parameters
to overfit. It fades when there are many interacting series, long-range or
nonlinear dependence, or exogenous drivers, which is most modern forecasting. It
is here because it is the vocabulary the whole field is built on: "stationary",
"autocorrelation", "differencing", "the order (p, d, q)" all come from here.

Box & Jenkins (1970).
"""
import numpy as np
from scipy.optimize import minimize

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_array


def difference(x, d=1):
    """Difference a series ``d`` times. ``I`` in ARIMA."""
    x = np.asarray(x, float)
    for _ in range(d):
        x = np.diff(x)
    return x


def integrate(diffs, history, d=1):
    """Undo ``difference``: cumulative-sum the forecasts back onto the last
    ``d`` observed values. Forecasts come out on the differenced scale, and this
    is what returns them to the original one."""
    x = np.asarray(diffs, float)
    for _ in range(d):
        last = history[-1]
        x = last + np.cumsum(x)
        history = history[:-1]
    return x


class AutoRegressive(BaseEstimator, RegressorMixin):
    """AR(p): regress each value on the previous ``p``.

    Fitted by ordinary least squares on the lag matrix -- an AR model IS a linear
    regression whose features are shifted copies of the series. That equivalence
    is worth seeing: nothing new is needed to fit it, only the right design
    matrix.
    """

    def __init__(self, p=1):
        self.p = p

    def _lag_matrix(self, x):
        n = len(x)
        rows = n - self.p
        X = np.column_stack([x[self.p - 1 - i: n - 1 - i] for i in range(self.p)])
        return X, x[self.p:]

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        self.mean_ = x.mean()
        xc = x - self.mean_
        X, y = self._lag_matrix(xc)
        # OLS on the lags -- an AR model is a linear regression in disguise
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.resid_ = y - X @ self.coef_
        self.sigma2_ = float(self.resid_.var())
        self._history = xc[-self.p:]
        return self

    def forecast(self, steps=1):
        """Forecast forward, feeding each prediction back in as the next lag.

        This feedback is why AR forecasts decay toward the mean: with no new
        shocks, each step pulls a little less far, and a stationary AR process
        forgets its starting point geometrically. The long-run forecast is just
        the series mean -- which is the honest thing to say once the past has
        washed out.
        """
        check_is_fitted(self, "coef_")
        hist = list(self._history)
        out = []
        for _ in range(steps):
            pred = np.dot(self.coef_, hist[::-1][:self.p])
            out.append(pred)
            hist.append(pred)
        return np.array(out) + self.mean_


class ARMA(BaseEstimator, RegressorMixin):
    """ARMA(p, q): autoregression plus a moving average of the errors.

    Fitted by maximum likelihood (Gaussian innovations), because the MA part
    cannot be fit by least squares -- the past errors are not observed, they are
    inferred from the model, so the residuals depend on the parameters
    recursively. The likelihood is optimised numerically, which is why ARMA is
    slower and less certain to converge than a pure AR fit.
    """

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q

    def _residuals(self, params, x):
        phi = params[:self.p]
        theta = params[self.p:self.p + self.q]
        n = len(x)
        e = np.zeros(n)
        for t in range(n):
            ar = sum(phi[i] * x[t - 1 - i] for i in range(self.p) if t - 1 - i >= 0)
            # the MA part reads back the errors this same recursion produced --
            # which is why the fit cannot be a simple least squares
            ma = sum(theta[j] * e[t - 1 - j] for j in range(self.q) if t - 1 - j >= 0)
            e[t] = x[t] - ar - ma
        return e

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        self.mean_ = x.mean()
        xc = x - self.mean_

        def neg_loglik(params):
            e = self._residuals(params, xc)
            s2 = np.mean(e ** 2) + 1e-12
            # Gaussian innovations: minimising this is maximum likelihood
            return 0.5 * len(xc) * np.log(s2)

        p0 = np.zeros(self.p + self.q)
        if self.p > 0:
            p0[0] = 0.1
        res = minimize(neg_loglik, p0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6})
        self.params_ = res.x
        self.ar_coef_ = res.x[:self.p]
        self.ma_coef_ = res.x[self.p:self.p + self.q]
        self.resid_ = self._residuals(res.x, xc)
        self.sigma2_ = float(np.mean(self.resid_ ** 2))
        self._x = xc
        return self

    def forecast(self, steps=1):
        check_is_fitted(self, "params_")
        x = list(self._x)
        e = list(self.resid_)
        out = []
        for _ in range(steps):
            ar = sum(self.ar_coef_[i] * x[-1 - i] for i in range(self.p))
            # future errors are unknown, so their expectation -- zero -- is used;
            # this is why the MA part stops contributing beyond q steps out
            ma = sum(self.ma_coef_[j] * e[-1 - j] for j in range(self.q)
                     if len(e) > j)
            pred = ar + ma
            out.append(pred)
            x.append(pred)
            e.append(0.0)
        return np.array(out) + self.mean_


class ARIMA(BaseEstimator, RegressorMixin):
    """ARIMA(p, d, q): difference to stationarity, fit ARMA, integrate back.

    The three parts compose exactly as the name says: difference ``d`` times, fit
    an ARMA(p, q) to the stationary result, forecast, then cumulatively sum the
    forecasts back up to the original scale. The differencing is what lets an
    otherwise stationary model handle a trend, and reintegrating is just its
    inverse.
    """

    def __init__(self, p=1, d=1, q=0):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        self._original = x
        # I: difference to (approximate) stationarity before modelling
        xd = difference(x, self.d) if self.d > 0 else x
        if self.q > 0:
            self._model = ARMA(self.p, self.q).fit(xd)
        else:
            self._model = AutoRegressive(self.p).fit(xd)
        self.resid_ = self._model.resid_
        return self

    def forecast(self, steps=1):
        check_is_fitted(self, "_model")
        fc = self._model.forecast(steps)
        if self.d == 0:
            return fc
        # I^-1: integrate the differenced forecasts back to the original scale
        return integrate(fc, self._original.copy(), self.d)


__all__ = ["AutoRegressive", "ARMA", "ARIMA", "difference", "integrate"]
