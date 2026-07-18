"""Time series v3: volatility, multivariate autoregression, automatic smoothing,
changepoint trends, and multi-seasonal decomposition.

The forecasting gaps left after the ARIMA/Kalman/smoothing core. GARCH models the
VARIANCE (volatility clustering) that ARIMA assumes constant. VAR extends
autoregression to several series that drive EACH OTHER. AutoETS picks the
exponential-smoothing form by information criterion. ProphetForecaster models a
trend that BENDS at changepoints plus Fourier seasonality. MSTL peels apart
several seasonal cycles at once.
"""
import numpy as np
from scipy.optimize import minimize

from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array
from .smoothing import STL


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


class VAR(BaseEstimator):
    """Vector autoregression: several series that predict EACH OTHER (Sims, 1980).

    A univariate AR explains a series by its own past. But interest rates, GDP and
    inflation move TOGETHER -- each one's future depends on the recent past of ALL
    of them. VAR(p) stacks that: every variable is regressed on ``p`` lags of every
    variable, a single multivariate OLS. It is the workhorse of macro-econometrics
    and the basis of Granger causality (does adding X's lags help predict Y?).
    Fits ``Y: (T, k)`` and forecasts all ``k`` series jointly.
    """

    def __init__(self, p=1):
        self.p = p

    def fit(self, Y):
        Y = check_array(Y)
        T, k = Y.shape
        rows, targets = [], []
        for t in range(self.p, T):
            lags = Y[t - self.p:t][::-1].ravel()       # p lags of all k series
            rows.append(np.concatenate([[1.0], lags]))
            targets.append(Y[t])
        X = np.array(rows); Z = np.array(targets)
        self.coef_, *_ = np.linalg.lstsq(X, Z, rcond=None)   # (1+p*k, k)
        self.k_ = k
        self._history = Y[-self.p:].copy()
        return self

    def forecast(self, steps=10):
        hist = list(self._history)
        out = []
        for _ in range(steps):
            lags = np.array(hist[-self.p:][::-1]).ravel()
            x = np.concatenate([[1.0], lags])
            pred = x @ self.coef_
            out.append(pred); hist.append(pred)
        return np.array(out)


class AutoETS(BaseEstimator, RegressorMixin):
    """Pick the exponential-smoothing model by information criterion (Hyndman, 2002).

    Exponential smoothing comes in a family -- with or without a trend, with or
    without a season, each additive or damped. AutoETS fits several of these
    ERROR-TREND-SEASONAL forms and selects the one with the best AIC, so you get
    the appropriate model instead of guessing. It is the engine behind R's
    ``ets()`` / ``forecast`` defaults. Additive components; ``season_length=1``
    disables seasonality.
    """

    def __init__(self, season_length=1):
        self.season_length = season_length

    def _fit_ets(self, y, trend, season):
        m = self.season_length
        n = len(y)
        # grid over smoothing params; additive components
        best = None
        alphas = np.linspace(0.1, 0.9, 5)
        betas = np.linspace(0.05, 0.5, 4) if trend else [0.0]
        gammas = np.linspace(0.05, 0.5, 4) if season else [0.0]
        for a in alphas:
            for b in betas:
                for g in gammas:
                    sse, params = self._run(y, a, b, g, trend, season)
                    k = 1 + trend + season + 1
                    aic = n * np.log(sse / n + 1e-12) + 2 * k
                    if best is None or aic < best[0]:
                        best = (aic, a, b, g, trend, season, params)
        return best

    def _run(self, y, a, b, g, trend, season):
        m = self.season_length
        level = y[0]
        tr = (y[m] - y[0]) / m if trend and len(y) > m else 0.0
        seas = list(y[:m] - y[:m].mean()) if season else [0.0] * max(m, 1)
        sse = 0.0
        for t in range(len(y)):
            s = seas[t % len(seas)] if season else 0.0
            fitted = level + (tr if trend else 0.0) + s
            e = y[t] - fitted
            sse += e * e
            new_level = a * (y[t] - s) + (1 - a) * (level + (tr if trend else 0.0))
            if trend:
                tr = b * (new_level - level) + (1 - b) * tr
            if season:
                seas[t % len(seas)] = g * (y[t] - new_level) + (1 - g) * s
            level = new_level
        return sse, (level, tr, seas)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        cands = [self._fit_ets(y, False, False)]
        if len(y) > 3:
            cands.append(self._fit_ets(y, True, False))
        if self.season_length > 1 and len(y) > 2 * self.season_length:
            cands.append(self._fit_ets(y, True, True))
            cands.append(self._fit_ets(y, False, True))
        self.best_ = min(cands, key=lambda c: c[0])
        _, _, _, _, self.trend_, self.season_, self.state_ = self.best_
        return self

    def forecast(self, steps=10):
        level, tr, seas = self.state_
        out = np.empty(steps)
        for h in range(steps):
            s = seas[h % len(seas)] if self.season_ else 0.0
            out[h] = level + ((h + 1) * tr if self.trend_ else 0.0) + s
        return out


class ProphetForecaster(BaseEstimator):
    """A trend that BENDS at changepoints, plus Fourier seasonality (Taylor, 2018).

    Business time series have a trend whose SLOPE changes (a launch, a policy) and
    smooth seasonal cycles. Prophet models exactly that as a regression:
    a piecewise-LINEAR trend with automatic CHANGEPOINTS where the slope may shift,
    plus Fourier terms for seasonality -- all fit at once by least squares. Being a
    regression (not autoregression) it handles gaps and irregular sampling and
    gives interpretable trend/seasonal components. ``seasonality`` is the period,
    ``n_changepoints`` how many candidate slope changes.
    """

    def __init__(self, n_changepoints=10, seasonality=0, fourier_order=3):
        self.n_changepoints = n_changepoints
        self.seasonality = seasonality
        self.fourier_order = fourier_order

    def _design(self, t):
        cols = [np.ones_like(t), t]                     # intercept + linear trend
        for cp in self.changepoints_:                   # slope changes (hinges)
            cols.append(np.clip(t - cp, 0, None))
        if self.seasonality:
            for k in range(1, self.fourier_order + 1):
                cols.append(np.sin(2 * np.pi * k * t / self.seasonality))
                cols.append(np.cos(2 * np.pi * k * t / self.seasonality))
        return np.column_stack(cols)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.n_ = len(y)
        t = np.arange(self.n_, dtype=float)
        self.changepoints_ = np.linspace(0, self.n_, self.n_changepoints + 2)[1:-1]
        X = self._design(t)
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def forecast(self, steps=10):
        t = np.arange(self.n_, self.n_ + steps, dtype=float)
        return self._design(t) @ self.coef_

    def predict(self, t):
        return self._design(np.asarray(t, float)) @ self.coef_


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


__all__ = ["GARCH", "VAR", "AutoETS", "ProphetForecaster", "MSTL"]
