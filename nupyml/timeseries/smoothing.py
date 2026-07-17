"""Exponential smoothing and STL: forecasts from weighted memory.

THE ONE IDEA BEHIND ALL OF IT
-----------------------------
Recent observations matter more than old ones, and the weights should fade
GEOMETRICALLY::

    level_t = alpha * observation_t + (1 - alpha) * level_{t-1}

Unrolled, that is a weighted average of the whole past with weights
``alpha, alpha(1-a), alpha(1-a)^2, ...`` -- exponentially decaying. One parameter
``alpha`` sets the memory: near 1 forgets fast and tracks the latest value; near
0 remembers long and smooths hard.

That single recursion, elaborated, is the entire Holt-Winters family, and it is
still among the best forecasters on plain seasonal business series -- cheap,
robust, and hard to beat with a handful of data points.

THE PROGRESSION
---------------
Each method adds one more thing that gets its own smoothed state and its own rate:

* ``SimpleExponentialSmoothing`` -- LEVEL only. Forecasts a flat line, so it is
  for series with no trend or season.
* ``Holt``                       -- LEVEL + TREND. Forecasts a sloped line. Adds
  ``beta`` for how fast the trend adapts.
* ``HoltWinters``                -- LEVEL + TREND + SEASON. Forecasts a sloped,
  repeating pattern. Adds ``gamma`` for the season.

Read them in order; each is the last plus one more smoothed component, and seeing
that makes the final three-equation method obvious instead of arbitrary.

WHY IT IS A STATE-SPACE MODEL IN DISGUISE
-----------------------------------------
Level, trend and season are hidden states, updated by new observations exactly as
the Kalman filter updates its state. Exponential smoothing IS a structural state-
space model with fixed gains, which is why the two halves of this package are the
same subject in two dialects -- see ``kalman.py``.
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, check_is_fitted


class SimpleExponentialSmoothing(BaseEstimator, RegressorMixin):
    """Level only: a geometrically-weighted average of the past.

    Forecasts a FLAT line at the last level -- with no trend term, it has no way
    to project a slope, so the best it can say about the future is "the same as
    now". Use it only when the series genuinely has no trend or season; otherwise
    it lags a trend forever, always one step behind.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        level = x[0]
        self.fitted_ = np.empty(len(x))
        for t in range(len(x)):
            self.fitted_[t] = level
            level = self.alpha * x[t] + (1 - self.alpha) * level
        self.level_ = level
        return self

    def forecast(self, steps=1):
        check_is_fitted(self, "level_")
        return np.full(steps, self.level_)


class Holt(BaseEstimator, RegressorMixin):
    """Level + trend: forecasts a sloped line.

    Two states now, each with its own smoothing rate: the LEVEL (where the series
    is) updated by ``alpha``, and the TREND (how fast it is moving) updated by
    ``beta``. The forecast extends the level along the trend, so unlike simple
    smoothing it can follow a rising or falling series instead of lagging it.

    ``damped`` multiplies the trend by ``phi < 1`` each step forward, so the
    projected slope FLATTENS out. Undamped trends extrapolated far ahead are the
    classic forecasting embarrassment -- a linear rise continued to absurdity --
    and damping is the standard, well-earned fix: real trends rarely persist
    forever, so a forecast should not assume they do.
    """

    def __init__(self, alpha=0.3, beta=0.1, damped=False, phi=0.98):
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.phi = phi

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        level = x[0]
        trend = x[1] - x[0] if len(x) > 1 else 0.0
        phi = self.phi if self.damped else 1.0
        self.fitted_ = np.empty(len(x))
        for t in range(len(x)):
            self.fitted_[t] = level + phi * trend
            prev_level = level
            level = self.alpha * x[t] + (1 - self.alpha) * (level + phi * trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * phi * trend
        self.level_, self.trend_ = level, trend
        return self

    def forecast(self, steps=1):
        check_is_fitted(self, "level_")
        phi = self.phi if self.damped else 1.0
        # damped: the trend's contribution is a geometric series phi + phi^2 +...,
        # which converges, so the forecast approaches a horizontal asymptote
        # instead of a runaway line
        factors = np.cumsum([phi ** (i + 1) for i in range(steps)]) if self.damped \
            else np.arange(1, steps + 1)
        return self.level_ + factors * self.trend_


class HoltWinters(BaseEstimator, RegressorMixin):
    """Level + trend + season: the full triple exponential smoothing.

    Three states, three rates: level (``alpha``), trend (``beta``) and a seasonal
    pattern of length ``season_length`` (``gamma``). The seasonal component is a
    vector, one smoothed factor per position in the cycle, updated as each cycle
    recurs.

    ADDITIVE vs MULTIPLICATIVE
    --------------------------
    The choice is not cosmetic. ADDITIVE season adds a fixed amount ("+50 every
    December"); MULTIPLICATIVE scales ("+20% every December"). Get it wrong and
    the model fights the data: additive smoothing on a series whose swings grow
    with its level will under-fit the peaks and over-fit the troughs, because it
    is forcing a constant offset onto a proportional effect. The tell is whether
    the seasonal swings widen as the series rises -- if they do, it is
    multiplicative.
    """

    def __init__(self, season_length=12, alpha=0.3, beta=0.1, gamma=0.1,
                 trend=True, seasonal="additive"):
        self.season_length = season_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trend = trend
        self.seasonal = seasonal

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        m = self.season_length
        if len(x) < 2 * m:
            raise ValueError(
                f"need at least two full seasons ({2 * m} points) to fit")

        # initialise from the first two cycles: level from the first cycle's mean,
        # trend from the change between the first two cycles' means, season from
        # each position's departure from its cycle mean
        level = x[:m].mean()
        trend = (x[m:2 * m].mean() - x[:m].mean()) / m if self.trend else 0.0
        if self.seasonal == "additive":
            season = list(x[:m] - level)
        else:
            season = list(x[:m] / (level + 1e-12))

        self.fitted_ = np.empty(len(x))
        for t in range(len(x)):
            s = season[t % m]
            if self.seasonal == "additive":
                self.fitted_[t] = level + trend + s
                prev_level = level
                level = self.alpha * (x[t] - s) + (1 - self.alpha) * (level + trend)
                season[t % m] = self.gamma * (x[t] - level) + (1 - self.gamma) * s
            else:
                self.fitted_[t] = (level + trend) * s
                prev_level = level
                level = self.alpha * (x[t] / (s + 1e-12)) \
                    + (1 - self.alpha) * (level + trend)
                season[t % m] = self.gamma * (x[t] / (level + 1e-12)) \
                    + (1 - self.gamma) * s
            if self.trend:
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend

        self.level_, self.trend_, self.season_ = level, trend, season
        self._t = len(x)
        return self

    def forecast(self, steps=1):
        check_is_fitted(self, "level_")
        m = self.season_length
        out = np.empty(steps)
        for h in range(steps):
            s = self.season_[(self._t + h) % m]
            base = self.level_ + (h + 1) * self.trend_
            out[h] = base + s if self.seasonal == "additive" else base * s
        return out


def seasonal_decompose(x, season_length=12, model="additive"):
    """Classical decomposition into trend, seasonal and remainder.

    The straightforward version: a centred moving average of one season's length
    estimates the TREND (averaging over a full cycle cancels the season out); the
    detrended series, averaged by position in the cycle, gives the SEASON; what is
    left is the REMAINDER. Simple, fast, and the right first look at any seasonal
    series -- plot the three parts and the structure is laid bare.

    ``STL`` below is the robust, more flexible successor; this is the classical
    method it improves on, kept because seeing the plain version makes STL's
    additions legible.
    """
    x = np.asarray(x, float).ravel()
    m = season_length
    n = len(x)

    # centred moving average over one full season removes the seasonal component
    trend = np.full(n, np.nan)
    half = m // 2
    for t in range(half, n - half):
        if m % 2 == 0:
            # even period needs a half-weight on the two ends to stay centred
            window = x[t - half: t + half + 1].copy()
            window[0] *= 0.5
            window[-1] *= 0.5
            trend[t] = window.sum() / m
        else:
            trend[t] = x[t - half: t + half + 1].mean()

    detrended = (x - trend) if model == "additive" else (x / trend)
    # average each position across cycles to get the repeating pattern
    seasonal_means = np.array([np.nanmean(detrended[i::m]) for i in range(m)])
    if model == "additive":
        seasonal_means -= seasonal_means.mean()          # season sums to zero
    else:
        seasonal_means /= seasonal_means.mean()          # season averages to one
    seasonal = np.array([seasonal_means[t % m] for t in range(n)])

    remainder = (x - trend - seasonal) if model == "additive" \
        else (x / (trend * seasonal))
    return {"trend": trend, "seasonal": seasonal, "remainder": remainder,
            "observed": x}


class STL(BaseEstimator):
    """Seasonal-Trend decomposition using LOESS.

    THE IMPROVEMENT OVER CLASSICAL DECOMPOSITION
    --------------------------------------------
    Classical decomposition uses a moving average for the trend and a single fixed
    pattern for the season. STL replaces both with LOESS smooths, which buys two
    things that matter in practice:

    * **The seasonal pattern may EVOLVE.** Retail seasonality in 2020 is not
      2010's; a fixed pattern is forced to average them into something wrong for
      both. STL smooths the season across cycles, so it can drift.
    * **It is ROBUST to outliers.** A single spike distorts a moving-average trend
      for its whole window; LOESS's robustness iterations (see ``nonparametric``)
      downweight it, so one bad month does not smear across the decomposition.

    The mechanism is an inner loop that alternately smooths the season (over the
    values at each cycle position) and the trend (over the deseasonalised series)
    until they stop changing -- backfitting again, the same idea as the GAM.

    Cleveland, Cleveland, McRae & Terpenning (1990).
    """

    def __init__(self, season_length=12, n_iter=2, trend_frac=0.5,
                 seasonal_frac=0.75):
        self.season_length = season_length
        self.n_iter = n_iter
        self.trend_frac = trend_frac
        self.seasonal_frac = seasonal_frac

    def _loess_1d(self, y, frac):
        from ..nonparametric import LOESS
        t = np.arange(len(y)).reshape(-1, 1)
        ok = ~np.isnan(y)
        model = LOESS(frac=frac, degree=1, n_iter=2).fit(t[ok], y[ok])
        return model.predict(t)

    def fit(self, x):
        x = np.asarray(x, float).ravel()
        m = self.season_length
        n = len(x)

        # a first trend estimate BEFORE any season, or the per-position seasonal
        # smooths below would absorb the trend into themselves (each cycle
        # position's series still carries the rise). This is the detail that
        # makes the trend come out as trend rather than leaking into the season
        trend = self._loess_1d(x, self.trend_frac)
        seasonal = np.zeros(n)
        # backfitting: smooth the season, subtract it, smooth the trend, repeat
        for _ in range(self.n_iter):
            detrended = x - trend
            # smooth EACH cycle-position's series separately, so the pattern can
            # evolve across cycles rather than being one frozen shape
            for pos in range(m):
                idx = np.arange(pos, n, m)
                if len(idx) >= 3:
                    seasonal[idx] = self._loess_1d(detrended[idx],
                                                   self.seasonal_frac)
                else:
                    seasonal[idx] = detrended[idx].mean()
            seasonal -= seasonal.mean()          # keep the season mean-zero
            deseasonalised = x - seasonal
            trend = self._loess_1d(deseasonalised, self.trend_frac)

        self.trend_ = trend
        self.seasonal_ = seasonal
        self.remainder_ = x - trend - seasonal
        self.observed_ = x
        return self


__all__ = ["SimpleExponentialSmoothing", "Holt", "HoltWinters", "STL",
           "seasonal_decompose"]
