"""J7: time series v3 -- GARCH, VAR, AutoETS, ProphetForecaster, MSTL.

GARCH recovers the volatility parameters and forecasts variance reverting to the
unconditional level; VAR recovers the coupling coefficients of two interacting
series; AutoETS selects trend/season and beats naive; Prophet fits a bent trend;
MSTL separates two seasonal cycles and reconstructs exactly.
"""
import numpy as np
import pytest

from nupyml.timeseries import (GARCH, VAR, AutoETS, ProphetForecaster, MSTL)


def test_garch_recovers_params_and_variance_reverts():
    rng = np.random.RandomState(0)
    n = 2000
    eps = np.zeros(n); var = np.zeros(n); var[0] = 1.0
    om, al, be = 0.1, 0.15, 0.8
    for t in range(1, n):
        var[t] = om + al * eps[t - 1] ** 2 + be * var[t - 1]
        eps[t] = np.sqrt(var[t]) * rng.randn()
    g = GARCH().fit(eps)
    assert abs(g.omega_ - om) < 0.06
    assert abs(g.alpha_ - al) < 0.06 and abs(g.beta_ - be) < 0.06
    uncond = g.omega_ / (1 - g.alpha_ - g.beta_)
    fc = g.forecast(50)
    assert abs(fc[-1] - uncond) < abs(fc[0] - uncond)   # reverts toward uncond


def test_var_recovers_coupling():
    rng = np.random.RandomState(0)
    T = 500
    Y = np.zeros((T, 2)); Y[0] = [1, 0]
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + 0.3 * Y[t - 1, 1] + 0.1 * rng.randn()
        Y[t, 1] = -0.2 * Y[t - 1, 0] + 0.6 * Y[t - 1, 1] + 0.1 * rng.randn()
    v = VAR(p=1).fit(Y)
    C = v.coef_[1:]                                     # drop intercept row
    assert abs(C[0, 0] - 0.5) < 0.12 and abs(C[1, 0] - 0.3) < 0.12
    assert abs(C[0, 1] - (-0.2)) < 0.12 and abs(C[1, 1] - 0.6) < 0.12
    assert v.forecast(5).shape == (5, 2)


def test_autoets_selects_trend_and_beats_naive():
    rng = np.random.RandomState(0)
    t = np.arange(60, dtype=float)
    y = 0.5 * t + 5 + rng.randn(60) * 0.5
    train, test = y[:50], y[50:]
    e = AutoETS(season_length=1).fit(train)
    assert e.trend_                                      # trend model selected
    fc = e.forecast(10)
    assert fc[-1] > fc[0]                                # forecast keeps rising
    naive = np.full(10, train[-1])
    assert np.mean((fc - test) ** 2) < np.mean((naive - test) ** 2)


def test_autoets_selects_seasonality():
    rng = np.random.RandomState(0)
    m = 12
    t = np.arange(120)
    y = 20 + 10 * np.sin(2 * np.pi * t / m) + rng.randn(120)
    e = AutoETS(season_length=m).fit(y)
    assert e.season_                                     # seasonal model selected


def test_prophet_fits_bent_trend():
    rng = np.random.RandomState(0)
    y = np.concatenate([np.arange(50) * 1.0,
                        50 + np.arange(50) * (-0.5)]) + rng.randn(100) * 0.5
    p = ProphetForecaster(n_changepoints=10).fit(y)
    fit = p.predict(np.arange(100))
    r2 = 1 - np.var(fit - y) / np.var(y)
    assert r2 > 0.95                                     # captures the slope change


def test_mstl_separates_two_seasonalities():
    rng = np.random.RandomState(0)
    t = np.arange(210)
    y = (0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
         + 2 * np.sin(2 * np.pi * t / 30) + rng.randn(210) * 0.3)
    ms = MSTL(periods=[7, 30]).fit(y)
    recon = ms.trend_ + ms.seasonal_[7] + ms.seasonal_[30] + ms.remainder_
    assert np.allclose(recon, y, atol=1e-6)             # exact decomposition
    # each seasonal component actually oscillates at its period
    assert ms.seasonal_[7].std() > 1.0 and ms.seasonal_[30].std() > 0.5
