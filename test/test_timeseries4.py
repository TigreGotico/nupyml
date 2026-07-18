"""K7: time series v4 -- SARIMA, TBATS, hierarchical reconciliation, N-BEATS.

SARIMA forecasts a seasonal series far below naive error; TBATS handles multiple
seasonalities on a positive series; reconciliation makes hierarchical forecasts
coherent (parts sum to the whole); N-BEATS forecasts a trend+seasonal series
below the naive baseline.
"""
import numpy as np
import pytest

from nupyml.timeseries import SARIMA, TBATS, reconcile_forecasts, NBeats


def test_sarima_forecasts_seasonal_series():
    rng = np.random.RandomState(0)
    m = 12
    t = np.arange(120)
    y = 0.2 * t + 10 * np.sin(2 * np.pi * t / m) + rng.randn(120) * 0.5
    train, test = y[:108], y[108:]
    sar = SARIMA(order=(1, 1, 0), seasonal_order=(1, 1, 0, 12)).fit(train)
    fc = sar.forecast(12)
    rmse = np.sqrt(np.mean((fc - test) ** 2))
    naive = np.sqrt(np.mean((train[-1] - test) ** 2))
    assert rmse < naive                                 # captures the season


def test_tbats_handles_multiple_seasonality():
    rng = np.random.RandomState(0)
    t = np.arange(200)
    y = 100 + 5 * np.sin(2 * np.pi * t / 7) + 3 * np.sin(2 * np.pi * t / 30) \
        + rng.randn(200) * 0.5
    tb = TBATS(periods=(7, 30), n_harmonics=3).fit(y)
    fc = tb.forecast(14)
    assert fc.shape == (14,)
    assert abs(fc.mean() - 100) < 5                     # stays around the level


@pytest.mark.parametrize("method", ["bottom_up", "ols", "mint"])
def test_reconciliation_makes_forecasts_coherent(method):
    # hierarchy: total = b1 + b2, summing matrix S
    S = np.array([[1, 1], [1, 0], [0, 1]])
    base = np.array([10.0, 4.0, 5.0])                   # incoherent: 4 + 5 != 10
    r = reconcile_forecasts(base, S, method=method)
    assert np.isclose(r[0], r[1] + r[2])               # coherent after reconciliation


def test_reconciliation_horizon_shape():
    S = np.array([[1, 1], [1, 0], [0, 1]])
    base = np.array([[10.0, 11.0], [4.0, 5.0], [5.0, 5.0]])   # (nodes, horizon)
    r = reconcile_forecasts(base, S, method="mint")
    assert r.shape == (3, 2)
    assert np.allclose(r[0], r[1] + r[2])


def test_nbeats_beats_naive():
    rng = np.random.RandomState(0)
    t = np.arange(200)
    y = 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.randn(200) * 0.3
    nb = NBeats(lookback=24, horizon=12, n_blocks=3, epochs=600,
                random_state=0).fit(y[:180])
    fc = nb.forecast()
    rmse = np.sqrt(np.mean((fc - y[180:192]) ** 2))
    naive = np.sqrt(np.mean((y[179] - y[180:192]) ** 2))
    assert rmse < naive
