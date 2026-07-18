"""I4: time series v2 -- Croston, Theta, SSA, SAX, DBA, Fourier features.

Croston forecasts an intermittent series' rate (beating naive-last); Theta
extrapolates a trend; SSA reconstructs a trend+seasonal signal from noise; SAX
symbolises; DBA averages under warping without smearing amplitude; Fourier
features are periodic.
"""
import numpy as np
import pytest

from nupyml.timeseries import (Croston, Theta, SSA, sax,
                              dtw_barycenter_averaging, fourier_features)


def test_croston_recovers_rate_and_beats_naive_on_average():
    cro_err, naive_err = [], []
    for seed in range(10):                             # average over random series
        rng = np.random.RandomState(seed)
        y = np.where(rng.rand(200) < 0.25, rng.poisson(3, 200) + 1, 0).astype(float)
        train, test = y[:180], y[180:]
        cro = Croston(alpha=0.1).fit(train).predict(len(test))
        assert abs(cro[0] - train.mean()) < 1.0        # forecasts near the demand rate
        cro_err.append(np.mean((cro - test) ** 2))
        naive_err.append(np.mean((np.full(len(test), train[-1]) - test) ** 2))
    # Croston's rate forecast beats repeating the (often-zero) last value on average
    assert np.mean(cro_err) < np.mean(naive_err)


def test_theta_extrapolates_trend():
    rng = np.random.RandomState(0)
    t = np.arange(100)
    y = 0.5 * t + 5 + rng.randn(100)
    th = Theta().fit(y)
    fc = th.predict(10)
    assert fc[-1] > fc[0] > y[0]                        # continues the upward trend
    assert abs(fc[0] - 55) < 5                          # ~ 0.5*100 + 5


def test_ssa_reconstructs_signal_from_noise():
    rng = np.random.RandomState(0)
    t = np.arange(120)
    clean = 0.1 * t + 3 * np.sin(2 * np.pi * t / 20)
    noisy = clean + rng.randn(120) * 0.5
    ssa = SSA(window=40, n_components=3).fit(noisy)
    # the low-rank reconstruction is closer to the clean signal than the noisy input
    assert (np.linalg.norm(ssa.reconstruction_ - clean)
            < np.linalg.norm(noisy - clean))
    assert np.corrcoef(ssa.reconstruction_, clean)[0, 1] > 0.95


def test_sax_symbolises_consistently():
    y = np.sin(np.linspace(0, 4 * np.pi, 64))
    s = sax(y, n_segments=8, alphabet_size=4)
    assert len(s) == 8
    assert set(s) <= set("abcd")
    # the same series scaled/shifted gives the same symbols (z-normalised)
    assert sax(2 * y + 10, n_segments=8, alphabet_size=4) == s


def test_dba_preserves_shape_better_than_pointwise_mean():
    base = np.sin(np.linspace(0, 2 * np.pi, 50))
    series = [np.roll(base, k) for k in (-4, 0, 4)]
    dba = dtw_barycenter_averaging(series, n_iter=5)
    pointwise = np.mean(series, axis=0)
    # warping alignment preserves the oscillation amplitude the pointwise mean cancels
    assert (dba.max() - dba.min()) >= (pointwise.max() - pointwise.min())


def test_fourier_features_are_periodic():
    ff = fourier_features(100, period=20, n_harmonics=2)
    assert ff.shape == (100, 4)
    # row t and row t+period are identical (the basis has that period)
    assert np.allclose(ff[0], ff[20], atol=1e-9)
