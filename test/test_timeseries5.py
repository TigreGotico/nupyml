"""L7: time series v5 -- VECM, dynamic factor model, Croston SBA/TSB, Kalman-EM,
DeepAR.

VECM recovers a cointegration relationship; the dynamic factor model recovers the
common factor of many series; Croston SBA corrects the bias and TSB fades on
obsolescence; Kalman-EM learns the state-space noise variances; DeepAR's intervals
cover the future.
"""
import numpy as np
import pytest

from nupyml.timeseries import (VECM, DynamicFactorModel, CrostonSBA, CrostonTSB,
                               KalmanEM, DeepAR)


def test_vecm_recovers_cointegration():
    rng = np.random.RandomState(0)
    T = 200
    trend = np.cumsum(rng.randn(T) * 0.5)
    Y = np.column_stack([trend + rng.randn(T) * 0.3,
                         trend + 2 + rng.randn(T) * 0.3])
    v = VECM().fit(Y)
    # the cointegration vector is ~ (1, -1): the two series move together
    assert abs(v.beta_[0] - 1.0) < 0.1 and abs(v.beta_[1] + 1.0) < 0.15
    fc = v.forecast(10)
    assert fc.shape == (10, 2)


def test_dynamic_factor_model_recovers_common_factor():
    rng = np.random.RandomState(0)
    factor = np.cumsum(rng.randn(150) * 0.4)
    loadings = rng.uniform(0.5, 1.5, 10)
    Y = np.outer(factor, loadings) + rng.randn(150, 10) * 0.3
    dfm = DynamicFactorModel(n_factors=1).fit(Y)
    assert abs(np.corrcoef(dfm.factors_[:, 0], factor)[0, 1]) > 0.95


def test_croston_sba_estimates_the_rate():
    rng = np.random.RandomState(0)
    y = np.where(rng.rand(200) < 0.2, rng.poisson(4, 200) + 1, 0).astype(float)
    sba = CrostonSBA(alpha=0.1).fit(y)
    assert abs(sba.rate_ - y.mean()) < 0.3               # near the true demand rate


def test_croston_tsb_fades_on_obsolescence():
    rng = np.random.RandomState(0)
    y = np.where(rng.rand(200) < 0.2, rng.poisson(4, 200) + 1, 0).astype(float)
    active = CrostonTSB(alpha=0.1, beta=0.1).fit(y).rate_
    y_obs = y.copy(); y_obs[100:] = 0                    # demand stops halfway
    faded = CrostonTSB(alpha=0.1, beta=0.1).fit(y_obs).rate_
    assert faded < active * 0.5                          # forecast decays toward zero


def test_kalman_em_learns_noise_variances():
    rng = np.random.RandomState(0)
    mu = np.cumsum(rng.randn(400) * 0.3)                 # process var ~ 0.09
    obs = mu + rng.randn(400) * 1.0                      # obs var ~ 1.0
    km = KalmanEM().fit(obs)
    assert abs(km.process_var_ - 0.09) < 0.1
    assert abs(km.obs_var_ - 1.0) < 0.3
    assert np.sqrt(np.mean((km.level_ - mu) ** 2)) < 0.6  # tracks the latent level


def test_deepar_intervals_cover_the_future():
    rng = np.random.RandomState(0)
    t = np.arange(300)
    y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 20) + rng.randn(300) * 0.5
    dar = DeepAR(lookback=20, epochs=300, random_state=0).fit(y[:280])
    mean, paths = dar.forecast(20, n_samples=200, random_state=1)
    lo, hi = np.percentile(paths, [5, 95], axis=0)
    coverage = np.mean((y[280:300] >= lo) & (y[280:300] <= hi))
    assert coverage >= 0.6                               # intervals cover the majority
    assert np.all(hi > lo)                               # genuine predictive spread
