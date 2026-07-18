"""L1: probabilistic v4 -- Ensemble Kalman Filter, Markov switching, Bayesian PMF,
Rao-Blackwellised particle filter.

EnKF denoises a noisy random walk; the Markov-switching model recovers two regimes;
Bayesian PMF reconstructs a low-rank matrix with a predictive spread; the RBPF
tracks a jump-Markov linear state and infers its mode.
"""
import numpy as np
import pytest
from itertools import permutations

from nupyml.inference import (EnsembleKalmanFilter, MarkovSwitchingModel,
                              BayesianPMF, RaoBlackwellisedParticleFilter)


def test_ensemble_kalman_filter_denoises():
    rng = np.random.RandomState(0)
    n = 60
    true = np.cumsum(rng.randn(n) * 0.5)
    obs = true + rng.randn(n) * 0.5
    enkf = EnsembleKalmanFilter(transition=np.eye(1), observation=np.eye(1),
                                process_cov=[[0.25]], obs_cov=[[0.25]],
                                n_ensemble=50, random_state=0)
    est = enkf.filter(obs.reshape(-1, 1), x0=[0.0]).ravel()
    assert np.sqrt(np.mean((est - true) ** 2)) < np.sqrt(np.mean((obs - true) ** 2))


def test_markov_switching_recovers_regimes():
    rng = np.random.RandomState(0)
    regimes = np.repeat([0, 1, 0, 1], 30)
    y = np.where(regimes == 0, rng.randn(120) * 0.3 - 2, rng.randn(120) * 0.3 + 2)
    ms = MarkovSwitchingModel(n_regimes=2, random_state=0).fit(y)
    pred = ms.smoothed_.argmax(1)
    acc = max(np.mean(np.array([p[r] for r in pred]) == regimes)
              for p in permutations(range(2)))
    assert acc > 0.9
    assert abs(np.sort(ms.means_)[0] - (-2)) < 0.5       # recovers the regime means


def test_bayesian_pmf_reconstructs_low_rank():
    rng = np.random.RandomState(0)
    U = rng.randn(20, 3); V = rng.randn(15, 3)
    R_full = U @ V.T
    R = R_full.copy()
    R[rng.rand(20, 15) < 0.4] = np.nan                   # 40% missing
    bpmf = BayesianPMF(n_factors=3, n_samples=40, burn_in=15, random_state=0).fit(R)
    missing = np.isnan(R)
    mae = np.mean(np.abs(bpmf.mean_[missing] - R_full[missing]))
    assert mae < 0.8                                     # recovers held-out entries
    mean, std = bpmf.predict(0, 0)
    assert std > 0                                       # a predictive distribution


def test_rbpf_tracks_state_and_mode():
    rng = np.random.RandomState(0)
    A = np.array([[1.0]])
    modes = [np.array([[1.0]]), np.array([[0.2]])]       # mode 1 attenuates the signal
    true = np.cumsum(rng.randn(40) * 0.3)
    true_mode = (np.arange(40) // 20)
    ys = np.array([modes[true_mode[t]][0, 0] * true[t] + rng.randn() * 0.2
                   for t in range(40)])
    rbpf = RaoBlackwellisedParticleFilter(A, modes, process_cov=[[0.09]],
                                          obs_cov=[[0.04]], n_particles=100,
                                          random_state=0)
    est = rbpf.filter(ys.reshape(-1, 1), x0=[0.0], P0=[[1.0]]).ravel()
    assert np.sqrt(np.mean((est - true) ** 2)) < 0.6     # tracks the linear state
    assert np.mean(rbpf.mode_estimates_ == true_mode) > 0.6   # infers the mode
