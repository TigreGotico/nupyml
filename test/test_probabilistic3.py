"""K1: probabilistic v3 -- Probabilistic PCA, RVM, Bayesian NN, structural TS.

PPCA recovers the noise variance and scores a proper likelihood; the RVM fits with
FEW relevance vectors and returns error bars; the MC-dropout net's uncertainty
grows in extrapolation; the structural time series decomposes trend+seasonal and
forecasts with widening intervals.
"""
import numpy as np
import pytest

from nupyml.decomposition import ProbabilisticPCA
from nupyml.inference import RelevanceVectorMachine, BayesianNeuralNetwork
from nupyml.timeseries import BayesianStructuralTimeSeries


def test_probabilistic_pca_recovers_noise_and_scores():
    rng = np.random.RandomState(0)
    Z = rng.randn(300, 2)
    W = rng.randn(2, 5)
    X = Z @ W + 0.1 * rng.randn(300, 5)                 # 2-D subspace + noise 0.01
    ppca = ProbabilisticPCA(n_components=2).fit(X)
    assert abs(ppca.noise_variance_ - 0.01) < 0.01
    assert ppca.transform(X).shape == (300, 2)
    # in-distribution data scores far higher than off-distribution data
    assert ppca.score(X) > ppca.score(rng.randn(300, 5) * 3)


def test_rvm_is_sparse_and_gives_error_bars():
    rng = np.random.RandomState(0)
    X = np.linspace(-5, 5, 80).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * rng.randn(80)
    rvm = RelevanceVectorMachine(gamma=0.5).fit(X, y)
    assert rvm.n_relevance_ < 20                         # far fewer than 80 points
    assert np.mean((rvm.predict(X) - y) ** 2) < 0.05     # still fits well
    mean, std = rvm.predict(np.linspace(-5, 5, 40).reshape(-1, 1), return_std=True)
    assert np.all(std > 0)                               # a predictive distribution


def test_bayesian_nn_uncertainty_grows_out_of_distribution():
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = np.sin(X).ravel()
    bnn = BayesianNeuralNetwork(hidden=(32, 32), dropout=0.1, epochs=500,
                                n_samples=50, random_state=0).fit(X, y)
    _, std_in = bnn.predict(np.linspace(-2, 2, 20).reshape(-1, 1), return_std=True)
    _, std_out = bnn.predict(np.linspace(5, 8, 20).reshape(-1, 1), return_std=True)
    assert std_out.mean() > std_in.mean()               # honest extrapolation error


def test_structural_time_series_decomposes_and_forecasts():
    rng = np.random.RandomState(0)
    t = np.arange(120)
    y = 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.randn(120) * 0.5
    bsts = BayesianStructuralTimeSeries(season_length=12).fit(y)
    assert abs(bsts.slope_[-1] - 0.3) < 0.2             # recovers the trend slope
    mean, std = bsts.forecast(24, return_std=True)
    assert mean[-1] > mean[0]                           # continues the upward trend
    assert std[-1] > std[0]                             # uncertainty widens with horizon
