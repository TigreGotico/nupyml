"""J1: probabilistic inference v2 -- NUTS, SVGD, expectation propagation, slice.

The three samplers must recover the mean AND covariance of a correlated Gaussian
target; SVGD's particles must do the same deterministically; EP must fit a probit
classifier with a valid (PSD) posterior covariance.
"""
import numpy as np
import pytest

from sklearn.datasets import make_classification

from nupyml.inference import (NUTS, SVGD, ExpectationPropagationClassifier,
                              SliceSampler)
from nupyml.model_selection import train_test_split
from nupyml.metrics import accuracy_score


MU = np.array([1.0, -2.0])
COV = np.array([[1.0, 0.8], [0.8, 1.0]])
PREC = np.linalg.inv(COV)


def _logp(x):
    return -0.5 * (x - MU) @ PREC @ (x - MU)


def _grad(x):
    return -PREC @ (x - MU)


def test_nuts_recovers_gaussian_moments():
    s = NUTS(_logp, _grad, step_size=0.25, random_state=0).sample(
        np.zeros(2), 2000, burn_in=500)
    assert np.allclose(s.mean(axis=0), MU, atol=0.2)
    c = np.cov(s.T)
    assert abs(c[0, 0] - 1.0) < 0.25 and abs(c[0, 1] - 0.8) < 0.25


def test_svgd_particles_match_gaussian():
    X = SVGD(_grad, n_particles=100, step_size=0.3, n_iter=800,
             random_state=0).sample(np.random.RandomState(0).randn(100, 2))
    assert np.allclose(X.mean(axis=0), MU, atol=0.2)
    assert abs(np.cov(X.T)[0, 1] - 0.8) < 0.3         # captures correlation


def test_slice_sampler_recovers_gaussian_moments():
    s = SliceSampler(_logp, width=2.0, random_state=0).sample(
        np.zeros(2), 2000, burn_in=300)
    assert np.allclose(s.mean(axis=0), MU, atol=0.2)
    assert abs(np.cov(s.T)[0, 0] - 1.0) < 0.25


def test_expectation_propagation_classifies_with_psd_posterior():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ep = ExpectationPropagationClassifier().fit(Xtr, ytr)
    assert accuracy_score(yte, ep.predict(Xte)) > 0.7
    # the EP posterior covariance is a valid covariance (positive semi-definite)
    assert np.all(np.linalg.eigvalsh(ep.cov_) > -1e-8)
    proba = ep.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_svgd_single_particle_finds_mode():
    # one particle has no repulsion -> plain gradient ascent to the MAP (the mean)
    X = SVGD(_grad, n_particles=1, step_size=0.2, n_iter=500,
             random_state=0).sample(np.zeros((1, 2)))
    assert np.allclose(X[0], MU, atol=0.1)
