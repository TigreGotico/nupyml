"""H10: spectral-mixture kernel, Student-t process, deep-kernel GP.

The spectral kernel must fit (and extrapolate) a periodic function where RBF
reverts to the mean; the Student-t process must match the GP mean and recover the
GP as nu grows; the deep-kernel GP must fit a nonlinear regression.
"""
import numpy as np
import pytest

from nupyml.gaussian_process import (GaussianProcessRegressor, RBF,
                                     SpectralMixtureKernel, StudentTProcess,
                                     DeepKernelGP)
from nupyml.metrics import r2_score


def test_spectral_mixture_fits_and_extrapolates_periodic():
    rng = np.random.RandomState(0)
    X = np.linspace(0, 4, 50).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel()) + 0.05 * rng.randn(50)
    smk = SpectralMixtureKernel(means=[1.0], variances=[0.3], weights=[1.0],
                                n_components=1)
    gp = GaussianProcessRegressor(kernel=smk, alpha=0.01, optimize=False).fit(X, y)
    assert r2_score(y, gp.predict(X)) > 0.9
    # extrapolate one period beyond the training range
    Xe = np.linspace(4, 5, 20).reshape(-1, 1)
    ye = np.sin(2 * np.pi * Xe.ravel())
    smk_err = np.mean((gp.predict(Xe) - ye) ** 2)
    rbf = GaussianProcessRegressor(kernel=RBF(0.2), alpha=0.01,
                                   optimize=False).fit(X, y)
    rbf_err = np.mean((rbf.predict(Xe) - ye) ** 2)   # RBF reverts to the mean
    assert smk_err < rbf_err


def test_student_t_process_matches_gp_and_recovers_it():
    rng = np.random.RandomState(0)
    X = np.linspace(-3, 3, 50).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.05 * rng.randn(50)
    tp = StudentTProcess(kernel=RBF(1.0), alpha=0.05, nu=4).fit(X, y)
    gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.05,
                                  optimize=False).fit(X, y)
    mt, st = tp.predict(X, return_std=True)
    mg, sg = gp.predict(X, return_std=True)
    assert np.allclose(mt, mg, atol=1e-6)            # same closed-form mean
    assert r2_score(y, mt) > 0.9
    # as nu -> infinity the t-process std collapses to the GP std
    tp_big = StudentTProcess(kernel=RBF(1.0), alpha=0.05, nu=1e6).fit(X, y)
    _, s_big = tp_big.predict(X, return_std=True)
    assert np.allclose(s_big, sg, rtol=0.05)


def test_deep_kernel_gp_fits_nonlinear_regression():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (200, 3))
    y = np.sin(X[:, 0] * 3) + X[:, 1] ** 2
    dk = DeepKernelGP(epochs=200, random_state=0).fit(X, y)
    pred, std = dk.predict(X, return_std=True)
    assert r2_score(y, pred) > 0.9
    assert np.all(std >= 0)                           # GP still gives uncertainty
