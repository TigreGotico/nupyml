"""M1: probabilistic v5 -- nested sampling, DPP, Bayesian quadrature, sparse GP.

Nested sampling recovers the Bayesian evidence of a Gaussian likelihood; the DPP
samples more diverse subsets than random; Bayesian quadrature recovers a known
integral with error bars; the sparse variational GP fits a regression with
uncertainty using few inducing points.
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from nupyml.inference import (NestedSampling, DeterminantalPointProcess,
                              BayesianQuadrature, SparseVariationalGP)


def test_nested_sampling_recovers_evidence():
    def loglik(x):
        return -0.5 * x[0] ** 2 - 0.5 * np.log(2 * np.pi)
    ns = NestedSampling(loglik, bounds=([-5], [5]), n_live=200,
                        random_state=0).run()
    # prior width 10, so Z = ∫ N(0,1)/10 ≈ 1/10 -> log Z ≈ -2.303
    assert abs(ns.log_evidence_ - np.log(0.1)) < 0.3
    assert abs(ns.posterior_mean_[0]) < 0.3             # posterior centred at 0


def test_dpp_samples_diverse_subsets():
    rng = np.random.RandomState(0)
    pts = np.vstack([rng.randn(5, 2) * 0.1 + [0, 0],
                     rng.randn(5, 2) * 0.1 + [3, 3],
                     rng.randn(5, 2) * 0.1 + [0, 3]])
    L = np.exp(-cdist(pts, pts, "sqeuclidean") / 1.0)
    dpp, rand = [], []
    for s in range(30):
        sel = DeterminantalPointProcess(L, random_state=s).sample()
        if len(sel) >= 2:
            dpp.append(np.mean(cdist(pts[sel], pts[sel])))
            r = np.random.RandomState(s).choice(15, len(sel), replace=False)
            rand.append(np.mean(cdist(pts[r], pts[r])))
    assert np.mean(dpp) > np.mean(rand)                 # DPP spreads its picks out


def test_bayesian_quadrature_recovers_integral():
    xs = np.linspace(0, 1, 15).reshape(-1, 1)
    ys = xs.ravel() ** 2
    bq = BayesianQuadrature(length_scale=0.2).fit(xs, ys, a=0, b=1)
    integral, std = bq.integral(return_std=True)
    assert abs(integral - 1 / 3) < 0.01                 # ∫_0^1 x^2 dx = 1/3
    assert std >= 0


def test_sparse_gp_fits_with_few_inducing_points():
    rng = np.random.RandomState(0)
    X = np.linspace(-4, 4, 200).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * rng.randn(200)
    svgp = SparseVariationalGP(n_inducing=12, length_scale=1.0, noise=0.1,
                               random_state=0).fit(X, y)
    Xt = np.linspace(-4, 4, 50).reshape(-1, 1)
    mean, std = svgp.predict(Xt, return_std=True)
    assert np.mean((mean - np.sin(Xt).ravel()) ** 2) < 0.02   # fits with 12 points
    assert np.all(std > 0)                              # predictive uncertainty
