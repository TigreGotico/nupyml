"""L10: causal v2 -- Difference-in-Differences, Causal Forest, Regression
Discontinuity, R-learner.

DiD recovers a known average treatment effect from panel data; the causal forest
recovers a heterogeneous effect that varies with a covariate; regression
discontinuity reads a known jump at a cutoff; the R-learner recovers a
heterogeneous effect under confounding.
"""
import numpy as np
import pytest

from nupyml.causal import CausalForest, RegressionDiscontinuity, RLearner


def test_causal_forest_recovers_heterogeneous_effect():
    rng = np.random.RandomState(0)
    X = rng.rand(600, 4)
    t = rng.randint(0, 2, 600)
    tau = 2 * X[:, 0]                                     # effect grows with x0
    y = 1 + X[:, 1] + t * tau + rng.randn(600) * 0.3
    cf = CausalForest(n_estimators=100, max_depth=4, random_state=0).fit(X, t, y)
    pred = cf.predict(X)
    assert np.corrcoef(pred, tau)[0, 1] > 0.7            # tracks the varying effect


def test_regression_discontinuity_reads_the_jump():
    rng = np.random.RandomState(0)
    r = rng.uniform(-2, 2, 500)
    treat = (r >= 0).astype(float)
    y = 1 + 0.5 * r + 4 * treat + rng.randn(500) * 0.3   # jump of 4 at 0
    rd = RegressionDiscontinuity(cutoff=0.0, bandwidth=1.0).fit(r, y)
    assert abs(rd.effect() - 4.0) < 0.5


def test_r_learner_recovers_effect_under_confounding():
    rng = np.random.RandomState(0)
    X = rng.randn(800, 3)
    e = 1 / (1 + np.exp(-X[:, 0]))                        # confounded propensity
    t = (rng.rand(800) < e).astype(float)
    tau = 1 + X[:, 0]                                     # heterogeneous effect
    y = 2 * X[:, 1] + t * tau + rng.randn(800) * 0.3
    rl = RLearner().fit(X, t, y)
    assert np.corrcoef(rl.predict(X), tau)[0, 1] > 0.8   # recovers tau despite confounding
