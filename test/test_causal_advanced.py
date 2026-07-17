"""G7: IV, double ML, CATE learners, panel methods, and DAG discovery.

Each recovers a KNOWN causal quantity from synthetic data with a planted truth:
IV beats biased OLS under unobserved confounding; double ML recovers the ATE with
ML nuisances; the meta-learners recover a heterogeneous effect; DiD and synthetic
control recover a panel effect; NOTEARS recovers the generating DAG.
"""
import numpy as np
import pytest

from nupyml.causal import (InstrumentalVariables, DoubleML, TLearner, XLearner,
                          difference_in_differences, SyntheticControl,
                          notears_linear)


def test_instrumental_variables_beats_biased_ols():
    rng = np.random.RandomState(0)
    n = 2000
    U = rng.randn(n)                                  # unobserved confounder
    Z = rng.randn(n)                                  # instrument
    T = 0.8 * Z + 0.6 * U + 0.3 * rng.randn(n)
    y = 2.0 * T + 1.5 * U + 0.3 * rng.randn(n)        # true effect = 2.0
    iv = InstrumentalVariables().fit(Z.reshape(-1, 1), T, y)
    ols = np.polyfit(T, y, 1)[0]
    assert abs(iv.coef_ - 2.0) < 0.15                 # IV recovers the truth
    assert abs(ols - 2.0) > abs(iv.coef_ - 2.0)       # OLS is more biased


def test_double_ml_recovers_ate_with_ml_nuisances():
    rng = np.random.RandomState(0)
    n = 2000
    X = rng.randn(n, 4)
    g = X @ np.array([1.0, 0.5, -0.5, 0.0])
    T = g + rng.randn(n)
    y = 1.5 * T + g + 0.3 * rng.randn(n)              # true ATE = 1.5
    dml = DoubleML(cv=2, random_state=0).fit(X, T, y)
    assert abs(dml.effect_ - 1.5) < 0.15


@pytest.mark.parametrize("Learner", [TLearner, XLearner])
def test_cate_learner_recovers_heterogeneous_effect(Learner):
    rng = np.random.RandomState(0)
    n = 2000
    X = rng.randn(n, 3)
    T = rng.randint(0, 2, n)
    tau = 1.0 + 2.0 * (X[:, 0] > 0)                   # effect depends on X0
    y = X[:, 1] + T * tau + 0.2 * rng.randn(n)
    est = Learner().fit(X, T, y)
    pred = est.predict_effect(X)
    assert np.corrcoef(pred, tau)[0, 1] > 0.8         # recovers the heterogeneity


def test_difference_in_differences_removes_common_trend():
    # control rises +2, treated rises +7 -> the excess +5 is the effect
    effect = difference_in_differences(y_control_pre=[10, 10], y_control_post=[12, 12],
                                       y_treated_pre=[20, 20], y_treated_post=[27, 27])
    assert effect == pytest.approx(5.0)


def test_synthetic_control_weights_are_a_convex_combination():
    rng = np.random.RandomState(0)
    # the treated pre-trend equals control 0 exactly -> weight concentrates there
    ctrl = np.column_stack([np.arange(20) + rng.randn(20) * 0.05,
                            np.arange(20)[::-1] * 1.0, np.ones(20)])
    treated = ctrl[:, 0].copy()
    sc = SyntheticControl().fit(treated, ctrl)
    assert np.isclose(sc.weights_.sum(), 1.0, atol=1e-3)
    assert np.all(sc.weights_ >= -1e-6)               # convex (non-negative)
    assert sc.weights_[0] > 0.8                       # weight concentrates on the match
    # the synthetic reproduces the treated unit's PRE-period trajectory well
    synthetic_pre = ctrl @ sc.weights_
    assert np.sqrt(np.mean((synthetic_pre - treated) ** 2)) < 1.0


def test_notears_recovers_the_generating_dag():
    rng = np.random.RandomState(0)
    n = 600
    X = np.zeros((n, 3))
    X[:, 0] = rng.randn(n)
    X[:, 1] = 2.0 * X[:, 0] + 0.3 * rng.randn(n)      # 0 -> 1
    X[:, 2] = 1.5 * X[:, 1] + 0.3 * rng.randn(n)      # 1 -> 2
    W = notears_linear(X, lambda1=0.05)
    edges = {(i, j) for i in range(3) for j in range(3) if abs(W[i, j]) > 0.3}
    assert (0, 1) in edges and (1, 2) in edges        # recovered the chain
    # acyclic: no mutual edge
    assert not any((j, i) in edges for (i, j) in edges)
    assert (0, 2) not in edges                        # no spurious direct edge
