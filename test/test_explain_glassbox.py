"""F3 interpretability: EBM glass box, ALE, H-statistic, surrogate, anchors, CF.

Each is held to its defining property: the EBM recovers a known additive function
and its shape functions match the true per-feature effects; ALE is unbiased and
monotonic where the effect is; the H-statistic separates an additive model (~0)
from a pure interaction (~1); the surrogate reports high fidelity on a
tree-approximable box; an anchor rule is high-precision; a counterfactual flips
the prediction.
"""
import numpy as np
import pytest

from nupyml.metrics import r2_score
from nupyml.datasets import make_classification
from nupyml.ensemble import RandomForestClassifier
from nupyml.explain import (ExplainableBoostingRegressor,
                           ExplainableBoostingClassifier,
                           accumulated_local_effects, h_statistic,
                           surrogate_tree, Anchors, counterfactual)


@pytest.fixture
def additive_data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (800, 3))
    # feature 0 quadratic, feature 1 linear, feature 2 irrelevant
    y = X[:, 0] ** 2 + 0.8 * X[:, 1] + 0.1 * rng.randn(800)
    return X, y


# --- EBM ------------------------------------------------------------------

def test_ebm_recovers_additive_function(additive_data):
    X, y = additive_data
    ebm = ExplainableBoostingRegressor(n_rounds=150, random_state=0).fit(X, y)
    assert r2_score(y, ebm.predict(X)) > 0.95


def test_ebm_shape_function_matches_true_effect(additive_data):
    X, y = additive_data
    ebm = ExplainableBoostingRegressor(n_rounds=150, random_state=0).fit(X, y)
    # feature 0's shape should be U-shaped (quadratic): ends high, middle low
    edges0, shape0 = ebm.explain_global(0)
    assert shape0[0] > shape0[len(shape0) // 2]
    assert shape0[-1] > shape0[len(shape0) // 2]
    # feature 1's shape should be monotonically increasing (linear effect)
    _, shape1 = ebm.explain_global(1)
    assert shape1[-1] > shape1[0]
    # feature 2 is irrelevant: its shape should be nearly flat
    _, shape2 = ebm.explain_global(2)
    assert shape2.std() < shape1.std()


def test_ebm_classifier_beats_chance():
    X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                               random_state=0)
    ebm = ExplainableBoostingClassifier(n_rounds=120, random_state=0).fit(X, y)
    assert (ebm.predict(X) == y).mean() > 0.8
    p = ebm.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)


# --- ALE ------------------------------------------------------------------

def test_ale_is_monotonic_for_a_monotonic_effect(additive_data):
    X, y = additive_data
    ebm = ExplainableBoostingRegressor(n_rounds=150, random_state=0).fit(X, y)
    _, ale = accumulated_local_effects(ebm, X, feature=1)   # linear-increasing
    assert np.all(np.diff(ale) > -0.05)                     # (up to noise)
    assert ale[-1] > ale[0]


# --- H-statistic ----------------------------------------------------------

def test_h_statistic_separates_additive_from_interaction(additive_data):
    X, y = additive_data
    ebm = ExplainableBoostingRegressor(n_rounds=150, random_state=0).fit(X, y)
    # an EBM is additive by construction -> H ~ 0
    assert h_statistic(ebm, X, 0, 1) < 0.1

    class Mult:                                     # pure interaction y = x0*x1
        def predict(self, Z):
            return Z[:, 0] * Z[:, 1]
    assert h_statistic(Mult(), X, 0, 1) > 0.8


# --- surrogate / anchors / counterfactual ---------------------------------

@pytest.fixture
def rf_box():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                               random_state=0)
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    return rf, X, y


def test_surrogate_tree_has_high_fidelity(rf_box):
    rf, X, y = rf_box
    tree, fidelity = surrogate_tree(rf, X, max_depth=5)
    assert fidelity > 0.85                          # the tree mimics the forest
    # fidelity is agreement with the BOX, not the truth
    assert np.mean(tree.predict(X) == rf.predict(X)) == pytest.approx(fidelity)


def test_anchor_rule_is_high_precision(rf_box):
    rf, X, y = rf_box
    anchor = Anchors(rf, threshold=0.9, random_state=0).explain(X[0], X)
    assert anchor.precision_ >= 0.9                 # the rule reliably implies the label
    assert 0 < anchor.coverage_ <= 1
    assert len(anchor.rule_) >= 1


def test_counterfactual_flips_the_prediction(rf_box):
    rf, X, y = rf_box
    orig = rf.predict(X[0:1])[0]
    cf, dist = counterfactual(rf, X[0], X, random_state=0)
    assert cf is not None
    assert rf.predict(cf.reshape(1, -1))[0] != orig
    assert dist > 0
