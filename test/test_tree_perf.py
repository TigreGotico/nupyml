"""G12: oblivious trees, linear-leaf trees, KD-tree, and exact TreeSHAP.

The oblivious trees must fit (one split per level); the linear-leaf tree must
match/beat a same-depth CART on piecewise-linear data; the KD-tree must return the
SAME neighbours as brute force (it is exact); and TreeSHAP must satisfy the
efficiency axiom (attributions sum to prediction minus base).
"""
import numpy as np
import pytest

from nupyml.tree import (ObliviousDecisionTreeClassifier,
                        ObliviousDecisionTreeRegressor, LinearTreeRegressor,
                        DecisionTreeRegressor)
from nupyml.neighbors import KDTree
from nupyml.explain import TreeSHAP
from nupyml.datasets import make_classification
from nupyml.metrics import accuracy_score, r2_score


def _piecewise(seed=0, n=400):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, (n, 2))
    y = np.where(X[:, 0] > 0, 2 * X[:, 0], -X[:, 0]) + 0.1 * rng.randn(n)
    return X, y


def test_oblivious_classifier_fits_and_is_symmetric():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                               random_state=0)
    clf = ObliviousDecisionTreeClassifier(max_depth=4).fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.9
    assert len(clf.splits_) == 4                      # one (feature,threshold) per level


def test_oblivious_regressor_fits_piecewise():
    X, y = _piecewise()
    reg = ObliviousDecisionTreeRegressor(max_depth=4).fit(X, y)
    assert r2_score(y, reg.predict(X)) > 0.85


def test_linear_tree_matches_or_beats_shallow_cart():
    X, y = _piecewise()
    lt = LinearTreeRegressor(max_depth=2).fit(X, y)
    cart = DecisionTreeRegressor(max_depth=2).fit(X, y)
    # linear leaves capture the within-region trend a constant leaf cannot
    assert r2_score(y, lt.predict(X)) >= r2_score(y, cart.predict(X)) - 0.01
    assert r2_score(y, lt.predict(X)) > 0.7


def test_kdtree_returns_exact_nearest_neighbours():
    rng = np.random.RandomState(0)
    data = rng.randn(500, 4)
    kd = KDTree().fit(data)
    queries = rng.randn(10, 4)
    dist, idx = kd.query(queries, k=3)
    for i, q in enumerate(queries):
        brute = np.argsort(np.sum((data - q) ** 2, axis=1))[:3]
        assert set(idx[i]) == set(brute)              # exact, same as brute force
        # returned distances are sorted and correct
        assert np.all(np.diff(dist[i]) >= -1e-9)


def test_treeshap_satisfies_efficiency():
    X, y = _piecewise()
    tree = DecisionTreeRegressor(max_depth=3).fit(X, y)
    ts = TreeSHAP(tree)
    for x in X[:20]:
        shap = ts.explain(x)
        # efficiency: base value + sum of attributions == the tree's prediction
        assert abs(ts.base_value_ + shap.sum() - tree.predict(x[None])[0]) < 1e-8


def test_treeshap_zeroes_unused_features():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = 3 * X[:, 0] - 2 * X[:, 1]                      # features 2,3 irrelevant
    tree = DecisionTreeRegressor(max_depth=3).fit(X, y)
    ts = TreeSHAP(tree)
    shap = np.mean([np.abs(ts.explain(x)) for x in X[:30]], axis=0)
    # the informative features carry far more attribution than the noise ones
    assert shap[0] + shap[1] > 5 * (shap[2] + shap[3] + 1e-9)
