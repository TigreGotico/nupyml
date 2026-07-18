"""J2: ensembles v2 -- EBM, Rotation Forest, Cascade Forest, Regularized Greedy Forest.

The EBM must be accurate AND exactly additive (its score = intercept + sum of
per-feature shape functions, the glass-box guarantee). Rotation Forest and Cascade
Forest must classify well; the RGF must fit a regression under its leaf budget.
"""
import numpy as np
import pytest

from sklearn.datasets import make_classification, make_moons, make_friedman1

from nupyml.ensemble import (ExplainableBoostingClassifier, RotationForestClassifier,
                             CascadeForestClassifier, RegularizedGreedyForest)
from nupyml.tree import DecisionTreeClassifier
from nupyml.metrics import accuracy_score, r2_score
from nupyml.model_selection import train_test_split


def test_ebm_is_accurate_and_exactly_additive():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ebm = ExplainableBoostingClassifier(n_rounds=300, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, ebm.predict(Xte)) > 0.75
    # GLASS BOX: the raw score is exactly the intercept plus each feature's shape
    additive = np.full(len(Xte), ebm.intercept_)
    for j in range(X.shape[1]):
        additive += ebm.shape_function(j, Xte[:, j])
    assert np.allclose(additive, ebm._score(Xte), atol=1e-8)


def test_rotation_forest_beats_single_tree():
    # rotation's edge shows on higher-dim structured data, not 2-D toys
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RotationForestClassifier(n_estimators=25, random_state=0).fit(Xtr, ytr)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))
    assert rf_acc > 0.8
    assert rf_acc > accuracy_score(yte, tree.predict(Xte))   # ensemble wins


def test_cascade_forest_classifies_and_grows_layers():
    X, y = make_classification(n_samples=400, n_features=10, n_informative=6,
                               random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    cf = CascadeForestClassifier(n_layers=4, n_forests=2, n_estimators=30,
                                 random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, cf.predict(Xte)) > 0.75
    assert len(cf.layers_) >= 1
    proba = cf.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_regularized_greedy_forest_fits_regression():
    X, y = make_friedman1(n_samples=400, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rgf = RegularizedGreedyForest(max_leaves=80, random_state=0).fit(Xtr, ytr)
    assert r2_score(yte, rgf.predict(Xte)) > 0.6
