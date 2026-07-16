import numpy as np
import pytest

from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.ensemble import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier,
)
from nupyml.linear_model import LogisticRegression
from nupyml.datasets import make_moons, make_blobs, make_classification, make_regression
from nupyml.model_selection import train_test_split

import sklearn.tree as sktree


def _moons_split(n=400, noise=0.25, seed=0):
    X, y = make_moons(n, noise=noise, random_state=seed)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def test_tree_classifier_overfits_train():
    X, y = make_moons(200, noise=0.2, random_state=0)
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.score(X, y) == 1.0
    assert clf.get_depth() >= 2 and clf.get_n_leaves() >= 4


def test_tree_classifier_matches_sklearn_generalization():
    Xtr, Xte, ytr, yte = _moons_split()
    ours = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr).score(Xte, yte)
    ref = sktree.DecisionTreeClassifier(max_depth=5, random_state=0).fit(
        Xtr, ytr).score(Xte, yte)
    assert ours >= ref - 0.05


def test_tree_entropy_criterion():
    Xtr, Xte, ytr, yte = _moons_split()
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.85


def test_tree_min_samples_leaf():
    X, y = make_moons(200, noise=0.3, random_state=1)
    clf = DecisionTreeClassifier(min_samples_leaf=20).fit(X, y)
    # every leaf must hold >= 20 samples
    def check(node):
        if node.is_leaf:
            assert node.n_samples >= 20
        else:
            check(node.left)
            check(node.right)
    check(clf.tree_)


def test_tree_regressor():
    X, y = make_regression(n_samples=300, n_features=4, n_informative=2,
                           noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ours = DecisionTreeRegressor(max_depth=6).fit(Xtr, ytr).score(Xte, yte)
    ref = sktree.DecisionTreeRegressor(max_depth=6, random_state=0).fit(
        Xtr, ytr).score(Xte, yte)
    assert ours >= ref - 0.1


def test_tree_predict_proba():
    X, y = make_blobs(n_samples=150, centers=3, random_state=0)
    proba = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
    assert proba.shape == (150, 3)
    assert np.allclose(proba.sum(axis=1), 1)


def test_random_forest_beats_single_tree():
    Xtr, Xte, ytr, yte = _moons_split(600, noise=0.35, seed=2)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(
        Xtr, ytr).score(Xte, yte)
    assert forest >= tree


def test_random_forest_regressor():
    X, y = make_regression(n_samples=300, n_features=5, n_informative=3,
                           noise=2.0, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    reg = RandomForestRegressor(n_estimators=30, random_state=0).fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.7


def test_extra_trees():
    Xtr, Xte, ytr, yte = _moons_split()
    assert ExtraTreesClassifier(n_estimators=30, random_state=0).fit(
        Xtr, ytr).score(Xte, yte) > 0.85
    X, y = make_regression(n_samples=200, n_features=4, noise=1.0, random_state=0)
    assert ExtraTreesRegressor(n_estimators=30, random_state=0).fit(
        X, y).score(X, y) > 0.8


def test_bagging():
    Xtr, Xte, ytr, yte = _moons_split()
    assert BaggingClassifier(n_estimators=20, random_state=0).fit(
        Xtr, ytr).score(Xte, yte) > 0.85
    X, y = make_regression(n_samples=200, n_features=4, noise=1.0, random_state=0)
    assert BaggingRegressor(n_estimators=20, random_state=0).fit(
        X, y).score(X, y) > 0.8


def test_adaboost():
    Xtr, Xte, ytr, yte = _moons_split(500, noise=0.2, seed=3)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr).score(Xte, yte)
    assert acc > stump
    assert acc > 0.85


def test_gradient_boosting_classifier_binary():
    Xtr, Xte, ytr, yte = _moons_split(500, noise=0.25, seed=4)
    clf = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.9
    assert np.allclose(clf.predict_proba(Xte).sum(axis=1), 1)


def test_gradient_boosting_classifier_multiclass():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(Xtr, ytr)
    import sklearn.ensemble as ske
    ref = ske.GradientBoostingClassifier(n_estimators=30, random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) >= ref.score(Xte, yte) - 0.05


def test_gradient_boosting_regressor():
    X, y = make_regression(n_samples=400, n_features=5, n_informative=3,
                           noise=1.0, random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2)
    reg = GradientBoostingRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.85


def test_hist_gradient_boosting_regressor():
    X, y = make_regression(n_samples=500, n_features=6, n_informative=4,
                           noise=1.0, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=3)
    reg = HistGradientBoostingRegressor(max_iter=100).fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.85


def test_hist_gradient_boosting_classifier_binary():
    Xtr, Xte, ytr, yte = _moons_split(600, noise=0.25, seed=6)
    clf = HistGradientBoostingClassifier(max_iter=60).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.9


def test_hist_gradient_boosting_classifier_multiclass():
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=2.0, random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7)
    clf = HistGradientBoostingClassifier(max_iter=40).fit(Xtr, ytr)
    import sklearn.ensemble as ske
    ref = ske.HistGradientBoostingClassifier(max_iter=40).fit(Xtr, ytr)
    assert clf.score(Xte, yte) >= ref.score(Xte, yte) - 0.05


def test_voting_classifier():
    Xtr, Xte, ytr, yte = _moons_split()
    clf = VotingClassifier([
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier(max_depth=5)),
        ("rf", RandomForestClassifier(n_estimators=20, random_state=0)),
    ], voting="soft").fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.85
    hard = VotingClassifier([
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier(max_depth=5)),
    ], voting="hard").fit(Xtr, ytr)
    assert hard.score(Xte, yte) > 0.8


def test_string_labels_through_ensembles():
    X, y = make_blobs(n_samples=150, centers=2, random_state=0)
    labels = np.array(["neg", "pos"])[y]
    for est in [DecisionTreeClassifier(max_depth=3),
                RandomForestClassifier(n_estimators=10, random_state=0),
                GradientBoostingClassifier(n_estimators=10, random_state=0),
                HistGradientBoostingClassifier(max_iter=10)]:
        est.fit(X, labels)
        assert set(est.predict(X)) <= {"neg", "pos"}
