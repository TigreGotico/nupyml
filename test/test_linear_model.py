import numpy as np
import pytest

from nupyml.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    Perceptron, SGDClassifier, SGDRegressor,
)
from nupyml.datasets import make_regression, make_classification, make_blobs
from nupyml.model_selection import train_test_split

import sklearn.linear_model as sklm  # test-only cross-validation


def test_ols_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=8, noise=0.5, random_state=0)
    ours = LinearRegression().fit(X, y)
    ref = sklm.LinearRegression().fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-8)
    assert abs(ours.intercept_ - ref.intercept_) < 1e-8


def test_ridge_matches_sklearn():
    X, y = make_regression(n_samples=80, n_features=10, noise=1.0, random_state=1)
    ours = Ridge(alpha=2.5).fit(X, y)
    ref = sklm.Ridge(alpha=2.5).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-6)


def test_lasso_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=15, n_informative=4,
                           noise=0.5, random_state=2)
    ours = Lasso(alpha=1.0, max_iter=5000, tol=1e-8).fit(X, y)
    ref = sklm.Lasso(alpha=1.0, max_iter=5000, tol=1e-8).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-3)
    # sparsity induced
    assert np.sum(np.abs(ours.coef_) < 1e-8) > 5


def test_elasticnet_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=12, n_informative=5,
                           noise=0.5, random_state=3)
    ours = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000, tol=1e-8).fit(X, y)
    ref = sklm.ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000, tol=1e-8).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-3)


def test_logistic_binary_close_to_sklearn():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                               random_state=0)
    ours = LogisticRegression(C=1.0).fit(X, y)
    ref = sklm.LogisticRegression(C=1.0).fit(X, y)
    assert np.allclose(ours.coef_.ravel(), ref.coef_.ravel(), atol=0.05)
    assert ours.score(X, y) >= ref.score(X, y) - 0.02
    proba = ours.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)


def test_logistic_multiclass():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=0)
    clf = LogisticRegression().fit(X, y)
    ref = sklm.LogisticRegression().fit(X, y)
    assert clf.score(X, y) >= ref.score(X, y) - 0.01
    assert clf.predict_proba(X).shape == (300, 4)


def test_logistic_string_labels():
    X, y = make_blobs(n_samples=100, centers=2, random_state=0)
    labels = np.array(["cat", "dog"])[y]
    clf = LogisticRegression().fit(X, labels)
    assert set(clf.predict(X)) <= {"cat", "dog"}
    assert clf.score(X, labels) > 0.9


def test_perceptron_separable():
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    clf = Perceptron(random_state=0).fit(X, y)
    assert clf.score(X, y) == 1.0


@pytest.mark.parametrize("loss", ["hinge", "log"])
def test_sgd_classifier(loss):
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = SGDClassifier(loss=loss, max_iter=200, random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.8


def test_sgd_classifier_multiclass():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=1)
    clf = SGDClassifier(max_iter=200, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.9


def test_sgd_regressor():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.5, random_state=0)
    # scale y for SGD stability
    ys = (y - y.mean()) / y.std()
    reg = SGDRegressor(max_iter=500, random_state=0).fit(X, ys)
    assert reg.score(X, ys) > 0.95
