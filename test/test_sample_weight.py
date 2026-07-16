import numpy as np
import pytest

import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb
import sklearn.tree as sktree

from nupyml.linear_model import (LinearRegression, Ridge, LogisticRegression,
                                 SGDClassifier, SGDRegressor)
from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             GradientBoostingRegressor, AdaBoostClassifier,
                             HistGradientBoostingClassifier,
                             HistGradientBoostingRegressor)
from nupyml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from nupyml.svm import SVC
from nupyml.cluster import KMeans, MiniBatchKMeans
from nupyml.preprocessing import StandardScaler, MinMaxScaler
from nupyml.nn import MLPClassifier, MLPRegressor
from nupyml import metrics as m
from nupyml.datasets import make_classification, make_regression, make_blobs

rng = np.random.RandomState(0)


def _dup_equivalence(fit_weighted, fit_duplicated):
    """Integer weights must equal duplicating the corresponding rows."""
    assert np.allclose(fit_weighted, fit_duplicated, atol=1e-6)


# ---------------------------------------------------------------------------
# linear models
# ---------------------------------------------------------------------------

def test_linear_regression_sample_weight_matches_sklearn():
    X, y = make_regression(n_samples=80, n_features=5, noise=1.0, random_state=0)
    w = rng.uniform(0.1, 3.0, size=80)
    ours = LinearRegression().fit(X, y, sample_weight=w)
    ref = sklm.LinearRegression().fit(X, y, sample_weight=w)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-8)
    assert abs(ours.intercept_ - ref.intercept_) < 1e-8


def test_linear_regression_integer_weights_equal_duplication():
    X, y = make_regression(n_samples=40, n_features=3, noise=0.5, random_state=1)
    w = rng.randint(1, 4, size=40).astype(float)
    weighted = LinearRegression().fit(X, y, sample_weight=w).coef_
    idx = np.repeat(np.arange(40), w.astype(int))
    dup = LinearRegression().fit(X[idx], y[idx]).coef_
    _dup_equivalence(weighted, dup)


def test_ridge_sample_weight_matches_sklearn():
    X, y = make_regression(n_samples=60, n_features=4, noise=1.0, random_state=2)
    w = rng.uniform(0.5, 2.0, size=60)
    ours = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    ref = sklm.Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-6)


def test_logistic_sample_weight_shifts_boundary():
    X, y = make_classification(n_samples=200, n_features=4, random_state=3)
    w = np.where(y == 1, 10.0, 1.0)
    plain = LogisticRegression().fit(X, y)
    weighted = LogisticRegression().fit(X, y, sample_weight=w)
    # upweighting the positive class must raise its recall
    assert (m.recall_score(y, weighted.predict(X))
            >= m.recall_score(y, plain.predict(X)))
    ref = sklm.LogisticRegression().fit(X, y, sample_weight=w)
    assert np.allclose(weighted.coef_.ravel(), ref.coef_.ravel(), atol=0.1)


def test_logistic_zero_weight_ignores_samples():
    X, y = make_classification(n_samples=100, n_features=4, random_state=4)
    w = np.ones(100)
    w[:20] = 0.0
    weighted = LogisticRegression().fit(X, y, sample_weight=w)
    subset = LogisticRegression().fit(X[20:], y[20:])
    assert np.allclose(weighted.coef_, subset.coef_, atol=1e-3)


# ---------------------------------------------------------------------------
# trees and ensembles
# ---------------------------------------------------------------------------

def test_tree_classifier_sample_weight_equals_duplication():
    X, y = make_classification(n_samples=60, n_features=4, random_state=5)
    w = rng.randint(1, 4, size=60).astype(float)
    weighted = DecisionTreeClassifier(max_depth=3).fit(X, y, sample_weight=w)
    idx = np.repeat(np.arange(60), w.astype(int))
    dup = DecisionTreeClassifier(max_depth=3).fit(X[idx], y[idx])
    assert np.array_equal(weighted.predict(X), dup.predict(X))


def test_tree_regressor_sample_weight_matches_sklearn():
    X, y = make_regression(n_samples=80, n_features=4, noise=1.0, random_state=6)
    w = rng.uniform(0.5, 2.0, size=80)
    ours = DecisionTreeRegressor(max_depth=3).fit(X, y, sample_weight=w)
    ref = sktree.DecisionTreeRegressor(max_depth=3, random_state=0).fit(
        X, y, sample_weight=w)
    assert np.corrcoef(ours.predict(X), ref.predict(X))[0, 1] > 0.98


def test_tree_zero_weight_ignores_samples():
    X, y = make_classification(n_samples=80, n_features=3, random_state=7)
    w = np.ones(80)
    w[:30] = 0.0
    weighted = DecisionTreeClassifier(max_depth=3).fit(X, y, sample_weight=w)
    subset = DecisionTreeClassifier(max_depth=3).fit(X[30:], y[30:])
    assert np.array_equal(weighted.predict(X), subset.predict(X))


def test_forest_and_boosting_accept_sample_weight():
    X, y = make_classification(n_samples=150, n_features=5, random_state=8)
    w = rng.uniform(0.5, 2.0, size=150)
    assert RandomForestClassifier(n_estimators=10, random_state=0).fit(
        X, y, sample_weight=w).score(X, y) > 0.8
    Xr, yr = make_regression(n_samples=150, n_features=5, noise=1.0,
                             random_state=9)
    assert RandomForestRegressor(n_estimators=10, random_state=0).fit(
        Xr, yr, sample_weight=w).score(Xr, yr) > 0.8
    assert GradientBoostingRegressor(n_estimators=20, random_state=0).fit(
        Xr, yr, sample_weight=w).score(Xr, yr) > 0.8
    assert HistGradientBoostingClassifier(max_iter=20).fit(
        X, y, sample_weight=w).score(X, y) > 0.8
    assert HistGradientBoostingRegressor(max_iter=20).fit(
        Xr, yr, sample_weight=w).score(Xr, yr) > 0.8


def test_adaboost_uses_direct_weighting():
    X, y = make_classification(n_samples=200, n_features=4, random_state=10)
    clf = AdaBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.85
    # base estimator must receive sample_weight rather than resampled data
    assert "sample_weight" in DecisionTreeClassifier.fit.__code__.co_varnames


def test_forest_oob_score():
    X, y = make_classification(n_samples=300, n_features=6, random_state=11)
    rf = RandomForestClassifier(n_estimators=30, oob_score=True,
                                random_state=0).fit(X, y)
    assert 0.5 < rf.oob_score_ <= 1.0
    assert rf.oob_score_ < rf.score(X, y) + 1e-9  # OOB never beats in-sample
    Xr, yr = make_regression(n_samples=300, n_features=5, noise=5.0,
                             random_state=12)
    rfr = RandomForestRegressor(n_estimators=30, oob_score=True,
                                random_state=0).fit(Xr, yr)
    assert rfr.oob_score_ > 0.5


def test_forest_warm_start_adds_trees():
    X, y = make_classification(n_samples=150, n_features=4, random_state=13)
    rf = RandomForestClassifier(n_estimators=5, warm_start=True, random_state=0)
    rf.fit(X, y)
    assert len(rf.estimators_) == 5
    rf.set_params(n_estimators=12).fit(X, y)
    assert len(rf.estimators_) == 12


def test_gradient_boosting_warm_start():
    X, y = make_regression(n_samples=150, n_features=4, noise=1.0,
                           random_state=14)
    gb = GradientBoostingRegressor(n_estimators=10, warm_start=True,
                                   random_state=0).fit(X, y)
    first = gb.score(X, y)
    gb.set_params(n_estimators=40).fit(X, y)
    assert len(gb.estimators_) == 40
    assert gb.score(X, y) > first


def test_histgb_early_stopping_stops_early():
    X, y = make_regression(n_samples=400, n_features=5, noise=20.0,
                           random_state=15)
    est = HistGradientBoostingRegressor(max_iter=200, early_stopping=True,
                                        n_iter_no_change=5, random_state=0)
    est.fit(X, y)
    assert est.n_iter_ < 200
    assert len(est.validation_score_) == est.n_iter_


# ---------------------------------------------------------------------------
# naive bayes / svm / clustering
# ---------------------------------------------------------------------------

def test_gaussian_nb_sample_weight_matches_sklearn():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=2.0, random_state=16)
    w = rng.uniform(0.5, 2.0, size=150)
    ours = GaussianNB().fit(X, y, sample_weight=w)
    ref = sknb.GaussianNB().fit(X, y, sample_weight=w)
    assert np.allclose(ours.theta_, ref.theta_, atol=1e-8)
    assert np.allclose(ours.class_prior_, ref.class_prior_, atol=1e-8)


def test_multinomial_nb_sample_weight_matches_sklearn():
    X = rng.poisson(2, size=(120, 10)).astype(float)
    y = rng.randint(0, 2, 120)
    w = rng.uniform(0.5, 2.0, size=120)
    ours = MultinomialNB().fit(X, y, sample_weight=w)
    ref = sknb.MultinomialNB().fit(X, y, sample_weight=w)
    assert np.allclose(ours.feature_log_prob_, ref.feature_log_prob_, atol=1e-8)


def test_svc_sample_weight_changes_boundary():
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.5, random_state=17)
    w = np.where(y == 1, 20.0, 1.0)
    plain = SVC(kernel="linear", random_state=0).fit(X, y)
    weighted = SVC(kernel="linear", random_state=0).fit(X, y, sample_weight=w)
    assert (m.recall_score(y, weighted.predict(X))
            >= m.recall_score(y, plain.predict(X)))


def test_kmeans_sample_weight_pulls_centroid():
    X = np.array([[0.0, 0], [0, 1], [10, 0], [10, 1]])
    w = np.array([1.0, 1, 100, 1])
    km = KMeans(n_clusters=1, n_init=1, random_state=0).fit(X, sample_weight=w)
    # heavy point dominates the single centroid
    assert km.cluster_centers_[0, 0] > 9.0


# ---------------------------------------------------------------------------
# partial_fit
# ---------------------------------------------------------------------------

def test_scaler_partial_fit_matches_full_fit():
    X = rng.normal(5, 3, size=(200, 4))
    ss = StandardScaler()
    for chunk in np.array_split(X, 5):
        ss.partial_fit(chunk)
    full = StandardScaler().fit(X)
    assert np.allclose(ss.mean_, full.mean_)
    assert np.allclose(ss.scale_, full.scale_, atol=1e-8)
    mm = MinMaxScaler()
    for chunk in np.array_split(X, 5):
        mm.partial_fit(chunk)
    assert np.allclose(mm.transform(X), MinMaxScaler().fit(X).transform(X))


def test_gaussian_nb_partial_fit_matches_full_fit():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=18)
    nb = GaussianNB()
    classes = np.unique(y)
    for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
        nb.partial_fit(Xc, yc, classes=classes)
    full = GaussianNB().fit(X, y)
    assert np.allclose(nb.theta_, full.theta_, atol=1e-8)
    assert np.allclose(nb.class_prior_, full.class_prior_, atol=1e-8)
    assert np.array_equal(nb.predict(X), full.predict(X))


def test_multinomial_nb_partial_fit_matches_full_fit():
    X = rng.poisson(2, size=(200, 8)).astype(float)
    y = rng.randint(0, 3, 200)
    nb = MultinomialNB()
    for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
        nb.partial_fit(Xc, yc, classes=np.unique(y))
    full = MultinomialNB().fit(X, y)
    assert np.allclose(nb.feature_log_prob_, full.feature_log_prob_)
    assert np.allclose(nb.class_log_prior_, full.class_log_prior_)


def test_bernoulli_nb_partial_fit():
    X = (rng.uniform(size=(200, 6)) > 0.5).astype(float)
    y = rng.randint(0, 2, 200)
    nb = BernoulliNB()
    for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
        nb.partial_fit(Xc, yc, classes=np.unique(y))
    full = BernoulliNB().fit(X, y)
    assert np.allclose(nb.feature_log_prob_, full.feature_log_prob_)


def test_sgd_partial_fit_learns_incrementally():
    X, y = make_classification(n_samples=400, n_features=5, random_state=19)
    clf = SGDClassifier(loss="log", learning_rate=0.1, random_state=0)
    classes = np.unique(y)
    for _ in range(50):
        for Xc, yc in zip(np.array_split(X, 8), np.array_split(y, 8)):
            clf.partial_fit(Xc, yc, classes=classes)
    assert clf.score(X, y) > 0.8


def test_sgd_regressor_partial_fit():
    X, y = make_regression(n_samples=300, n_features=4, noise=0.5,
                           random_state=20)
    y = (y - y.mean()) / y.std()
    reg = SGDRegressor(learning_rate=0.05, random_state=0)
    for _ in range(200):
        for Xc, yc in zip(np.array_split(X, 6), np.array_split(y, 6)):
            reg.partial_fit(Xc, yc)
    assert reg.score(X, y) > 0.8


def test_minibatch_kmeans_partial_fit():
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.6, random_state=21)
    km = MiniBatchKMeans(n_clusters=3, random_state=0)
    for _ in range(20):
        for chunk in np.array_split(X, 8):
            km.partial_fit(chunk)
    from nupyml.metrics import adjusted_rand_score
    assert adjusted_rand_score(y, km.predict(X)) > 0.9


# ---------------------------------------------------------------------------
# MLP warm_start / early_stopping / lbfgs
# ---------------------------------------------------------------------------

def test_mlp_warm_start_continues_training():
    X, y = make_classification(n_samples=200, n_features=4, random_state=22)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=5, warm_start=True,
                        n_iter_no_change=100, random_state=0).fit(X, y)
    n_first = len(mlp.loss_curve_)
    mlp.fit(X, y)
    assert len(mlp.loss_curve_) > n_first


def test_mlp_early_stopping():
    X, y = make_classification(n_samples=300, n_features=5, random_state=23)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                        early_stopping=True, n_iter_no_change=5,
                        random_state=0).fit(X, y)
    assert mlp.n_iter_ < 500
    assert len(mlp.validation_scores_) == mlp.n_iter_


def test_mlp_lbfgs_solver():
    X, y = make_classification(n_samples=200, n_features=4, random_state=24)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), solver="lbfgs", max_iter=200,
                        random_state=0).fit(X, y)
    assert mlp.score(X, y) > 0.85
    Xr, yr = make_regression(n_samples=150, n_features=4, noise=0.5,
                             random_state=25)
    reg = MLPRegressor(hidden_layer_sizes=(32,), solver="lbfgs", max_iter=300,
                       random_state=0).fit(Xr, yr)
    assert reg.score(Xr, yr) > 0.9


# ---------------------------------------------------------------------------
# weighted metrics
# ---------------------------------------------------------------------------

def test_weighted_metrics_match_sklearn():
    import sklearn.metrics as skm
    y = rng.randint(0, 2, 100)
    p = rng.randint(0, 2, 100)
    w = rng.uniform(0.5, 3.0, size=100)
    assert m.accuracy_score(y, p, sample_weight=w) == pytest.approx(
        skm.accuracy_score(y, p, sample_weight=w))
    assert m.f1_score(y, p, sample_weight=w) == pytest.approx(
        skm.f1_score(y, p, sample_weight=w))
    assert m.precision_score(y, p, sample_weight=w) == pytest.approx(
        skm.precision_score(y, p, sample_weight=w))
    yt = rng.normal(size=50)
    yp = yt + rng.normal(scale=0.3, size=50)
    ws = rng.uniform(0.5, 2.0, size=50)
    assert m.mean_squared_error(yt, yp, sample_weight=ws) == pytest.approx(
        skm.mean_squared_error(yt, yp, sample_weight=ws))
    assert m.r2_score(yt, yp, sample_weight=ws) == pytest.approx(
        skm.r2_score(yt, yp, sample_weight=ws))
