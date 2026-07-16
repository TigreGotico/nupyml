import numpy as np
import pytest

import sklearn.metrics as skm

from nupyml import metrics as m
from nupyml.model_selection import (
    GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, ShuffleSplit,
    StratifiedShuffleSplit, RepeatedKFold, LeavePOut, PredefinedSplit,
    RandomizedSearchCV, HalvingGridSearchCV, learning_curve, validation_curve,
    permutation_importance, cross_val_score, get_scorer_names,
)
from nupyml.linear_model import LogisticRegression, Ridge
from nupyml.datasets import make_classification, make_regression, make_blobs

rng = np.random.RandomState(0)
y_true = rng.randint(0, 2, 200)
y_score = np.clip(y_true * 0.6 + rng.uniform(size=200) * 0.7, 0, 1)
y_pred = (y_score > 0.5).astype(int)


@pytest.mark.parametrize("ours,ref,args", [
    (m.balanced_accuracy_score, skm.balanced_accuracy_score, (y_true, y_pred)),
    (m.matthews_corrcoef, skm.matthews_corrcoef, (y_true, y_pred)),
    (m.cohen_kappa_score, skm.cohen_kappa_score, (y_true, y_pred)),
    (m.brier_score_loss, skm.brier_score_loss, (y_true, y_score)),
    (m.hinge_loss, skm.hinge_loss, (y_true, y_score * 2 - 1)),
    (m.average_precision_score, skm.average_precision_score, (y_true, y_score)),
    (m.median_absolute_error, skm.median_absolute_error, (y_score, y_pred)),
    (m.explained_variance_score, skm.explained_variance_score, (y_score, y_score * 0.9)),
    (m.mean_absolute_percentage_error, skm.mean_absolute_percentage_error,
     (y_score + 1, y_score + 1.1)),
    (m.max_error, skm.max_error, (y_score, y_score * 0.5)),
    (m.mean_pinball_loss, skm.mean_pinball_loss, (y_score, y_score * 0.8)),
])
def test_metric_matches_sklearn(ours, ref, args):
    assert ours(*args) == pytest.approx(ref(*args), rel=1e-9)


def test_precision_recall_curve_matches_sklearn():
    p1, r1, t1 = m.precision_recall_curve(y_true, y_score)
    p2, r2, t2 = skm.precision_recall_curve(y_true, y_score)
    assert np.allclose(p1, p2) and np.allclose(r1, r2) and np.allclose(t1, t2)


def test_multiclass_agreement_metrics():
    a = rng.randint(0, 4, 300)
    b = np.where(rng.uniform(size=300) < 0.7, a, rng.randint(0, 4, 300))
    for ours, ref in [
        (m.mutual_info_score, skm.mutual_info_score),
        (m.normalized_mutual_info_score, skm.normalized_mutual_info_score),
        (m.homogeneity_score, skm.homogeneity_score),
        (m.completeness_score, skm.completeness_score),
        (m.v_measure_score, skm.v_measure_score),
        (m.fowlkes_mallows_score, skm.fowlkes_mallows_score),
    ]:
        assert ours(a, b) == pytest.approx(ref(a, b), rel=1e-9)


def test_cluster_quality_scores():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0)
    assert m.calinski_harabasz_score(X, y) == pytest.approx(
        skm.calinski_harabasz_score(X, y), rel=1e-9)
    assert m.davies_bouldin_score(X, y) == pytest.approx(
        skm.davies_bouldin_score(X, y), rel=1e-9)


def test_ndcg_and_topk():
    yt = np.array([[3, 2, 3, 0, 1, 2]])
    ys = np.array([[3.0, 2.5, 1.1, 0.2, 1.0, 2.0]])  # tie-free scores
    assert m.ndcg_score(yt, ys) == pytest.approx(skm.ndcg_score(yt, ys), rel=1e-9)
    y3 = np.array([0, 1, 2])
    s3 = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.5, 0.4]])
    assert m.top_k_accuracy_score(y3, s3, k=2) == pytest.approx(
        skm.top_k_accuracy_score(y3, s3, k=2))


def test_calibration_curve():
    from sklearn.calibration import calibration_curve as sk_calibration_curve
    pt, pp = m.calibration_curve(y_true, y_score, n_bins=5)
    rt, rp = sk_calibration_curve(y_true, y_score, n_bins=5)
    assert np.allclose(pt, rt) and np.allclose(pp, rp)


def test_deviances():
    yt = np.array([1.0, 2, 3, 4])
    yp = np.array([1.2, 1.8, 3.3, 3.9])
    assert m.mean_poisson_deviance(yt, yp) == pytest.approx(
        skm.mean_poisson_deviance(yt, yp), rel=1e-9)
    assert m.mean_gamma_deviance(yt, yp) == pytest.approx(
        skm.mean_gamma_deviance(yt, yp), rel=1e-9)


# ---------------------------------------------------------------------------
# splitters
# ---------------------------------------------------------------------------

def test_group_kfold_no_group_leakage():
    X = np.arange(30)[:, None]
    groups = np.repeat(np.arange(10), 3)
    for tr, te in GroupKFold(n_splits=5).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[te])


def test_stratified_group_kfold():
    X = np.zeros((40, 1))
    y = np.tile([0, 0, 0, 1], 10)
    groups = np.repeat(np.arange(10), 4)
    for tr, te in StratifiedGroupKFold(n_splits=5).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        assert 0 < y[te].mean() < 1  # both classes present


def test_time_series_split_ordering():
    X = np.arange(50)
    for tr, te in TimeSeriesSplit(n_splits=4).split(X):
        assert tr.max() < te.min()


def test_time_series_split_gap():
    X = np.arange(50)
    for tr, te in TimeSeriesSplit(n_splits=3, gap=2).split(X):
        assert te.min() - tr.max() > 2


def test_shuffle_splits():
    X = np.arange(40)
    ss = list(ShuffleSplit(n_splits=3, test_size=0.25, random_state=0).split(X))
    assert len(ss) == 3
    for tr, te in ss:
        assert len(te) == 10 and not set(tr) & set(te)
    y = np.repeat([0, 1], 20)
    for tr, te in StratifiedShuffleSplit(n_splits=3, test_size=0.2,
                                         random_state=0).split(X, y):
        assert abs(y[te].mean() - 0.5) < 0.2


def test_repeated_kfold_and_leave_p_out():
    X = np.arange(10)
    rk = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
    assert len(list(rk.split(X))) == 15
    lp = LeavePOut(p=2)
    assert lp.get_n_splits(X) == 45


def test_predefined_split():
    tf = np.array([0, 1, -1, 0, 1])
    splits = list(PredefinedSplit(tf).split())
    assert len(splits) == 2
    assert 2 in splits[0][0] and 2 in splits[1][0]  # -1 always in train


# ---------------------------------------------------------------------------
# search + curves
# ---------------------------------------------------------------------------

def test_scoring_strings():
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    for name in ["accuracy", "f1", "roc_auc", "neg_log_loss"]:
        scores = cross_val_score(LogisticRegression(), X, y, cv=3, scoring=name)
        assert np.isfinite(scores).all()
    assert "balanced_accuracy" in get_scorer_names()


def test_randomized_search():
    X, y = make_classification(n_samples=150, n_features=6, random_state=1)
    rs = RandomizedSearchCV(LogisticRegression(),
                            {"C": [0.01, 0.1, 1.0, 10.0]},
                            n_iter=3, cv=3, random_state=0).fit(X, y)
    assert rs.best_score_ > 0.7
    assert len(rs.cv_results_["params"]) == 3
    import scipy.stats
    rs2 = RandomizedSearchCV(LogisticRegression(),
                             {"C": scipy.stats.uniform(0.1, 10)},
                             n_iter=4, cv=3, random_state=0).fit(X, y)
    assert rs2.best_params_["C"] > 0


def test_halving_grid_search():
    X, y = make_classification(n_samples=300, n_features=8, random_state=2)
    hs = HalvingGridSearchCV(LogisticRegression(),
                             {"C": [0.01, 0.1, 1.0, 10.0]},
                             factor=2, min_resources=60, cv=3,
                             random_state=0).fit(X, y)
    assert hs.best_score_ > 0.7
    assert hs.predict(X).shape == (300,)


def test_learning_curve_improves():
    X, y = make_classification(n_samples=400, n_features=8, random_state=3)
    sizes, tr, te = learning_curve(LogisticRegression(), X, y,
                                   train_sizes=[0.2, 1.0], cv=3)
    assert te[1].mean() >= te[0].mean() - 0.05


def test_validation_curve():
    X, y = make_regression(n_samples=150, n_features=8, noise=10, random_state=4)
    tr, te = validation_curve(Ridge(), X, y, "alpha", [0.01, 100.0], cv=3)
    assert tr.shape == (2, 3)
    assert tr[0].mean() >= tr[1].mean()  # less regularization fits train better


def test_permutation_importance_finds_informative():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=2,
                               random_state=5)
    clf = LogisticRegression().fit(X, y)
    res = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
    # informative features are the first two by construction
    top2 = set(np.argsort(-res.importances_mean)[:2])
    assert top2 & {0, 1}
