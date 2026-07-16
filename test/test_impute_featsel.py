import numpy as np
import pytest

import sklearn.feature_selection as skfs
import sklearn.impute as skimp

from nupyml.impute import (SimpleImputer, MissingIndicator, KNNImputer,
                           IterativeImputer)
from nupyml.feature_selection import (
    f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr,
    SelectFromModel, RFE, RFECV, SequentialFeatureSelector,
)
from nupyml.linear_model import LogisticRegression, Ridge
from nupyml.ensemble import RandomForestClassifier
from nupyml.datasets import make_classification, make_regression

rng = np.random.RandomState(0)


def _X_missing():
    X = rng.normal(size=(50, 4)).copy()
    X[::7, 0] = np.nan
    X[::5, 2] = np.nan
    return X


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_imputer_matches_sklearn(strategy):
    X = _X_missing()
    ours = SimpleImputer(strategy=strategy).fit_transform(X)
    ref = skimp.SimpleImputer(strategy=strategy).fit_transform(X)
    assert np.allclose(ours, ref)
    assert not np.isnan(ours).any()


def test_simple_imputer_constant_and_sentinel():
    X = np.array([[1.0, -1.0], [2.0, 3.0]])
    out = SimpleImputer(missing_values=-1.0, strategy="constant",
                        fill_value=9.0).fit_transform(X)
    assert out[0, 1] == 9.0


def test_missing_indicator():
    X = _X_missing()
    ours = MissingIndicator().fit_transform(X)
    ref = skimp.MissingIndicator().fit_transform(X)
    assert np.array_equal(ours, ref)


def test_knn_imputer_matches_sklearn():
    X = _X_missing()
    ours = KNNImputer(n_neighbors=3).fit_transform(X)
    ref = skimp.KNNImputer(n_neighbors=3).fit_transform(X)
    assert np.allclose(ours, ref, atol=1e-8)


def test_iterative_imputer_beats_mean():
    # correlated columns: iterative should reconstruct better than mean
    n = 200
    a = rng.normal(size=n)
    X_full = np.column_stack([a, 2 * a + rng.normal(scale=0.1, size=n),
                              rng.normal(size=n)])
    X = X_full.copy()
    miss = rng.uniform(size=n) < 0.2
    X[miss, 1] = np.nan
    it = IterativeImputer(random_state=0).fit_transform(X)
    mean = SimpleImputer().fit_transform(X)
    err_it = np.abs(it[miss, 1] - X_full[miss, 1]).mean()
    err_mean = np.abs(mean[miss, 1] - X_full[miss, 1]).mean()
    assert err_it < err_mean / 2


def test_iterative_imputer_transform_new_data():
    X = _X_missing()
    imp = IterativeImputer(random_state=0).fit(X)
    X2 = _X_missing()
    out = imp.transform(X2)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# scoring functions
# ---------------------------------------------------------------------------

def test_f_classif_matches_sklearn():
    X, y = make_classification(n_samples=120, n_features=6, random_state=1)
    F1, p1 = f_classif(X, y)
    F2, p2 = skfs.f_classif(X, y)
    assert np.allclose(F1, F2) and np.allclose(p1, p2)


def test_f_regression_matches_sklearn():
    X, y = make_regression(n_samples=120, n_features=6, noise=1.0, random_state=2)
    F1, p1 = f_regression(X, y)
    F2, p2 = skfs.f_regression(X, y)
    assert np.allclose(F1, F2) and np.allclose(p1, p2)


def test_chi2_matches_sklearn():
    X = rng.poisson(3, size=(100, 8)).astype(float)
    y = rng.randint(0, 3, 100)
    s1, p1 = chi2(X, y)
    s2, p2 = skfs.chi2(X, y)
    assert np.allclose(s1, s2) and np.allclose(p1, p2)


def test_mutual_info_ranks_informative_features():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=2,
                               random_state=3)
    mi = mutual_info_classif(X, y)
    assert set(np.argsort(-mi)[:2]) & {0, 1}
    Xr, yr = make_regression(n_samples=300, n_features=5, n_informative=1,
                             random_state=4)
    mir = mutual_info_regression(Xr, yr)
    # the single informative feature has the largest MI
    informative = np.argmax(np.abs(np.corrcoef(Xr.T, yr)[-1, :-1]))
    assert np.argmax(mir) == informative


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

def test_variance_threshold():
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    vt = VarianceThreshold().fit(X)
    assert list(vt.get_support()) == [False, True]
    assert vt.transform(X).shape == (30, 1)


def test_select_kbest_and_percentile():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=3,
                               random_state=5)
    kb = SelectKBest(f_classif, k=3).fit(X, y)
    assert kb.transform(X).shape == (200, 3)
    ref = skfs.SelectKBest(skfs.f_classif, k=3).fit(X, y)
    assert np.array_equal(kb.get_support(), ref.get_support())
    sp = SelectPercentile(f_classif, percentile=30).fit(X, y)
    assert sp.transform(X).shape[1] == 3


def test_select_fpr():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=2,
                               random_state=6)
    fpr = SelectFpr(f_classif, alpha=0.01).fit(X, y)
    assert 1 <= fpr.get_support().sum() <= 8


def test_select_from_model():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=3,
                               random_state=7)
    sfm = SelectFromModel(RandomForestClassifier(n_estimators=20,
                                                 random_state=0)).fit(X, y)
    assert 1 <= sfm.transform(X).shape[1] < 10
    sfm2 = SelectFromModel(LogisticRegression(), max_features=2).fit(X, y)
    assert sfm2.transform(X).shape[1] <= 2


def test_rfe():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=2,
                               random_state=8)
    rfe = RFE(LogisticRegression(), n_features_to_select=3).fit(X, y)
    assert rfe.get_support().sum() == 3
    assert rfe.score(X, y) > 0.7
    assert rfe.ranking_.max() > 1


def test_rfecv_picks_reasonable_size():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=2,
                               random_state=9)
    rfecv = RFECV(LogisticRegression(), cv=3).fit(X, y)
    assert 1 <= rfecv.n_features_ <= 10
    assert rfecv.estimator_.score(rfecv.transform(X), y) > 0.7


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_sequential_feature_selector(direction):
    X, y = make_regression(n_samples=150, n_features=6, n_informative=2,
                           noise=0.5, random_state=10)
    sfs = SequentialFeatureSelector(Ridge(), n_features_to_select=2,
                                    direction=direction, cv=3).fit(X, y)
    assert sfs.get_support().sum() == 2
    # informative features should be found
    w = np.abs(np.corrcoef(X.T, y)[-1, :-1])
    top2 = set(np.argsort(-w)[:2])
    assert set(sfs.get_support(indices=True)) == top2


def test_inverse_transform():
    X, y = make_classification(n_samples=100, n_features=6, random_state=11)
    kb = SelectKBest(f_classif, k=2).fit(X, y)
    Xt = kb.transform(X)
    Xi = kb.inverse_transform(Xt)
    assert Xi.shape == X.shape
    assert np.allclose(Xi[:, kb.get_support()], Xt)
