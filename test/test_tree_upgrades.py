import numpy as np
import pytest

import sklearn.tree as skt
import sklearn.ensemble as ske

from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             HistGradientBoostingClassifier,
                             HistGradientBoostingRegressor)
from nupyml.datasets import make_moons, make_regression, make_classification

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# cost-complexity pruning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_ccp_alpha_matches_sklearn(alpha):
    X, y = make_moons(300, noise=0.3, random_state=0)
    ours = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0).fit(X, y)
    ref = skt.DecisionTreeClassifier(ccp_alpha=alpha, random_state=0).fit(X, y)
    assert ours.get_n_leaves() == ref.get_n_leaves()
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=0.02)


def test_ccp_alpha_shrinks_the_tree():
    X, y = make_moons(300, noise=0.3, random_state=1)
    full = DecisionTreeClassifier(random_state=0).fit(X, y)
    pruned = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0).fit(X, y)
    assert pruned.get_n_leaves() < full.get_n_leaves()
    assert pruned.get_depth() <= full.get_depth()


def test_ccp_alpha_improves_generalization():
    from nupyml.model_selection import cross_val_score
    X, y = make_moons(400, noise=0.4, random_state=2)
    # a single split is too noisy to show this; average over folds
    full = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5)
    pruned = cross_val_score(DecisionTreeClassifier(ccp_alpha=0.02,
                                                    random_state=0), X, y, cv=5)
    assert pruned.mean() > full.mean()
    # and the pruned tree fits the training data less well: that is the trade
    full_fit = DecisionTreeClassifier(random_state=0).fit(X, y)
    pruned_fit = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0).fit(X, y)
    assert full_fit.score(X, y) > pruned_fit.score(X, y)


def test_cost_complexity_pruning_path():
    X, y = make_moons(200, noise=0.3, random_state=3)
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
    assert len(path.ccp_alphas) == len(path.impurities)
    assert np.all(np.diff(path.ccp_alphas) >= -1e-12)   # non-decreasing
    assert np.all(np.diff(path.impurities) >= -1e-12)   # pruning adds impurity
    assert path.ccp_alphas[0] == 0.0
    # the largest alpha in the path collapses the tree to a single leaf
    biggest = DecisionTreeClassifier(ccp_alpha=path.ccp_alphas[-1] + 1e-9,
                                     random_state=0).fit(X, y)
    assert biggest.get_n_leaves() == 1


def test_ccp_alpha_in_random_forest():
    X, y = make_moons(300, noise=0.35, random_state=4)
    plain = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    pruned = RandomForestClassifier(n_estimators=20, ccp_alpha=0.02,
                                    random_state=0).fit(X, y)
    assert (np.mean([t.get_n_leaves() for t in pruned.estimators_])
            < np.mean([t.get_n_leaves() for t in plain.estimators_]))


def test_regressor_ccp_alpha():
    X, y = make_regression(n_samples=200, n_features=4, noise=10.0,
                           random_state=5)
    full = DecisionTreeRegressor(random_state=0).fit(X, y)
    pruned = DecisionTreeRegressor(ccp_alpha=100.0, random_state=0).fit(X, y)
    assert pruned.get_n_leaves() < full.get_n_leaves()


# ---------------------------------------------------------------------------
# missing values
# ---------------------------------------------------------------------------

def _with_nans(X, frac=0.2, col=0, seed=0):
    r = np.random.RandomState(seed)
    Xn = X.copy()
    Xn[r.uniform(size=len(X)) < frac, col] = np.nan
    return Xn


def test_tree_fits_with_nan_without_imputation():
    X, y = make_moons(300, noise=0.2, random_state=6)
    Xn = _with_nans(X)
    clf = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xn, y)
    assert clf.score(Xn, y) > 0.85
    assert np.isfinite(clf.predict(Xn)).all()


def test_tree_learns_missingness_direction():
    # rows with a missing feature belong to class 1: the tree should route
    # NaN to whichever child is pure, without any imputation
    r = np.random.RandomState(7)
    X = r.normal(size=(300, 2))
    y = (r.uniform(size=300) < 0.5).astype(int)
    X[y == 1, 0] = np.nan
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_tree_nan_at_predict_time_only():
    X, y = make_moons(200, noise=0.2, random_state=8)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    Xq = _with_nans(X[:20], frac=1.0)
    pred = clf.predict(Xq)
    assert len(pred) == 20 and set(np.unique(pred)) <= {0, 1}


def test_tree_still_rejects_infinity():
    X, y = make_moons(50, random_state=9)
    Xbad = X.copy()
    Xbad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        DecisionTreeClassifier().fit(Xbad, y)


def test_histgb_handles_nan():
    X, y = make_moons(400, noise=0.25, random_state=10)
    Xn = _with_nans(X)
    clf = HistGradientBoostingClassifier(max_iter=40).fit(Xn, y)
    assert clf.score(Xn, y) > 0.85
    ref = ske.HistGradientBoostingClassifier(max_iter=40).fit(Xn, y)
    assert clf.score(Xn, y) >= ref.score(Xn, y) - 0.1


def test_histgb_regressor_handles_nan():
    X, y = make_regression(n_samples=300, n_features=5, noise=5.0,
                           random_state=11)
    Xn = _with_nans(X, frac=0.15)
    reg = HistGradientBoostingRegressor(max_iter=40).fit(Xn, y)
    assert reg.score(Xn, y) > 0.7


def test_forest_handles_nan():
    X, y = make_moons(300, noise=0.2, random_state=12)
    Xn = _with_nans(X)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(Xn, y)
    assert rf.score(Xn, y) > 0.9


# ---------------------------------------------------------------------------
# monotonic constraints
# ---------------------------------------------------------------------------

def test_monotonic_increasing_regressor():
    r = np.random.RandomState(13)
    X = r.uniform(0, 10, size=(400, 2))
    y = 2 * X[:, 0] + r.normal(0, 3, size=400)     # noisy but increasing in x0
    m = DecisionTreeRegressor(monotonic_cst=[1, 0], max_depth=5,
                              random_state=0).fit(X, y)
    grid = np.column_stack([np.linspace(0, 10, 100), np.full(100, 5.0)])
    pred = m.predict(grid)
    assert np.all(np.diff(pred) >= -1e-9)
    # the unconstrained tree is free to wiggle against the trend
    plain = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    assert not np.all(np.diff(plain.predict(grid)) >= -1e-9)


def test_monotonic_decreasing_regressor():
    r = np.random.RandomState(14)
    X = r.uniform(0, 10, size=(400, 2))
    y = -1.5 * X[:, 0] + r.normal(0, 3, size=400)
    m = DecisionTreeRegressor(monotonic_cst=[-1, 0], max_depth=5,
                              random_state=0).fit(X, y)
    grid = np.column_stack([np.linspace(0, 10, 100), np.full(100, 5.0)])
    assert np.all(np.diff(m.predict(grid)) <= 1e-9)


def test_monotonic_classifier_probability():
    r = np.random.RandomState(15)
    X = r.uniform(0, 10, size=(400, 2))
    y = (r.uniform(size=400) < X[:, 0] / 10).astype(int)
    m = DecisionTreeClassifier(monotonic_cst=[1, 0], max_depth=4,
                               random_state=0).fit(X, y)
    grid = np.column_stack([np.linspace(0, 10, 100), np.full(100, 5.0)])
    proba = m.predict_proba(grid)[:, 1]
    assert np.all(np.diff(proba) >= -1e-9)


def test_monotonic_constraint_costs_some_fit():
    r = np.random.RandomState(16)
    X = r.uniform(0, 10, size=(300, 1))
    y = np.sin(X.ravel())            # not monotone at all
    free = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    forced = DecisionTreeRegressor(monotonic_cst=[1], max_depth=5,
                                   random_state=0).fit(X, y)
    # forcing monotonicity on non-monotone data must hurt the fit
    assert forced.score(X, y) < free.score(X, y)


def test_monotonic_rejects_multiclass():
    X, y = make_classification(n_samples=100, n_features=3, n_classes=3,
                               n_informative=3, random_state=17)
    with pytest.raises(ValueError, match="binary"):
        DecisionTreeClassifier(monotonic_cst=[1, 0, 0]).fit(X, y)


def test_monotonic_in_random_forest():
    r = np.random.RandomState(18)
    X = r.uniform(0, 10, size=(300, 2))
    y = 2 * X[:, 0] + r.normal(0, 2, size=300)
    rf = RandomForestRegressor(n_estimators=15, monotonic_cst=[1, 0],
                               max_depth=4, random_state=0).fit(X, y)
    grid = np.column_stack([np.linspace(0, 10, 60), np.full(60, 5.0)])
    # every member is monotone, so the average is too
    assert np.all(np.diff(rf.predict(grid)) >= -1e-9)


# ---------------------------------------------------------------------------
# categorical features in HistGB
# ---------------------------------------------------------------------------

def test_histgb_categorical_features():
    r = np.random.RandomState(19)
    n = 400
    cat = r.randint(0, 4, size=n).astype(float)
    num = r.normal(size=n)
    # each category carries its own offset: no ordinal relationship
    effect = np.array([5.0, -5.0, 2.0, -2.0])[cat.astype(int)]
    y = effect + num + r.normal(0, 0.3, size=n)
    X = np.column_stack([cat, num])
    reg = HistGradientBoostingRegressor(max_iter=60,
                                        categorical_features=[0]).fit(X, y)
    assert reg.score(X, y) > 0.9
    assert 0 in reg._mapper.categorical_


def test_histgb_categorical_rejects_too_many_levels():
    r = np.random.RandomState(20)
    X = np.column_stack([np.arange(50.0), r.normal(size=50)])
    y = r.normal(size=50)
    with pytest.raises(ValueError, match="more than max_bins"):
        HistGradientBoostingRegressor(max_bins=10,
                                      categorical_features=[0]).fit(X, y)


def test_histgb_binmapper_reserves_a_missing_bin():
    from nupyml.ensemble._hist_gb import _BinMapper
    X = np.array([[1.0], [2.0], [np.nan], [3.0]])
    mapper = _BinMapper(max_bins=8).fit(X)
    bins = mapper.transform(X)
    # NaN gets its own bin, distinct from every observed value's bin
    assert bins[2, 0] == mapper.missing_bin_
    assert bins[2, 0] not in bins[[0, 1, 3], 0]
