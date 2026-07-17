"""F4: ordinal regression, multi-task elastic net, and quantile forests.

Ordinal models are held to the property that separates them from a plain
classifier -- respecting the ORDER, so near-misses cost less (lower MAE). The
multi-task elastic net must share support across tasks (zero whole feature rows).
The quantile forest must give calibrated intervals and monotone, spread-aware
quantiles.
"""
import numpy as np
import pytest

from nupyml.ordinal import OrdinalLogistic, OrdinalRidge
from nupyml.linear_model import MultiTaskElasticNet, MultiTaskLasso, LogisticRegression
from nupyml.ensemble import QuantileForest


def _ordinal_data(seed=0, n=600):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    score = X @ np.array([1.5, -1.0, 0.5, 0.0]) + 0.5 * rng.randn(n)
    y = np.digitize(score, [-1.0, 0.0, 1.2])       # 4 ordered classes 0..3
    return X[:400], y[:400], X[400:], y[400:]


# --- ordinal --------------------------------------------------------------

def test_ordinal_logistic_respects_order_better_than_a_classifier():
    Xtr, ytr, Xte, yte = _ordinal_data()
    ordi = OrdinalLogistic().fit(Xtr, ytr)
    nominal = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    mae_ord = np.mean(np.abs(ordi.predict(Xte) - yte))
    mae_nom = np.mean(np.abs(nominal.predict(Xte) - yte))
    # the ordinal model's mistakes are, on average, nearer the true rank
    assert mae_ord <= mae_nom + 0.02
    assert mae_ord < 0.5


def test_ordinal_logistic_thresholds_are_ordered_and_probs_normalised():
    Xtr, ytr, Xte, yte = _ordinal_data()
    ordi = OrdinalLogistic().fit(Xtr, ytr)
    assert np.all(np.diff(ordi.thresholds_) > 0)   # thresholds strictly increasing
    p = ordi.predict_proba(Xte)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.shape[1] == 4


def test_ordinal_ridge_runs_and_predicts_valid_classes():
    Xtr, ytr, Xte, yte = _ordinal_data()
    orr = OrdinalRidge().fit(Xtr, ytr)
    pred = orr.predict(Xte)
    assert set(np.unique(pred)).issubset(set(np.unique(ytr)))
    assert np.mean(np.abs(pred - yte)) < 0.6


# --- multi-task elastic net -----------------------------------------------

def test_multitask_elasticnet_shares_support():
    rng = np.random.RandomState(0)
    n = 400
    X = rng.randn(n, 6)
    W = rng.randn(6, 3)
    W[3:] = 0.0                                     # features 3,4,5 irrelevant
    Y = X @ W + 0.1 * rng.randn(n, 3)
    mten = MultiTaskElasticNet(alpha=0.3, l1_ratio=0.8).fit(X, Y)
    # irrelevant features should be zeroed for ALL tasks at once (row-sparse)
    zero_rows = np.all(np.abs(mten.coef_.T) < 1e-6, axis=1)
    assert zero_rows[3:].sum() >= 2
    # relevant features are kept
    assert not zero_rows[0]


def test_multitask_elasticnet_l1_ratio_one_matches_lasso():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 5)
    Y = X @ rng.randn(5, 2) + 0.1 * rng.randn(200, 2)
    en = MultiTaskElasticNet(alpha=0.5, l1_ratio=1.0).fit(X, Y)
    la = MultiTaskLasso(alpha=0.5).fit(X, Y)
    assert np.allclose(en.coef_, la.coef_, atol=1e-3)


# --- quantile forest ------------------------------------------------------

@pytest.fixture
def hetero_data():
    rng = np.random.RandomState(0)
    n = 800
    X = rng.uniform(-2, 2, (n, 3))
    # noise grows with |x1| -- a heteroscedastic target
    y = 2 * X[:, 0] + (0.3 + np.abs(X[:, 1])) * rng.randn(n)
    return X[:600], y[:600], X[600:], y[600:]


def test_quantile_forest_interval_coverage_is_near_nominal(hetero_data):
    Xtr, ytr, Xte, yte = hetero_data
    qf = QuantileForest(n_estimators=100, min_samples_leaf=10,
                        random_state=0).fit(Xtr, ytr)
    lo, hi = qf.predict_interval(Xte, coverage=0.8)
    coverage = np.mean((yte >= lo) & (yte <= hi))
    assert 0.7 <= coverage <= 0.9                   # ~80% nominal


def test_quantile_forest_quantiles_are_ordered(hetero_data):
    Xtr, ytr, Xte, yte = hetero_data
    qf = QuantileForest(n_estimators=80, min_samples_leaf=10,
                        random_state=0).fit(Xtr, ytr)
    q = qf.predict_quantile(Xte, [0.1, 0.5, 0.9])
    assert np.all(q[:, 0] <= q[:, 1] + 1e-9)
    assert np.all(q[:, 1] <= q[:, 2] + 1e-9)


def test_quantile_forest_widens_where_noise_is_larger(hetero_data):
    Xtr, ytr, Xte, yte = hetero_data
    qf = QuantileForest(n_estimators=100, min_samples_leaf=10,
                        random_state=0).fit(Xtr, ytr)
    lo, hi = qf.predict_interval(Xte, coverage=0.8)
    width = hi - lo
    # wider intervals where |x1| (the noise driver) is large vs small
    high_noise = np.abs(Xte[:, 1]) > 1.3
    low_noise = np.abs(Xte[:, 1]) < 0.4
    assert width[high_noise].mean() > width[low_noise].mean()
