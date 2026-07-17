"""Category encoders, feature-engineering transformers, and advanced selection.

Encoder tests plant a category-target relationship and check the encoding
reflects it (while shrinking rare levels); engineering tests check outliers are
clipped, rare levels folded, and cycles wrapped; selection tests plant a few
informative features among noise and check each selector finds them.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification, make_regression
from nupyml.encoders import (WOEEncoder, JamesSteinEncoder, MEstimateEncoder,
                            LeaveOneOutEncoder, BinaryEncoder, CountEncoder,
                            Winsorizer, RareLabelEncoder, CyclicalEncoder)
from nupyml.feature_selection import Boruta, mrmr, relieff, StabilitySelection


@pytest.fixture
def categorical_target():
    """Three levels whose positive rate decreases a -> b -> c."""
    rng = np.random.RandomState(0)
    cats = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 200).reshape(-1, 1)
    y = np.concatenate([rng.binomial(1, 0.8, 200), rng.binomial(1, 0.5, 200),
                        rng.binomial(1, 0.2, 200)])
    return cats, y


# --- target encoders ------------------------------------------------------

def test_woe_orders_categories_by_log_odds(categorical_target):
    cats, y = categorical_target
    woe = WOEEncoder().fit(cats, y)
    e = woe.transform([["a"], ["b"], ["c"]]).ravel()
    assert e[0] > e[1] > e[2]                    # a (0.8) > b (0.5) > c (0.2)
    assert e[0] > 0 > e[2]                        # positive vs negative log-odds


def test_james_stein_recovers_category_means(categorical_target):
    cats, y = categorical_target
    js = JamesSteinEncoder().fit(cats, y)
    assert js.transform([["a"]])[0, 0] == pytest.approx(0.8, abs=0.1)
    assert js.transform([["c"]])[0, 0] == pytest.approx(0.2, abs=0.1)


def test_james_stein_shrinks_a_rare_category_hard():
    """A category seen a handful of times gets pulled toward the global mean,
    while a frequent one keeps its own mean -- reliability-driven shrinkage."""
    rng = np.random.RandomState(0)
    cats = np.array(["common"] * 300 + ["rare"] * 4).reshape(-1, 1)
    y = np.concatenate([rng.binomial(1, 0.5, 300), np.ones(4)])   # rare looks 100%
    js = JamesSteinEncoder().fit(cats, y)
    global_mean = y.mean()
    rare_enc = js.transform([["rare"]])[0, 0]
    # the rare category's raw mean is 1.0; shrinkage pulls it well below that
    assert rare_enc < 1.0
    assert abs(rare_enc - global_mean) < abs(1.0 - global_mean)


def test_m_estimate_smoothing_controls_shrinkage(categorical_target):
    cats, y = categorical_target
    light = MEstimateEncoder(m=1).fit(cats, y).transform([["a"]])[0, 0]
    heavy = MEstimateEncoder(m=100).fit(cats, y).transform([["a"]])[0, 0]
    # heavier smoothing pulls 'a' closer to the global mean
    gm = y.mean()
    assert abs(heavy - gm) < abs(light - gm)


def test_leave_one_out_excludes_the_row_itself():
    """A singleton category's training encoding must NOT be its own label -- the
    leak LOO removes."""
    cats = np.array([["x"], ["x"], ["y"]])
    y = np.array([1.0, 0.0, 1.0])
    loo = LeaveOneOutEncoder()
    tr = loo.fit_transform(cats, y)
    # row 0 is category x: its LOO mean excludes itself -> just row 1's value (0)
    assert tr[0, 0] == pytest.approx(0.0)
    assert tr[1, 0] == pytest.approx(1.0)         # excludes itself -> row 0's (1)


# --- target-free encoders -------------------------------------------------

def test_binary_encoder_is_compact(categorical_target):
    """log2 columns instead of one-hot's n."""
    cats, _ = categorical_target
    be = BinaryEncoder().fit(cats)
    encoded = be.transform(cats)
    assert encoded.shape[1] == 2                  # 3 categories -> 2 bits < 3
    assert set(np.unique(encoded)).issubset({0.0, 1.0})


def test_count_encoder_reports_frequency(categorical_target):
    cats, _ = categorical_target
    ce = CountEncoder().fit(cats)
    assert ce.transform([["a"]])[0, 0] == 200      # 'a' appears 200 times


# --- feature engineering --------------------------------------------------

def test_winsorizer_clips_outliers():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    w = Winsorizer(lower=0, upper=80).fit(X)
    out = w.transform(X)
    assert out.max() < 100                        # the outlier is pulled in
    assert out.min() == 1.0                        # low end untouched here


def test_rare_label_encoder_groups_infrequent_levels():
    rare = np.array(["x"] * 100 + ["y"] * 100 + ["z"] * 3).reshape(-1, 1)
    rl = RareLabelEncoder(tol=0.05).fit(rare)
    out = np.unique(rl.transform(rare))
    assert "Rare" in out and "z" not in out       # z folded into Rare
    assert "x" in out and "y" in out              # frequent levels kept


def test_cyclical_encoder_wraps_around():
    """Hour 23 and hour 0 should be neighbours after (sin, cos) encoding, unlike
    their raw integers."""
    hours = np.array([[0], [12], [23]])
    enc = CyclicalEncoder(period=24).fit(hours).transform(hours)
    d_23_0 = np.linalg.norm(enc[2] - enc[0])       # adjacent on the clock
    d_0_12 = np.linalg.norm(enc[0] - enc[1])       # opposite on the clock
    assert d_23_0 < d_0_12


def test_cyclical_encoder_output_lies_on_the_unit_circle():
    x = np.arange(24).reshape(-1, 1)
    enc = CyclicalEncoder(period=24).fit(x).transform(x)
    # sin^2 + cos^2 = 1 for every point
    assert np.allclose(enc[:, 0] ** 2 + enc[:, 1] ** 2, 1.0)


# --- advanced feature selection -------------------------------------------

@pytest.fixture
def sparse_signal():
    return make_classification(n_samples=400, n_features=10, n_informative=3,
                               random_state=0)


def test_boruta_selects_informative_features(sparse_signal):
    X, y = sparse_signal
    bor = Boruta(n_iter=15, random_state=0).fit(X, y)
    # it keeps a handful (the relevant ones), not all ten, not none
    assert 1 <= bor.support_.sum() <= 7


def test_boruta_transform_matches_support(sparse_signal):
    X, y = sparse_signal
    bor = Boruta(n_iter=15, random_state=0).fit(X, y)
    assert bor.transform(X).shape[1] == bor.support_.sum()


def test_mrmr_selects_non_redundant_features():
    """mRMR should avoid picking near-duplicate features -- if two columns are
    copies, it should not select both."""
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=400, n_features=6, n_informative=3,
                               random_state=0)
    # append an exact copy of feature 0
    X = np.column_stack([X, X[:, 0]])
    selected = mrmr(X, y, n_features=4, discrete=True)
    # it should not pick both the original feature 0 and its copy (index 6)
    assert not (0 in selected and 6 in selected)


def test_relieff_ranks_informative_features_highly(sparse_signal):
    X, y = sparse_signal
    weights = relieff(X, y, random_state=0)
    # the top-weighted features should overlap the informative ones (0,1,2)
    top3 = set(np.argsort(weights)[::-1][:3])
    assert len(top3 & {0, 1, 2}) >= 2


def test_stability_selection_keeps_consistent_features():
    X, y = make_regression(n_samples=400, n_features=10, n_informative=3,
                           noise=5.0, random_state=0)
    ss = StabilitySelection(n_bootstrap=30, threshold=0.5, random_state=0).fit(
        X, y.astype(float))
    assert 1 <= ss.support_.sum() <= 6
    # selection frequencies are genuine fractions
    assert np.all((ss.selection_frequency_ >= 0) & (ss.selection_frequency_ <= 1))
