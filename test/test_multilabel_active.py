"""Multi-label classification and active learning.

Multi-label tests build a target with both correlated and independent labels and
check each method predicts the set well; active-learning tests check the query
strategies pick boundary/diverse points rather than random ones.
"""
import numpy as np
import pytest

from nupyml.linear_model import LogisticRegression
from nupyml.multilabel import (BinaryRelevance, ClassifierChain, LabelPowerset,
                               RAkEL, MLkNN)
from nupyml.active import (uncertainty_sampling, margin_sampling,
                          entropy_sampling, query_by_committee,
                          expected_model_change, core_set)


@pytest.fixture
def multilabel_data():
    """Three labels: two correlated (share features 0,1), one independent."""
    rng = np.random.RandomState(0)
    n = 600
    X = rng.normal(size=(n, 6))
    y0 = (X[:, 0] + X[:, 1] > 0).astype(int)
    y1 = (X[:, 0] + X[:, 1] > 0.5).astype(int)       # correlated with y0
    y2 = (X[:, 2] - X[:, 3] > 0).astype(int)         # independent
    Y = np.column_stack([y0, y1, y2])
    return X[:400], X[400:], Y[:400], Y[400:]


def _subset_accuracy(Yt, Yp):
    return np.mean(np.all(Yt == Yp, axis=1))


def _hamming(Yt, Yp):
    return 1 - np.mean(Yt != Yp)


@pytest.mark.parametrize("clf_factory", [
    lambda: BinaryRelevance(),
    lambda: ClassifierChain(random_state=0),
    lambda: LabelPowerset(),
    lambda: RAkEL(random_state=0),
    lambda: MLkNN(k=10),
])
def test_multilabel_classifier_predicts_the_label_set(clf_factory, multilabel_data):
    Xtr, Xte, Ytr, Yte = multilabel_data
    clf = clf_factory().fit(Xtr, Ytr)
    pred = clf.predict(Xte)
    assert pred.shape == Yte.shape
    assert _hamming(Yte, pred) > 0.85               # most labels correct


def test_binary_relevance_output_is_binary(multilabel_data):
    Xtr, Xte, Ytr, Yte = multilabel_data
    pred = BinaryRelevance().fit(Xtr, Ytr).predict(Xte)
    assert set(np.unique(pred)).issubset({0, 1})


def test_classifier_chain_uses_earlier_labels(multilabel_data):
    """The chain feeds earlier labels forward, so it should predict the coherent
    label sets at least as well as independent binary relevance on correlated
    labels."""
    Xtr, Xte, Ytr, Yte = multilabel_data
    chain = ClassifierChain(random_state=0).fit(Xtr, Ytr)
    br = BinaryRelevance().fit(Xtr, Ytr)
    assert _subset_accuracy(Yte, chain.predict(Xte)) >= \
        _subset_accuracy(Yte, br.predict(Xte)) - 0.05


def test_label_powerset_only_predicts_seen_combinations(multilabel_data):
    """Every prediction must be a label-set that appeared in training -- the
    defining property (and limitation) of the powerset transform."""
    Xtr, Xte, Ytr, Yte = multilabel_data
    lp = LabelPowerset().fit(Xtr, Ytr)
    seen = {tuple(row) for row in np.unique(Ytr, axis=0)}
    pred = lp.predict(Xte)
    assert all(tuple(row) in seen for row in pred)


def test_rakel_bounds_the_class_explosion(multilabel_data):
    """Each internal powerset is over only `labelset_size` labels, so it never
    faces the full 2^n_labels class blow-up."""
    Xtr, Xte, Ytr, Yte = multilabel_data
    rakel = RAkEL(labelset_size=2, n_models=8, random_state=0).fit(Xtr, Ytr)
    assert all(len(s) == 2 for s in rakel.subsets_)
    assert rakel.predict(Xte).shape == Yte.shape


def test_mlknn_handles_a_rare_label():
    """MLkNN's Bayesian prior lets it still predict a rare label rather than
    always suppressing it the way a bare neighbour-vote would."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 4))
    common = (X[:, 0] > 0).astype(int)
    rare = (X[:, 1] > 1.5).astype(int)              # ~7% positive
    Y = np.column_stack([common, rare])
    ml = MLkNN(k=10).fit(X, Y)
    pred = ml.predict(X)
    assert pred.shape == Y.shape
    assert pred[:, 1].sum() > 0                     # it predicts the rare label at all


# --- active learning ------------------------------------------------------

@pytest.fixture
def boundary_pool():
    """A linear problem: the boundary is x0 = 0, so informative points have small
    |x0|."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=300).fit(X[:40], y[:40])
    return model, X[40:], y[40:]


def test_uncertainty_sampling_picks_boundary_points(boundary_pool):
    model, pool, _ = boundary_pool
    idx = uncertainty_sampling(model, pool, n_instances=10)
    # the queried points sit far closer to the x0=0 boundary than average
    assert np.abs(pool[idx, 0]).mean() < np.abs(pool[:, 0]).mean() / 2


def test_margin_sampling_picks_boundary_points(boundary_pool):
    model, pool, _ = boundary_pool
    idx = margin_sampling(model, pool, n_instances=10)
    assert np.abs(pool[idx, 0]).mean() < np.abs(pool[:, 0]).mean() / 2


def test_entropy_sampling_picks_boundary_points(boundary_pool):
    model, pool, _ = boundary_pool
    idx = entropy_sampling(model, pool, n_instances=10)
    assert np.abs(pool[idx, 0]).mean() < np.abs(pool[:, 0]).mean() / 2


def test_query_strategies_return_the_requested_count(boundary_pool):
    model, pool, _ = boundary_pool
    for fn in (uncertainty_sampling, margin_sampling, entropy_sampling,
               expected_model_change):
        assert len(fn(model, pool, n_instances=7)) == 7


def test_query_by_committee_scores_disagreement():
    """Points where committee members disagree should be preferred."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    # a committee trained on different small subsets disagrees near the boundary
    committee = [LogisticRegression(max_iter=200).fit(
        X[i * 20:(i + 1) * 20 + 30], y[i * 20:(i + 1) * 20 + 30]) for i in range(3)]
    idx = query_by_committee(committee, X, n_instances=10)
    assert len(idx) == 10
    # queried points are nearer the boundary than average (where models disagree)
    assert np.abs(X[idx, 0]).mean() < np.abs(X[:, 0]).mean()


def test_core_set_selects_diverse_points():
    """Core-set should spread its picks out, not cluster them -- the minimum
    pairwise distance among its picks should exceed that of a random pick."""
    rng = np.random.RandomState(0)
    pool = rng.normal(size=(300, 2))
    idx = core_set(pool, X_labelled=None, n_instances=10)
    from scipy.spatial.distance import pdist
    core_spread = pdist(pool[idx]).min()
    random_spread = pdist(pool[rng.choice(300, 10, replace=False)]).min()
    assert core_spread >= random_spread


def test_core_set_covers_the_space():
    """Every pool point should end up near some selected point -- the coverage
    guarantee uncertainty sampling lacks."""
    rng = np.random.RandomState(0)
    pool = rng.uniform(-5, 5, (200, 2))
    idx = core_set(pool, n_instances=20)
    from scipy.spatial.distance import cdist
    coverage = cdist(pool, pool[idx]).min(axis=1).max()   # worst-covered point
    # 20 centers over a 10x10 box: nothing should be too far from a center
    assert coverage < 4.0
