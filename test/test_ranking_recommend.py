"""F11: ranking metrics, learning-to-rank, and two more recommenders.

Metrics are checked against known values; the LTR models must beat a random
ordering on linearly-rankable query groups; Slope One and co-clustering CF must
beat the global-mean baseline on a low-rank rating matrix.
"""
import numpy as np
import pytest

from nupyml.ranking import (dcg_at_k, ndcg_at_k, average_precision,
                           mean_average_precision, mean_reciprocal_rank,
                           hit_rate_at_k, ndcg_from_scores, RankNet, LambdaMART)
from nupyml.recommend import SlopeOne, CoClusteringCF


# --- metrics --------------------------------------------------------------

def test_ndcg_is_one_for_ideal_order_and_less_otherwise():
    assert ndcg_at_k([3, 2, 1, 0]) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2, 3]) < 1.0
    # reverse order is the worst arrangement of the same relevances
    assert ndcg_at_k([0, 1, 2, 3]) < ndcg_at_k([3, 0, 2, 1])


def test_dcg_discounts_later_positions():
    # the same relevant item earns less deeper down
    assert dcg_at_k([1, 0, 0]) > dcg_at_k([0, 0, 1])


def test_average_precision_known_value():
    # relevant at ranks 1,3,5 -> mean of precisions 1/1, 2/3, 3/5
    ap = average_precision([1, 0, 1, 0, 1])
    assert ap == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)


def test_mrr_and_hit_rate():
    lists = [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
    assert mean_reciprocal_rank(lists) == pytest.approx((1 / 3 + 1 + 0) / 3)
    assert hit_rate_at_k(lists, k=1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(lists, k=3) == pytest.approx(2 / 3)


def test_ndcg_of_empty_relevance_is_zero():
    assert ndcg_at_k([0, 0, 0]) == 0.0


# --- learning to rank -----------------------------------------------------

def _ranking_data(seed=0, nq=40, items=8, d=5):
    rng = np.random.RandomState(seed)
    w = rng.randn(d)
    X, y, groups = [], [], []
    for _ in range(nq):
        Xg = rng.randn(items, d)
        rel = Xg @ w
        rel = rel - rel.min()
        rel = np.round(3 * rel / (rel.max() + 1e-9)).astype(int)
        X.append(Xg); y.append(rel); groups.append(items)
    return np.vstack(X), np.concatenate(y), groups


def _mean_ndcg(model, X, y, groups, k=5):
    start, nd = 0, []
    for g in groups:
        sl = slice(start, start + g); start += g
        nd.append(ndcg_from_scores(model.predict(X[sl]), y[sl], k))
    return np.mean(nd)


def test_ranknet_beats_random_ordering():
    X, y, g = _ranking_data()
    rn = RankNet(n_epochs=200, random_state=0).fit(X, y, g)
    rng = np.random.RandomState(1)
    start, rand = 0, []
    for gi in g:
        sl = slice(start, start + gi); start += gi
        rand.append(ndcg_from_scores(rng.randn(gi), y[sl], 5))
    assert _mean_ndcg(rn, X, y, g) > np.mean(rand) + 0.1
    assert _mean_ndcg(rn, X, y, g) > 0.85


def test_lambdamart_beats_random_ordering():
    X, y, g = _ranking_data()
    lm = LambdaMART(n_estimators=40, max_depth=3, random_state=0).fit(X, y, g)
    assert _mean_ndcg(lm, X, y, g) > 0.8
    # rank() is consistent with predict()
    order = lm.rank(X[:8])
    assert np.array_equal(order, np.argsort(-lm.predict(X[:8])))


# --- recommenders ---------------------------------------------------------

@pytest.fixture
def ratings():
    rng = np.random.RandomState(0)
    nu, ni, k = 40, 25, 3
    full = (rng.rand(nu, k) @ rng.rand(k, ni)) * 4 + 1
    tr, te = [], []
    for u in range(nu):
        for i in range(ni):
            if rng.rand() < 0.6:
                tr.append((u, i, full[u, i]))
            elif rng.rand() < 0.3:
                te.append((u, i, full[u, i]))
    return tr, te


def _rmse(model, te):
    return np.sqrt(np.mean([(model.predict(u, i) - r) ** 2 for u, i, r in te]))


def _baseline(tr, te):
    gm = np.mean([r for _, _, r in tr])
    return np.sqrt(np.mean([(gm - r) ** 2 for _, _, r in te]))


def test_slope_one_beats_global_mean(ratings):
    tr, te = ratings
    assert _rmse(SlopeOne().fit(tr), te) < _baseline(tr, te)


def test_coclustering_cf_beats_global_mean(ratings):
    tr, te = ratings
    model = CoClusteringCF(random_state=0).fit(tr)
    assert _rmse(model, te) < _baseline(tr, te)


def test_slope_one_falls_back_for_unseen_indices(ratings):
    tr, te = ratings
    so = SlopeOne().fit(tr)
    # a user/item beyond the training range -> the global mean, no crash
    assert so.predict(9999, 9999) == pytest.approx(so.global_mean_)
