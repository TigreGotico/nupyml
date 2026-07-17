"""Sketches, approximate search, streaming learners, patterns, recommenders.

Each structure is held to the guarantee it trades exactness for: the Bloom filter
never gives a false negative, HyperLogLog counts distinct items to a bounded
error, LSH recovers most true neighbours, the three rule miners agree, and the
recommenders reconstruct a planted low-rank matrix.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification
from nupyml.model_selection import train_test_split
from nupyml.search import (BloomFilter, CountMinSketch, HyperLogLog,
                           ReservoirSampler, TDigest, MinHash,
                           RandomProjectionLSH, MinHashLSH, IVFIndex,
                           ProductQuantizer)
from nupyml.streaming import (FTRLProximal, Hedge, OnlineGradientDescent,
                              HoeffdingTree)
from nupyml.patterns import apriori, fpgrowth, eclat, association_rules
from nupyml.recommend import (MatrixFactorization, ALS, BPR,
                              FactorizationMachine)


# --- sketches -------------------------------------------------------------

def test_bloom_filter_has_no_false_negatives():
    """The one-sided guarantee: everything added is always reported present."""
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    added = [f"item{i}" for i in range(500)]
    for item in added:
        bf.add(item)
    assert all(item in bf for item in added)      # never a false negative


def test_bloom_filter_false_positive_rate_is_controlled():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    for i in range(1000):
        bf.add(f"item{i}")
    # things never added are mostly reported absent, at roughly the target rate
    false_positives = sum(f"absent{i}" in bf for i in range(2000))
    assert false_positives / 2000 < 0.05


def test_count_min_never_underestimates():
    """Collisions only add, so every estimate is an over-estimate -- heavy
    hitters are near-exact."""
    cm = CountMinSketch(width=1000, depth=5)
    for _ in range(1000):
        cm.add("hot")
    for _ in range(5):
        cm.add("cold")
    assert cm.estimate("hot") >= 1000              # never under
    assert cm.estimate("cold") >= 5
    assert cm.estimate("hot") == pytest.approx(1000, abs=20)


def test_hyperloglog_counts_distinct_items():
    hll = HyperLogLog(p=12)
    for i in range(50000):
        hll.add(f"user{i}")
    estimate = hll.count()
    assert abs(estimate - 50000) / 50000 < 0.05    # within a few percent


def test_hyperloglog_ignores_duplicates():
    """It counts DISTINCT items, so repeats must not inflate it."""
    hll = HyperLogLog(p=10)
    for _ in range(10000):
        hll.add("same")
    assert hll.count() < 5                          # essentially one distinct item


def test_reservoir_sample_is_uniform():
    """Every item seen has equal probability of being retained, whatever the
    stream length -- checked by the sample mean matching the population mean."""
    means = []
    for seed in range(30):
        rs = ReservoirSampler(k=100, random_state=seed)
        for i in range(10000):
            rs.add(i)
        means.append(np.mean(rs.sample()))
    assert abs(np.mean(means) - 4999.5) < 300       # centred on the true mean


def test_reservoir_holds_exactly_k():
    rs = ReservoirSampler(k=50, random_state=0)
    for i in range(1000):
        rs.add(i)
    assert len(rs.sample()) == 50


def test_tdigest_estimates_quantiles_at_the_tail():
    """The point of t-digest: accurate extreme quantiles, where a fixed histogram
    is worst."""
    rng = np.random.RandomState(0)
    td = TDigest(compression=100)
    data = rng.normal(0, 1, 20000)
    for v in data:
        td.add(v)
    assert abs(td.quantile(0.5) - 0.0) < 0.1
    assert abs(td.quantile(0.99) - np.percentile(data, 99)) < 0.3


def test_minhash_estimates_jaccard():
    """The MinHash identity: signature agreement estimates Jaccard similarity."""
    mh = MinHash(n_perm=256)
    a = set(range(100))
    b = set(range(50, 150))                          # true Jaccard = 50/150 = 1/3
    est = MinHash.jaccard(mh.signature(a), mh.signature(b))
    assert abs(est - 1 / 3) < 0.1


# --- approximate nearest neighbours ---------------------------------------

@pytest.fixture
def ann_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(2000, 20))
    query = X[0] + 0.01 * rng.normal(size=20)        # very near point 0
    exact = np.argsort(np.linalg.norm(X - query, axis=1))[:5]
    return X, query, exact


def test_lsh_recovers_most_true_neighbours(ann_data):
    X, query, exact = ann_data
    lsh = RandomProjectionLSH(n_bits=10, n_tables=12, random_state=0).fit(X)
    found = set(lsh.query(query, k=5))
    assert 0 in found                                # the near-duplicate is found
    assert len(found & set(exact)) >= 3              # decent recall


def test_lsh_ranking_is_exact_among_candidates(ann_data):
    """LSH approximates WHICH points are considered, but ranks them exactly."""
    X, query, exact = ann_data
    lsh = RandomProjectionLSH(n_bits=8, n_tables=15, random_state=0).fit(X)
    found = lsh.query(query, k=5)
    dists = np.linalg.norm(X[found] - query, axis=1)
    assert np.all(np.diff(dists) >= 0)               # returned in true distance order


def test_ivf_recall_improves_with_more_probes(ann_data):
    """n_probe is an honest accuracy dial: probing more cells recovers more true
    neighbours."""
    X, query, exact = ann_data
    few = IVFIndex(n_clusters=50, n_probe=1, random_state=0).fit(X)
    many = IVFIndex(n_clusters=50, n_probe=15, random_state=0).fit(X)
    recall_few = len(set(few.query(query, 5)) & set(exact))
    recall_many = len(set(many.query(query, 5)) & set(exact))
    assert recall_many >= recall_few


def test_product_quantizer_compresses_and_finds_neighbours(ann_data):
    X, query, exact = ann_data
    pq = ProductQuantizer(n_subvectors=4, n_codes=64, random_state=0).fit(X)
    found = set(pq.query(query, k=10))
    assert len(found & set(exact)) >= 2              # approximate but useful
    assert pq.codes_.shape == (2000, 4)              # 20 floats -> 4 codes


def test_minhash_lsh_finds_near_duplicates():
    """Banded MinHash should retrieve sets above the Jaccard threshold."""
    rng = np.random.RandomState(0)
    base = set(range(100))
    sets = [base]                                    # index 0: the query itself
    sets.append(set(range(20, 120)))                 # ~0.66 Jaccard with base
    sets += [set(rng.randint(0, 1000, 100)) for _ in range(50)]   # unrelated
    lsh = MinHashLSH(n_bands=16, rows_per_band=4, random_state=0).fit(sets)
    matches = lsh.query(base, threshold=0.4)
    assert 0 in matches and 1 in matches             # both similar sets retrieved


# --- streaming ------------------------------------------------------------

def test_ftrl_learns_online():
    X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ftrl = FTRLProximal(l1=1.0, l2=1.0).fit(Xtr, ytr)
    assert (ftrl.predict(Xte) == yte).mean() > 0.8


def test_ftrl_produces_a_sparse_model_that_keeps_the_signal():
    """FTRL's whole reason to exist: genuine sparsity that drops noise features
    while keeping the informative ones. Plain SGD with L2 does not do this."""
    rng = np.random.RandomState(0)
    n = 3000
    signal = (rng.uniform(size=(n, 5)) > 0.5).astype(float)
    logit = signal @ [2, -2, 1.5, -1.5, 1.0]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    noise = (rng.uniform(size=(n, 45)) > 0.5).astype(float)
    X = np.hstack([signal, noise])

    ftrl = FTRLProximal(l1=8.0, l2=1.0).fit(X, y)
    active = np.abs(ftrl.z_) > ftrl.l1
    assert active[:5].sum() == 5                     # every signal feature kept
    assert active[5:].sum() < 40                     # much of the noise dropped
    assert ftrl.sparsity_ > 0.1


def test_online_gradient_descent_learns():
    X, y = make_classification(n_samples=1500, n_features=15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ogd = OnlineGradientDescent(learning_rate=0.1).fit(Xtr, ytr)
    assert (ogd.predict(Xte) == yte).mean() > 0.8


def test_hedge_concentrates_on_the_best_expert():
    """Multiplicative weights: the reliably-good expert should accumulate the
    mass, and regret against it should be small."""
    rng = np.random.RandomState(0)
    hedge = Hedge(3, learning_rate=0.5).reset()
    for _ in range(500):
        losses = np.clip(np.array([0.1, 0.5, 0.9]) + rng.normal(0, 0.05, 3), 0, 1)
        hedge.update(losses)
    assert hedge.weights_[0] > 0.8
    assert hedge.regret() < 5.0                      # small, sublinear regret


def test_hedge_regret_is_measured_against_the_best():
    rng = np.random.RandomState(0)
    hedge = Hedge(4, learning_rate=0.3).reset()
    for _ in range(300):
        hedge.update(rng.uniform(0, 1, 4))
    # regret is nonnegative: you cannot beat the best expert in hindsight
    assert hedge.regret() >= -1e-9


def test_hoeffding_tree_learns_from_a_stream():
    X, y = make_classification(n_samples=5000, n_features=6, n_informative=4,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ht = HoeffdingTree(n_classes=2, grace_period=50, delta=1e-3).fit(Xtr, ytr)
    assert (ht.predict(Xte) == yte).mean() > 0.8


def test_hoeffding_tree_grows_by_splitting():
    """A confident split turns a leaf into an internal node -- the tree should
    have grown past a single root."""
    X, y = make_classification(n_samples=5000, n_features=6, n_informative=4,
                               random_state=0)
    ht = HoeffdingTree(n_classes=2, grace_period=50, delta=1e-2).fit(X, y)
    assert ht.n_nodes() > 1


def test_hoeffding_tree_updates_incrementally():
    """It must learn from one example at a time without ever seeing the batch.

    The rule is axis-aligned (one feature dominates) on purpose: with two equally
    good features the Hoeffding gap between them never exceeds epsilon and the
    tree correctly REFUSES to split -- a real property of the bound, not a bug, so
    a fair test gives it a clear winner to find.
    """
    rng = np.random.RandomState(0)
    ht = HoeffdingTree(n_classes=2, grace_period=30, delta=1e-2)
    for _ in range(3000):
        x = rng.normal(size=4)
        y = int(x[0] > 0.0)                          # depends on one feature
        ht.partial_fit(x, y)
    Xte = rng.normal(size=(500, 4))
    yte = (Xte[:, 0] > 0.0).astype(int)
    assert (ht.predict(Xte) == yte).mean() > 0.75


# --- association rules ----------------------------------------------------

@pytest.fixture
def transactions():
    base = [["bread", "milk"],
            ["bread", "diaper", "beer", "eggs"],
            ["milk", "diaper", "beer", "cola"],
            ["bread", "milk", "diaper", "beer"],
            ["bread", "milk", "diaper", "cola"]]
    return base * 20


def test_the_three_miners_agree(transactions):
    """Apriori, FP-Growth and ECLAT compute the SAME frequent itemsets by three
    different routes -- a strong cross-check that all three are correct."""
    a = apriori(transactions, min_support=0.3)
    f = fpgrowth(transactions, min_support=0.3)
    e = eclat(transactions, min_support=0.3)
    assert set(a) == set(f) == set(e)
    assert all(abs(a[k] - f[k]) < 1e-9 and abs(a[k] - e[k]) < 1e-9 for k in a)


def test_apriori_respects_the_support_threshold(transactions):
    freq = apriori(transactions, min_support=0.5)
    assert all(s >= 0.5 - 1e-9 for s in freq.values())


def test_apriori_principle_prunes_supersets(transactions):
    """A frequent superset implies all its subsets are frequent (downward
    closure) -- the property the pruning relies on."""
    freq = apriori(transactions, min_support=0.3)
    from itertools import combinations
    for itemset in freq:
        for r in range(1, len(itemset)):
            for sub in combinations(itemset, r):
                assert frozenset(sub) in freq        # every subset is present


def test_association_rules_enforce_lift(transactions):
    """Lift, not just confidence: every returned rule has A and B co-occurring
    more than independence predicts."""
    freq = apriori(transactions, min_support=0.3)
    rules = association_rules(freq, min_confidence=0.5, min_lift=1.01)
    assert rules
    assert all(r["lift"] >= 1.01 for r in rules)
    assert all(0 <= r["confidence"] <= 1 for r in rules)


# --- recommenders ---------------------------------------------------------

@pytest.fixture
def low_rank_ratings():
    """A planted rank-3 user-item matrix, sparsely observed."""
    rng = np.random.RandomState(0)
    n_u, n_i, k = 100, 80, 3
    P, Q = rng.normal(size=(n_u, k)), rng.normal(size=(n_i, k))
    R = P @ Q.T
    triples = [(u, i, R[u, i]) for u in range(n_u) for i in range(n_i)
               if rng.uniform() < 0.3]
    return triples, R, n_u, n_i


def test_matrix_factorization_reconstructs_a_low_rank_matrix(low_rank_ratings):
    triples, R, n_u, n_i = low_rank_ratings
    mf = MatrixFactorization(n_factors=3, n_epochs=80, random_state=0).fit(triples)
    rng = np.random.RandomState(1)
    errs = [(mf.predict(rng.randint(n_u), rng.randint(n_i)))
            for _ in range(200)]
    # training RMSE should fall well below the data's own spread
    assert mf.history_[-1] < R.std() / 2


def test_als_reconstructs_a_low_rank_matrix(low_rank_ratings):
    """ALS solves each half exactly, so it should fit the low-rank structure
    tightly."""
    triples, R, n_u, n_i = low_rank_ratings
    als = ALS(n_factors=3, n_iter=20, random_state=0).fit(triples)
    assert als.history_[-1] < R.std() / 4


def test_matrix_factorization_recommends(low_rank_ratings):
    triples, R, n_u, n_i = low_rank_ratings
    mf = MatrixFactorization(n_factors=3, n_epochs=40, random_state=0).fit(triples)
    recs = mf.recommend(0, n=5)
    assert len(recs) == 5


def test_bpr_ranks_liked_above_disliked(low_rank_ratings):
    """The implicit-feedback objective: a touched item should outrank an
    untouched one. Checked against the true preference matrix."""
    triples, R, n_u, n_i = low_rank_ratings
    interactions = [(u, i) for u in range(n_u) for i in np.argsort(R[u])[-5:]]
    bpr = BPR(n_factors=3, n_epochs=50, random_state=0).fit(interactions)
    hits = 0
    for u in range(n_u):
        liked, disliked = np.argsort(R[u])[-1], np.argsort(R[u])[0]
        hits += bpr.score(u, liked) > bpr.score(u, disliked)
    assert hits > 85                                 # right for the vast majority


def test_bpr_does_not_recommend_known_items(low_rank_ratings):
    triples, R, n_u, n_i = low_rank_ratings
    interactions = [(u, i) for u in range(n_u) for i in np.argsort(R[u])[-5:]]
    bpr = BPR(n_factors=3, n_epochs=30, random_state=0).fit(interactions)
    recs = set(bpr.recommend(0, n=10))
    assert not (recs & bpr.user_items_[0])           # nothing already interacted


def test_factorization_machine_classifies():
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    fm = FactorizationMachine(n_factors=4, n_epochs=40, task="classification",
                              random_state=0).fit(Xtr, ytr)
    assert (fm.predict(Xte) == yte).mean() > 0.8
    p = fm.predict_proba(Xte)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_factorization_machine_regresses():
    from nupyml.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=6, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    # standardise: the FM's SGD is scale-sensitive like any gradient method
    from nupyml.preprocessing import StandardScaler
    sx = StandardScaler().fit(Xtr)
    ys = ytr.std()
    fm = FactorizationMachine(n_factors=4, n_epochs=60, task="regression",
                              random_state=0).fit(sx.transform(Xtr), ytr / ys)
    from nupyml.metrics import r2_score
    pred = fm.predict(sx.transform(Xte)) * ys
    assert r2_score(yte, pred) > 0.7
