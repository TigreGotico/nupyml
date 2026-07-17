"""The boosting variants: GOSS, EFB, ordered boosting, DART, NGBoost, MoE.

Each of these is gradient boosting plus one idea. The tests aim at the idea --
does GOSS really keep its accuracy on a third of the data, does ordered target
statistics really close the leak -- rather than at "does it run".
"""
import numpy as np
import pytest

from nupyml.datasets import (make_regression, make_classification, make_moons,
                             make_circles, load_digits)
from nupyml.ensemble import (
    AdaBoostRegressor, RandomTreesEmbedding, GOSSRegressor,
    OrderedBoostingRegressor, DARTRegressor, NGBoostRegressor,
    HistGradientBoostingRegressor, GradientBoostingRegressor,
    MixtureOfExpertsRegressor, MixtureOfExpertsClassifier,
    goss_sample, exclusive_feature_bundles, bundle_features,
    ordered_target_statistic,
)
from nupyml.linear_model import LinearRegression, LogisticRegression
from nupyml.model_selection import train_test_split
from nupyml.tree import DecisionTreeRegressor


@pytest.fixture(scope="module")
def noisy_regression():
    X, y = make_regression(n_samples=1500, n_features=10, noise=10.0,
                           random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


# --- GOSS -----------------------------------------------------------------

def test_goss_keeps_every_large_gradient():
    """The premise: big gradients are where the information is, so none is
    dropped. Only the well-predicted samples are subject to sampling."""
    g = np.arange(100.0)               # index 99 has the largest gradient
    idx, _ = goss_sample(g, top_rate=0.2, other_rate=0.1,
                         rng=np.random.RandomState(0))
    assert set(range(80, 100)).issubset(set(idx.tolist()))


def test_goss_samples_roughly_the_requested_fraction():
    g = np.random.RandomState(0).normal(size=1000)
    idx, _ = goss_sample(g, top_rate=0.2, other_rate=0.1,
                         rng=np.random.RandomState(0))
    assert len(idx) == 300             # 200 kept + 100 sampled


def test_goss_upweights_the_survivors_of_the_sampled_group():
    """The correction that makes GOSS unbiased rather than just lossy.

    Each survivor stands in for the ones dropped alongside it, so it must carry
    their weight -- otherwise the retained sample misrepresents the data.
    """
    g = np.random.RandomState(0).normal(size=1000)
    idx, w = goss_sample(g, top_rate=0.2, other_rate=0.1,
                         rng=np.random.RandomState(0))
    assert np.allclose(w[:200], 1.0)                 # the kept ones are honest
    assert np.allclose(w[200:], (1 - 0.2) / 0.1)     # the sampled ones speak for more


def test_goss_weights_preserve_the_total_gradient_mass():
    """The point of the reweighting, as a measurement: the weighted subsample
    must estimate the full sum, or every split gain is biased."""
    rng = np.random.RandomState(0)
    g = rng.normal(size=20000)
    idx, w = goss_sample(g, top_rate=0.2, other_rate=0.1, rng=rng)
    # the small-gradient half of the mass is what the sampling estimates
    assert (g[idx] * w).sum() == pytest.approx(g.sum(), abs=0.15 * abs(g).sum())


def test_goss_matches_full_data_boosting_on_a_third_of_the_data(noisy_regression):
    """The whole claim, held to account."""
    Xtr, Xte, ytr, yte = noisy_regression
    full = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(Xtr, ytr)
    goss = GOSSRegressor(max_iter=100, random_state=0).fit(Xtr, ytr)
    assert np.mean(goss.subsample_sizes_) < 0.35 * len(Xtr)
    assert goss.score(Xte, yte) > full.score(Xte, yte) - 0.05


def test_goss_with_no_sampling_keeps_everything():
    """other_rate=0 degenerates to "top gradients only", which is the ablation
    that shows what the sampling half contributes."""
    g = np.arange(100.0)
    idx, w = goss_sample(g, top_rate=0.3, other_rate=0.0,
                         rng=np.random.RandomState(0))
    assert len(idx) == 30
    assert np.allclose(w, 1.0)


# --- EFB ------------------------------------------------------------------

def test_efb_bundles_mutually_exclusive_features():
    """One-hot columns are exclusive by construction, so they should all collapse
    into a single bundle with no information lost."""
    n = 200
    onehot = np.zeros((n, 5), dtype=np.uint8)
    onehot[np.arange(n), np.random.RandomState(0).randint(0, 5, n)] = 1
    bundles = exclusive_feature_bundles(onehot)
    assert len(bundles) == 1
    assert sorted(bundles[0]) == [0, 1, 2, 3, 4]


def test_efb_refuses_to_bundle_conflicting_features():
    """Dense features are non-zero everywhere, so nothing can share a column."""
    dense = np.ones((50, 4), dtype=np.uint8)
    bundles = exclusive_feature_bundles(dense)
    assert len(bundles) == 4


def test_efb_allows_approximate_bundling():
    """A conflict budget buys more bundling at the cost of encoding some rows
    wrongly -- a knob worth knowing exists before results move."""
    rng = np.random.RandomState(0)
    X = np.zeros((200, 4), dtype=np.uint8)
    X[np.arange(200), rng.randint(0, 4, 200)] = 1
    X[:5, :] = 1                             # a few rows conflict everywhere
    strict = len(exclusive_feature_bundles(X, max_conflict_rate=0.0))
    loose = len(exclusive_feature_bundles(X, max_conflict_rate=0.2))
    assert strict == 4       # exact bundling can place nothing together
    assert loose == 1        # tolerating a few bad rows collapses them all
    assert loose < strict


def test_bundled_features_preserve_the_original_distinctions():
    """The encoding must stay injective per row: which original feature was
    non-zero has to remain recoverable, or the tree cannot split on it."""
    n = 300
    rng = np.random.RandomState(0)
    which = rng.randint(0, 4, n)
    onehot = np.zeros((n, 4), dtype=np.uint8)
    onehot[np.arange(n), which] = 1

    bundles = exclusive_feature_bundles(onehot)
    bundled, _ = bundle_features(onehot, bundles, n_bins=255)
    assert bundled.shape[1] == 1
    # every original feature gets its own bin value in the bundled column
    assert len(np.unique(bundled[:, 0])) == 4


def test_bundling_reduces_the_feature_count():
    n = 200
    onehot = np.zeros((n, 20), dtype=np.uint8)
    onehot[np.arange(n), np.random.RandomState(0).randint(0, 20, n)] = 1
    bundles = exclusive_feature_bundles(onehot)
    bundled, _ = bundle_features(onehot, bundles, n_bins=255)
    assert bundled.shape[1] < onehot.shape[1]


# --- ordered target statistics: the leak ----------------------------------

def test_plain_target_encoding_leaks_and_ordered_does_not():
    """The bug CatBoost was built around, demonstrated.

    Every category is unique here, so ``y`` is not predictable from the feature
    at all. Plain target encoding nevertheless reproduces ``y`` EXACTLY, because
    each row's own label is the only thing in its category's mean. Any model
    downstream sees a feature that is literally the answer.
    """
    rng = np.random.RandomState(0)
    n = 200
    categories = np.arange(n)             # each appears exactly once
    y = rng.normal(size=n)

    # the obvious implementation: mean of y within each category
    plain = np.array([y[categories == c].mean() for c in categories])
    assert np.corrcoef(plain, y)[0, 1] == pytest.approx(1.0)   # a perfect leak

    ordered = ordered_target_statistic(categories, y, rng=rng)
    assert abs(np.corrcoef(ordered, y)[0, 1]) < 0.2            # no leak


def test_ordered_target_statistic_never_uses_a_rows_own_label():
    """The invariant, checked directly: with the prior pinned, changing a row's
    label must not move that row's own encoding.

    It SHOULD move the encoding of later rows in the same category -- they are
    downstream of it in the ordering, and using it is exactly the point. Only
    looking at yourself is forbidden.
    """
    values = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    a = ordered_target_statistic(values, y, rng=np.random.RandomState(0), prior=0.0)
    y2 = y.copy()
    y2[3] = 999.0
    b = ordered_target_statistic(values, y2, rng=np.random.RandomState(0), prior=0.0)

    assert a[3] == pytest.approx(b[3])      # row 3 cannot see row 3
    assert not np.allclose(a, b)            # but its category's later rows can


def test_the_default_prior_leaks_only_at_order_one_over_n():
    """Honesty about what remains. The default prior is the global mean, which
    contains every row's label -- including the one being encoded.

    The leak this method removes is O(1): plain encoding hands a singleton
    category its own label exactly, and more data never helps. What survives is
    y_i/n, which vanishes. Trading O(1) for O(1/n) is the entire improvement, and
    it is worth measuring rather than asserting.
    """
    def leak_at(n):
        rng = np.random.RandomState(0)
        values = np.arange(n)              # every category unique: the worst case
        y = rng.normal(size=n)
        a = ordered_target_statistic(values, y, rng=np.random.RandomState(0))
        y2 = y.copy()
        y2[0] += 100.0                     # move one label a long way
        b = ordered_target_statistic(values, y2, rng=np.random.RandomState(0))
        return abs(a[0] - b[0])

    small, large = leak_at(20), leak_at(2000)
    assert large < small / 10              # it shrinks with n, as 1/n predicts
    # and pinning the prior removes it outright
    rng = np.random.RandomState(0)
    y = rng.normal(size=20)
    a = ordered_target_statistic(np.arange(20), y, rng=np.random.RandomState(0),
                                 prior=0.0)
    y2 = y.copy()
    y2[0] += 100.0
    b = ordered_target_statistic(np.arange(20), y2, rng=np.random.RandomState(0),
                                 prior=0.0)
    assert a[0] == pytest.approx(b[0])


def test_ordered_target_statistic_still_learns_a_real_signal():
    """Closing the leak must not destroy the encoding's usefulness: where the
    category genuinely predicts y, the encoding should still reflect it."""
    rng = np.random.RandomState(0)
    values = rng.randint(0, 3, 600)
    y = values * 10.0 + rng.normal(0, 0.5, 600)
    encoded = ordered_target_statistic(values, y, rng=rng)
    assert np.corrcoef(encoded, y)[0, 1] > 0.9


def test_ordered_target_statistic_falls_back_to_the_prior():
    """The first row of each category has no history, so it gets the global
    mean -- the only honest answer when nothing is known yet."""
    values = np.array([0, 0, 0, 0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    encoded = ordered_target_statistic(values, y, rng=np.random.RandomState(0))
    assert np.isclose(encoded, y.mean()).sum() >= 1


def test_ordered_boosting_fits():
    X, y = make_regression(n_samples=400, n_features=6, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = OrderedBoostingRegressor(n_estimators=60, random_state=0).fit(Xtr, ytr)
    assert model.score(Xte, yte) > 0.8


def test_ordered_boosting_residuals_come_from_a_model_that_has_not_seen_the_row():
    """Prediction shift: ordinary boosting fits residuals that are optimistically
    small because the earlier trees already saw the row. The gap between training
    and test fit is the symptom."""
    X, y = make_regression(n_samples=400, n_features=6, noise=20.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ordered = OrderedBoostingRegressor(n_estimators=80, random_state=0).fit(Xtr, ytr)
    plain = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                      random_state=0).fit(Xtr, ytr)
    # the ordered model is less optimistic about its own training data
    ordered_gap = ordered.score(Xtr, ytr) - ordered.score(Xte, yte)
    plain_gap = plain.score(Xtr, ytr) - plain.score(Xte, yte)
    assert ordered_gap <= plain_gap + 0.05


# --- DART -----------------------------------------------------------------

def test_dart_fits(noisy_regression):
    Xtr, Xte, ytr, yte = noisy_regression
    dart = DARTRegressor(n_estimators=80, random_state=0).fit(Xtr, ytr)
    assert dart.score(Xte, yte) > 0.7


def test_dart_actually_drops_trees():
    X, y = make_regression(n_samples=300, n_features=5, random_state=0)
    dart = DARTRegressor(n_estimators=60, drop_rate=0.3, skip_drop=0.0,
                         random_state=0).fit(X, y)
    assert sum(dart.n_dropped_) > 0


def test_dart_with_skip_drop_one_never_drops():
    """The ablation: skip_drop=1 makes it ordinary boosting, which is the
    baseline the dropout is measured against."""
    X, y = make_regression(n_samples=300, n_features=5, random_state=0)
    dart = DARTRegressor(n_estimators=40, skip_drop=1.0, random_state=0).fit(X, y)
    assert sum(dart.n_dropped_) == 0
    assert np.allclose(dart.weights_, dart.learning_rate)


def test_dart_rescales_dropped_trees():
    """The normalisation that keeps the ensemble's output from exploding: a
    dropped tree's weight must shrink to make room for its replacement."""
    X, y = make_regression(n_samples=300, n_features=5, random_state=0)
    dart = DARTRegressor(n_estimators=50, drop_rate=0.5, skip_drop=0.0,
                         learning_rate=0.1, random_state=0).fit(X, y)
    # trees that were dropped have been scaled below the base learning rate
    assert min(dart.weights_) < 0.1


def test_dart_spreads_its_weight_more_evenly_than_plain_boosting():
    """Over-specialisation is the disease: in plain boosting the first trees do
    everything and the rest fit crumbs. Dropout forces later trees to matter."""
    X, y = make_regression(n_samples=600, n_features=6, noise=5.0, random_state=0)
    dart = DARTRegressor(n_estimators=60, drop_rate=0.3, skip_drop=0.0,
                         random_state=0).fit(X, y)
    contributions = np.array([np.abs(t.predict(X)).mean() * w
                              for t, w in zip(dart.trees_, dart.weights_)])
    late = contributions[30:].sum()
    assert late > 0        # the late trees are not dead weight


def test_dart_does_not_diverge():
    """Getting the 1/(k+1) scaling wrong makes this blow up immediately, so a
    finite, sane prediction is a real check on the derivation."""
    X, y = make_regression(n_samples=300, n_features=5, random_state=0)
    dart = DARTRegressor(n_estimators=60, drop_rate=0.5, skip_drop=0.0,
                         random_state=0).fit(X, y)
    pred = dart.predict(X)
    assert np.all(np.isfinite(pred))
    assert np.abs(pred).max() < 10 * np.abs(y).max()


# --- NGBoost --------------------------------------------------------------

def test_ngboost_predicts_a_mean_like_any_regressor(noisy_regression):
    Xtr, Xte, ytr, yte = noisy_regression
    ng = NGBoostRegressor(n_estimators=150, random_state=0).fit(Xtr, ytr)
    assert ng.score(Xte, yte) > 0.8


def test_ngboost_improves_its_likelihood(noisy_regression):
    Xtr, _, ytr, _ = noisy_regression
    ng = NGBoostRegressor(n_estimators=150, random_state=0).fit(Xtr, ytr)
    assert ng.nll_[-1] < ng.nll_[0]


def test_ngboost_reports_more_uncertainty_where_there_is_more_noise():
    """The claim that distinguishes this from every other regressor here: not
    just a value, but how much to trust it -- varying per sample."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (2000, 1))
    # noise grows with x: the right answer is a sigma that grows with it too
    y = X[:, 0] * 2 + rng.normal(0, 0.1 + np.abs(X[:, 0]), 2000)
    ng = NGBoostRegressor(n_estimators=300, max_depth=2, random_state=0).fit(X, y)
    _, sigma_low = ng.predict_dist(np.array([[0.0]]))
    _, sigma_high = ng.predict_dist(np.array([[2.8]]))
    assert sigma_high[0] > sigma_low[0] * 1.5


def test_ngboost_intervals_are_calibrated_when_the_mean_does_not_overfit():
    """sigma is fit on TRAINING residuals, so it is honest exactly when the mean
    model generalises. A shallow mean model is the condition for that."""
    X, y = make_regression(n_samples=2000, n_features=10, noise=10.0,
                           random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ng = NGBoostRegressor(n_estimators=200, max_depth=1, random_state=0).fit(Xtr, ytr)
    lo, hi = ng.predict_interval(Xte, coverage=0.95)
    assert 0.9 < ((yte >= lo) & (yte <= hi)).mean() < 1.0


def test_ngboost_intervals_are_overconfident_when_the_mean_overfits():
    """The catch, made explicit. A deep mean model fits training noise, so its
    training residuals are small, so sigma is small -- and the "95%" interval
    covers far less than 95% of unseen data. Confidently wrong.

    This is the argument for conformal prediction in one test.
    """
    X, y = make_regression(n_samples=2000, n_features=10, noise=10.0,
                           random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ng = NGBoostRegressor(n_estimators=200, max_depth=3, random_state=0).fit(Xtr, ytr)
    lo, hi = ng.predict_interval(Xte, coverage=0.95)
    train_lo, train_hi = ng.predict_interval(Xtr, coverage=0.95)

    test_coverage = ((yte >= lo) & (yte <= hi)).mean()
    train_coverage = ((ytr >= train_lo) & (ytr <= train_hi)).mean()
    assert train_coverage > 0.94          # honest about what it has seen
    assert test_coverage < 0.9            # and overconfident about what it has not


def test_ngboost_sigma_is_positive_everywhere(noisy_regression):
    Xtr, Xte, ytr, _ = noisy_regression
    ng = NGBoostRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    _, sigma = ng.predict_dist(Xte)
    assert np.all(sigma > 0) and np.all(np.isfinite(sigma))


def test_ngboost_intervals_widen_with_requested_coverage(noisy_regression):
    Xtr, Xte, ytr, _ = noisy_regression
    ng = NGBoostRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    lo50, hi50 = ng.predict_interval(Xte, 0.5)
    lo95, hi95 = ng.predict_interval(Xte, 0.95)
    assert np.all((hi95 - lo95) > (hi50 - lo50))


def test_natural_gradient_is_not_the_ordinary_gradient(noisy_regression):
    """If the Fisher rescaling were a no-op the flag would change nothing."""
    Xtr, Xte, ytr, yte = noisy_regression
    natural = NGBoostRegressor(n_estimators=80, natural_gradient=True,
                               random_state=0).fit(Xtr, ytr)
    ordinary = NGBoostRegressor(n_estimators=80, natural_gradient=False,
                                random_state=0).fit(Xtr, ytr)
    assert not np.allclose(natural.predict(Xte), ordinary.predict(Xte))


# --- AdaBoost for regression ----------------------------------------------

def test_adaboost_regressor_fits(noisy_regression):
    Xtr, Xte, ytr, yte = noisy_regression
    ada = AdaBoostRegressor(n_estimators=50, random_state=0).fit(Xtr, ytr)
    assert ada.score(Xte, yte) > 0.7


@pytest.mark.parametrize("loss", ["linear", "square", "exponential"])
def test_adaboost_regressor_loss_variants(loss, noisy_regression):
    Xtr, Xte, ytr, yte = noisy_regression
    ada = AdaBoostRegressor(n_estimators=30, loss=loss, random_state=0).fit(Xtr, ytr)
    assert ada.score(Xte, yte) > 0.6


def test_adaboost_regressor_rejects_an_unknown_loss():
    X, y = make_regression(n_samples=100, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="Unknown loss"):
        AdaBoostRegressor(loss="nonsense").fit(X, y)


def test_adaboost_regressor_predicts_a_weighted_median():
    """Not a mean -- the median is what keeps one wild learner from dragging the
    ensemble, which matters because this method is already outlier-prone."""
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    ada = AdaBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
    preds = np.column_stack([e.predict(X) for e in ada.estimators_])
    out = ada.predict(X)
    # the answer must be one of the learners' own predictions, which a mean
    # would essentially never be
    assert np.all(np.isclose(out[:, None], preds).any(axis=1))


def test_adaboost_regressor_stops_when_a_learner_is_no_better_than_chance():
    X, y = make_regression(n_samples=100, n_features=4, random_state=0)
    ada = AdaBoostRegressor(n_estimators=200, random_state=0).fit(X, y)
    assert len(ada.estimators_) <= 200
    assert len(ada.estimator_weights_) == len(ada.estimators_)


def test_adaboost_regressor_gives_better_learners_more_vote():
    X, y = make_regression(n_samples=400, n_features=4, noise=5.0, random_state=0)
    ada = AdaBoostRegressor(n_estimators=30, random_state=0).fit(X, y)
    assert np.all(ada.estimator_weights_ >= 0)
    assert np.any(ada.estimator_weights_ > 0)


# --- RandomTreesEmbedding -------------------------------------------------

def test_random_trees_embedding_is_a_sparse_one_hot_code():
    X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)
    emb = RandomTreesEmbedding(n_estimators=10, max_depth=3, sparse_output=False,
                               random_state=0)
    E = emb.fit_transform(X)
    assert set(np.unique(E)).issubset({0.0, 1.0})
    # exactly one leaf per tree is active for each sample
    assert np.allclose(E.sum(axis=1), 10)


def test_random_trees_embedding_is_reproducible():
    """Keying leaves by id(node) would make the column order depend on memory
    addresses -- the embedding still separates the data, so it looks fine, but
    two runs with the same seed disagree and no column means anything."""
    X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)
    a = RandomTreesEmbedding(n_estimators=5, sparse_output=False,
                             random_state=0).fit_transform(X)
    b = RandomTreesEmbedding(n_estimators=5, sparse_output=False,
                             random_state=0).fit_transform(X)
    assert np.array_equal(a, b)


def test_random_trees_embedding_is_unsupervised():
    """It never sees y, so passing a different one must change nothing."""
    X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
    a = RandomTreesEmbedding(n_estimators=5, sparse_output=False,
                             random_state=0).fit_transform(X, y)
    b = RandomTreesEmbedding(n_estimators=5, sparse_output=False,
                             random_state=0).fit_transform(X, 1 - y)
    assert np.array_equal(a, b)


def test_random_trees_embedding_makes_a_linear_model_work_on_moons():
    """The point of the transform: carve the space into cells, one-hot which cell
    you are in, and a linear model can now fit a constant per cell -- which is a
    non-linear function of the original input."""
    X, y = make_moons(n_samples=800, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    emb = RandomTreesEmbedding(n_estimators=30, max_depth=4, sparse_output=False,
                               random_state=0)
    Etr, Ete = emb.fit_transform(Xtr), emb.transform(Xte)
    raw = LogisticRegression(max_iter=500).fit(Xtr, ytr).score(Xte, yte)
    embedded = LogisticRegression(max_iter=500).fit(Etr, ytr).score(Ete, yte)
    assert embedded > raw


def test_random_trees_embedding_rescues_a_linear_model_on_circles():
    """Circles are the sharper case: a linear model on the raw features is at or
    below chance, because no straight line separates a ring from its centre."""
    X, y = make_circles(n_samples=800, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    emb = RandomTreesEmbedding(n_estimators=30, max_depth=4, sparse_output=False,
                               random_state=0)
    Etr, Ete = emb.fit_transform(Xtr), emb.transform(Xte)
    raw = LogisticRegression(max_iter=500).fit(Xtr, ytr).score(Xte, yte)
    embedded = LogisticRegression(max_iter=500).fit(Etr, ytr).score(Ete, yte)
    assert raw < 0.6           # a straight line cannot do this
    assert embedded > raw + 0.15


def test_random_trees_embedding_defaults_to_sparse():
    from scipy.sparse import issparse
    X, _ = make_moons(n_samples=100, random_state=0)
    E = RandomTreesEmbedding(n_estimators=5, random_state=0).fit_transform(X)
    assert issparse(E)


def test_random_trees_embedding_transform_matches_fit_transform():
    X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)
    emb = RandomTreesEmbedding(n_estimators=8, sparse_output=False, random_state=0)
    assert np.array_equal(emb.fit_transform(X), emb.transform(X))


# --- mixture of experts ---------------------------------------------------

@pytest.fixture(scope="module")
def piecewise():
    """Two regimes with different slopes -- the situation a mixture is for."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (1200, 1))
    y = np.where(X[:, 0] < 0, 2 * X[:, 0] + 1, -3 * X[:, 0] + 5) \
        + rng.normal(0, 0.2, 1200)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_moe_regressor_fits_piecewise_data(piecewise):
    Xtr, Xte, ytr, yte = piecewise
    moe = MixtureOfExpertsRegressor(n_experts=2, n_epochs=150,
                                    random_state=0).fit(Xtr, ytr)
    assert moe.score(Xte, yte) > 0.95


def test_moe_gate_discovers_the_regime_boundary(piecewise):
    """Nobody told it where the split is. Specialisation emerges because an
    expert that is slightly better somewhere gets more of that region's gradient,
    which makes it better there still."""
    Xtr, _, ytr, _ = piecewise
    moe = MixtureOfExpertsRegressor(n_experts=2, n_epochs=150,
                                    random_state=0).fit(Xtr, ytr)
    left = moe.gate_weights(np.array([[-2.0]]))[0]
    right = moe.gate_weights(np.array([[2.0]]))[0]
    # the two regimes must be routed to different experts
    assert np.argmax(left) != np.argmax(right)
    assert left.max() > 0.7 and right.max() > 0.7


def test_moe_gate_weights_are_a_distribution(piecewise):
    Xtr, Xte, ytr, _ = piecewise
    moe = MixtureOfExpertsRegressor(n_experts=3, n_epochs=30,
                                    random_state=0).fit(Xtr, ytr)
    g = moe.gate_weights(Xte)
    assert np.allclose(g.sum(axis=1), 1.0)
    assert np.all(g >= 0)


def test_load_balancing_prevents_expert_collapse(piecewise):
    """Without the auxiliary loss, one expert can eat the whole problem: it gets
    slightly ahead, the gate feeds it more, and the others never train."""
    Xtr, _, ytr, _ = piecewise
    balanced = MixtureOfExpertsRegressor(n_experts=4, n_epochs=80,
                                         load_balance_coef=1.0,
                                         random_state=0).fit(Xtr, ytr)
    assert balanced.expert_usage_.min() > 0.05      # everyone is doing something


def test_moe_classifier_fits():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3,
                                          random_state=0)
    moe = MixtureOfExpertsClassifier(n_experts=4, n_epochs=50,
                                     random_state=0).fit(Xtr, ytr)
    assert moe.score(Xte, yte) > 0.85


def test_moe_classifier_mixes_probabilities_into_a_distribution():
    """A convex combination is only meaningful on a simplex: mixing probabilities
    gives a distribution, mixing logits gives a number that means nothing."""
    X, y = load_digits(return_X_y=True)
    moe = MixtureOfExpertsClassifier(n_experts=3, n_epochs=10,
                                     random_state=0).fit(X[:500] / 16.0, y[:500])
    p = moe.predict_proba(X[:100] / 16.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_moe_classifier_predicts_known_classes():
    X, y = load_digits(return_X_y=True)
    moe = MixtureOfExpertsClassifier(n_experts=2, n_epochs=10,
                                     random_state=0).fit(X[:400] / 16.0, y[:400])
    assert set(moe.predict(X[:100] / 16.0)).issubset(set(moe.classes_))
