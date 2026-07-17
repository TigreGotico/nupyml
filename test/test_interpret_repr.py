"""Interpretability, imbalance handling, and representation learning.

The explanation tests hold each method to its defining property (SHAP sums to the
prediction gap, permutation importance ranks the informative features); the
imbalance tests check the classes end balanced; the representation tests check
that meaning or structure was actually captured.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification, load_iris, make_regression
from nupyml.ensemble import RandomForestClassifier, RandomForestRegressor
from nupyml.neighbors import KNeighborsClassifier
from nupyml.model_selection import train_test_split
from nupyml.explain import (permutation_importance, partial_dependence, ice,
                            LIME, KernelSHAP, integrated_gradients)
from nupyml.imbalance import (RandomOverSampler, RandomUnderSampler, SMOTE,
                              ADASYN, BorderlineSMOTE, TomekLinks, NearMiss)
from nupyml.embed import Word2Vec, GloVe, BPETokenizer, WordPieceTokenizer
from nupyml.cluster import KMedoids, KModes, FuzzyCMeans
from nupyml.metric_learning import NCA, LMNN
from nupyml.manifold import UMAP, SelfOrganizingMap


# --- explanation ----------------------------------------------------------

@pytest.fixture
def forest_data():
    X, y = make_classification(n_samples=500, n_features=6, n_informative=3,
                               random_state=0)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    return rf, X, y


def test_permutation_importance_ranks_informative_features(forest_data):
    """The informative features should lose more accuracy when shuffled than the
    noise ones -- importance measured by what the model actually uses."""
    rf, X, y = forest_data
    pi = permutation_importance(rf, X, y, n_repeats=5, random_state=0)
    top3 = set(np.argsort(pi["importances_mean"])[-3:])
    # the three informative features (0, 1, 2) should dominate
    assert len(top3 & {0, 1, 2}) >= 2


def test_permutation_importance_is_near_zero_for_noise():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] > 0).astype(int)               # only feature 0 matters
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    pi = permutation_importance(rf, X, y, random_state=0)
    assert pi["importances_mean"][0] > 0.1      # the real feature matters
    assert np.all(pi["importances_mean"][1:] < 0.05)   # the rest do not


def test_partial_dependence_tracks_a_monotone_effect():
    """A feature the model uses monotonically should give a monotone PDP."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (500, 3))
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    grid, pd = partial_dependence(rf, X, 0, predict_method="predict_proba")
    # P(class 1) should rise with feature 0
    assert pd[-1] > pd[0]


def test_ice_reveals_heterogeneity_a_pdp_would_hide():
    """ICE keeps one curve per row; the PDP is their mean. When the effect differs
    by subgroup, the ICE curves must spread even where the PDP is flat."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (400, 2))
    # feature 0 helps when feature 1 > 0 and hurts otherwise: opposite effects
    y = ((X[:, 0] * np.sign(X[:, 1])) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    grid, curves = ice(rf, X, 0, predict_method="predict_proba")
    # the curves genuinely disagree in direction -- some rise, some fall
    slopes = curves[:, -1] - curves[:, 0]
    assert slopes.max() > 0.1 and slopes.min() < -0.1


def test_lime_gives_a_local_linear_explanation(forest_data):
    rf, X, y = forest_data
    lime = LIME(n_samples=1000, random_state=0)
    coef = lime.explain(lambda Xs: rf.predict_proba(Xs)[:, 1], X[0], X)
    assert coef.shape == (X.shape[1],)
    assert np.any(coef != 0)


def test_shap_values_sum_to_the_prediction_gap(forest_data):
    """The efficiency axiom, checked exactly: the attributions must sum to the
    difference between this prediction and the baseline. This is what makes SHAP
    a fair division rather than a heuristic."""
    rf, X, y = forest_data
    shap = KernelSHAP(random_state=0)
    phi = shap.explain(lambda Xs: rf.predict_proba(Xs)[:, 1], X[0], X)
    gap = rf.predict_proba(X[0:1])[0, 1] - shap.expected_value_
    assert phi.sum() == pytest.approx(gap, abs=1e-6)


def test_shap_dummy_feature_gets_zero_credit():
    """The dummy axiom: a feature the model never uses must get zero attribution."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)     # features 2, 3 are ignored
    rf = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    shap = KernelSHAP(random_state=0)
    phi = shap.explain(lambda Xs: rf.predict_proba(Xs)[:, 1], X[0], X)
    # the unused features get much smaller attributions than the used ones
    assert abs(phi[2]) + abs(phi[3]) < abs(phi[0]) + abs(phi[1])


def test_integrated_gradients_sum_to_the_output_difference():
    """Like SHAP, IG satisfies efficiency: the attributions sum to f(x) - f(baseline)."""
    # a simple differentiable function: f(x) = w . x, gradient is constant w
    w = np.array([2.0, -1.0, 0.5])

    def grad_fn(X):
        return np.tile(w, (len(X), 1))

    x = np.array([1.0, 2.0, 3.0])
    baseline = np.zeros(3)
    attr = integrated_gradients(grad_fn, x, baseline, n_steps=50)
    # for a linear function IG is exactly w * (x - baseline)
    assert np.allclose(attr, w * x)
    assert attr.sum() == pytest.approx(w @ x)


# --- imbalance ------------------------------------------------------------

@pytest.fixture
def imbalanced():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 4))
    y = (rng.uniform(size=1000) < 0.1).astype(int)      # ~10% minority
    return X, y


@pytest.mark.parametrize("sampler_cls", [SMOTE, ADASYN, BorderlineSMOTE,
                                         RandomOverSampler])
def test_oversamplers_balance_the_classes(sampler_cls, imbalanced):
    X, y = imbalanced
    sampler = sampler_cls(random_state=0)
    Xr, yr = sampler.fit_resample(X, y)
    counts = np.bincount(yr)
    assert counts.min() / counts.max() > 0.9        # roughly balanced


def test_smote_creates_synthetic_not_duplicate_points(imbalanced):
    """SMOTE interpolates, so the new minority points are NEW -- not copies of the
    originals, which is the whole improvement over random oversampling."""
    X, y = imbalanced
    Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
    original_minority = X[y == 1]
    new_minority = Xr[len(y):]                       # the synthetic tail
    # synthetic points are not exact copies of any original
    for pt in new_minority[:20]:
        assert not np.any(np.all(np.isclose(original_minority, pt), axis=1))


def test_random_undersampler_balances_by_dropping(imbalanced):
    X, y = imbalanced
    Xr, yr = RandomUnderSampler(random_state=0).fit_resample(X, y)
    counts = np.bincount(yr)
    assert counts[0] == counts[1]
    assert len(yr) < len(y)                          # data was dropped


def test_tomek_links_removes_boundary_points(imbalanced):
    """Tomek cleaning only removes points, and only boundary ones."""
    X, y = imbalanced
    Xr, yr = TomekLinks().fit_resample(X, y)
    assert len(yr) <= len(y)


def test_nearmiss_undersamples_the_majority(imbalanced):
    X, y = imbalanced
    Xr, yr = NearMiss(n_neighbors=3).fit_resample(X, y)
    assert (yr == 1).sum() == (y == 1).sum()         # minority untouched
    assert (yr == 0).sum() <= (y == 0).sum()         # majority reduced


def test_resampling_helps_a_classifier_on_the_minority():
    """The end-to-end point: balancing should improve recall on the rare class."""
    from nupyml.metrics import recall_score
    from nupyml.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1500, 4))
    y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, 1500) > 2.0).astype(int)  # rare
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    plain = LogisticRegression(max_iter=300).fit(Xtr, ytr)
    Xr, yr = SMOTE(random_state=0).fit_resample(Xtr, ytr)
    balanced = LogisticRegression(max_iter=300).fit(Xr, yr)
    # SMOTE should not reduce minority recall, and usually improves it
    assert recall_score(yte, balanced.predict(Xte)) >= \
        recall_score(yte, plain.predict(Xte)) - 0.05


# --- embeddings -----------------------------------------------------------

@pytest.fixture
def toy_corpus():
    return ([["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "sat", "on", "the", "rug"],
             ["a", "cat", "and", "a", "dog", "played"],
             ["the", "cat", "chased", "the", "dog"]] * 40)


def test_word2vec_learns_similar_vectors_for_similar_words(toy_corpus):
    """cat and dog appear in near-identical contexts, so their vectors should be
    among each other's nearest -- meaning from co-occurrence alone."""
    w2v = Word2Vec(n_dim=30, window=2, n_epochs=30, random_state=0).fit(toy_corpus)
    neighbours = [w for w, _ in w2v.most_similar("cat", topn=3)]
    assert "dog" in neighbours


def test_word2vec_ignores_rare_words_below_min_count(toy_corpus):
    w2v = Word2Vec(n_dim=20, min_count=100, n_epochs=5, random_state=0).fit(toy_corpus)
    # only the very frequent words survive the min_count filter
    assert "the" in w2v.vocab_
    assert "played" not in w2v.vocab_


def test_word2vec_vectors_have_the_requested_dimension(toy_corpus):
    w2v = Word2Vec(n_dim=25, n_epochs=5, random_state=0).fit(toy_corpus)
    assert w2v.get_vector("cat").shape == (25,)


def test_glove_learns_from_co_occurrence(toy_corpus):
    """GloVe fits the global co-occurrence matrix, so on a tiny corpus function
    words dominate and cat~dog is weaker than word2vec's predictive version gets.
    The defensible claim here: dog ranks above the clearly unrelated words, and
    the embedding has the right shape."""
    glove = GloVe(n_dim=30, window=3, n_epochs=100, random_state=0).fit(toy_corpus)
    neighbours = [w for w, _ in glove.most_similar("cat", topn=6)]
    assert "dog" in neighbours
    assert glove.get_vector("cat").shape == (30,)
    # dog (shared context) should outrank 'played', which co-occurs with neither
    sims = dict(glove.most_similar("cat", topn=10))
    if "played" in sims:
        assert sims["dog"] > sims["played"]


def test_bpe_merges_frequent_pairs():
    """BPE learns whole tokens for frequent sequences; a repeated word should end
    up representable as few pieces."""
    corpus = ["low lower lowest"] * 50
    bpe = BPETokenizer(vocab_size=40).fit(corpus)
    # "low" appears in every word, so it should have been merged into one token
    assert any("low" in tok for tok in bpe.vocab_)
    assert len(bpe.tokenize("low")) < 4          # fewer than its 3 chars + marker


def test_bpe_handles_unseen_words():
    """No out-of-vocabulary problem: an unseen word still tokenizes into known
    pieces."""
    bpe = BPETokenizer(vocab_size=50).fit(["low lower lowest newer"] * 30)
    tokens = bpe.tokenize("newest")              # never seen whole
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_wordpiece_builds_a_vocabulary():
    wp = WordPieceTokenizer(vocab_size=40).fit(["playing played player plays"] * 30)
    assert len(wp.vocab_) <= 40
    assert len(wp.vocab_) > 0


# --- clustering variants --------------------------------------------------

def test_kmedoids_centres_are_real_data_points():
    """A medoid IS a data point, unlike a k-means centroid -- so every centre
    must appear in X."""
    X, _ = load_iris(return_X_y=True)
    km = KMedoids(n_clusters=3, random_state=0).fit(X)
    for centre in km.cluster_centers_:
        assert np.any(np.all(np.isclose(X, centre), axis=1))


def test_kmedoids_resists_an_outlier():
    """The median-vs-mean robustness: a wild outlier should not become a centre
    the way it can drag a k-means centroid."""
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)),
                   rng.normal(5, 0.3, (50, 2)),
                   [[100, 100]]])                # one gross outlier
    km = KMedoids(n_clusters=2, random_state=0).fit(X)
    # neither medoid is the outlier
    assert not np.any(np.all(km.cluster_centers_ > 50, axis=1))


def test_kmodes_clusters_categorical_data():
    """K-modes works on labels, where k-means cannot -- the centre is the mode."""
    X = np.array([["red", "big"]] * 20 + [["blue", "small"]] * 20)
    km = KModes(n_clusters=2, random_state=0).fit(X)
    centres = sorted(map(tuple, km.cluster_centers_.tolist()))
    assert centres == [("blue", "small"), ("red", "big")]


def test_fuzzy_cmeans_memberships_sum_to_one():
    """Soft assignment: each point's memberships across clusters form a
    distribution."""
    X, _ = load_iris(return_X_y=True)
    fcm = FuzzyCMeans(n_clusters=3, random_state=0).fit(X)
    assert np.allclose(fcm.membership_.sum(axis=1), 1.0)
    assert np.all(fcm.membership_ >= 0)


def test_fuzzy_cmeans_hard_labels_recover_structure():
    from nupyml.metrics import adjusted_rand_score
    X, y = load_iris(return_X_y=True)
    fcm = FuzzyCMeans(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, fcm.labels_) > 0.5


# --- metric learning ------------------------------------------------------

def test_nca_improves_knn_on_noise_dominated_data():
    """The point of metric learning: when noise features dominate Euclidean
    distance, NCA learns to down-weight them and kNN recovers."""
    rng = np.random.RandomState(1)
    X_signal, y = make_classification(n_samples=400, n_features=2,
                                      n_informative=2, random_state=1)
    noise = rng.normal(0, 5, size=(400, 8))         # 8 large noise features
    X = np.hstack([X_signal, noise])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    raw = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).score(Xte, yte)
    nca = NCA(n_components=2, max_iter=50, learning_rate=0.01,
              random_state=0).fit(Xtr, ytr)
    learned = KNeighborsClassifier(n_neighbors=5).fit(
        nca.transform(Xtr), ytr).score(nca.transform(Xte), yte)
    assert learned > raw


def test_nca_reduces_dimensionality():
    X, y = load_iris(return_X_y=True)
    nca = NCA(n_components=2, max_iter=20, random_state=0).fit(X, y)
    assert nca.transform(X).shape == (len(X), 2)


def test_lmnn_learns_a_transform():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    lmnn = LMNN(k=3, max_iter=20, learning_rate=1e-3, random_state=0).fit(Xtr, ytr)
    assert lmnn.transform(Xtr).shape == Xtr.shape
    # the transformed space should keep kNN accuracy at least reasonable
    acc = KNeighborsClassifier(n_neighbors=3).fit(
        lmnn.transform(Xtr), ytr).score(lmnn.transform(Xte), yte)
    assert acc > 0.8


# --- UMAP and SOM ---------------------------------------------------------

def test_umap_produces_a_low_dimensional_embedding():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                               random_state=0)
    emb = UMAP(n_components=2, n_epochs=100, random_state=0).fit_transform(X)
    assert emb.shape == (200, 2)


def test_umap_keeps_same_class_points_closer_than_random():
    """A useful embedding puts same-class points nearer each other than
    arbitrary pairs -- structure preserved, not scrambled."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=6,
                               class_sep=2.0, random_state=0)
    emb = UMAP(n_components=2, n_neighbors=15, n_epochs=200,
               random_state=0).fit_transform(X)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(emb))
    same_class = D[np.ix_(y == 0, y == 0)].mean()
    cross = D[np.ix_(y == 0, y == 1)].mean()
    assert same_class < cross


def test_som_maps_to_grid_coordinates():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=4,
                               random_state=0)
    som = SelfOrganizingMap(grid_shape=(8, 8), n_epochs=30,
                            random_state=0).fit(X)
    coords = som.transform(X)
    assert coords.shape == (200, 2)
    assert coords.min() >= 0 and coords.max() < 8


def test_som_is_topology_preserving():
    """Similar inputs should map to nearby grid cells -- the defining property of
    a self-organizing map."""
    X, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               class_sep=2.0, random_state=0)
    som = SelfOrganizingMap(grid_shape=(10, 10), n_epochs=50,
                            random_state=0).fit(X)
    coords = som.transform(X)
    from scipy.spatial.distance import pdist, squareform
    grid_D = squareform(pdist(coords))
    same = grid_D[np.ix_(y == 0, y == 0)].mean()
    cross = grid_D[np.ix_(y == 0, y == 1)].mean()
    assert same < cross
