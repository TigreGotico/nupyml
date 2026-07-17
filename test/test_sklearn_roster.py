"""The remaining sklearn roster: covariance, cross-decomposition, transformers,
dummies, and the smaller estimators.

Covariance shrinkage is held to its defining property (an invertible estimate
where the sample covariance is singular); PLS/CCA to recovering planted shared
structure; the transformers to the distribution change they promise; the dummies
to being the baselines real models must beat.
"""
import numpy as np
import pytest
from scipy.stats import skew

from nupyml.datasets import load_iris, make_regression, make_classification
from nupyml.model_selection import train_test_split
from nupyml.metrics import adjusted_rand_score
from nupyml.covariance import (EmpiricalCovariance, LedoitWolf, OAS,
                               GraphicalLasso, MinCovDet, ShrunkCovariance)
from nupyml.cross_decomposition import PLSRegression, CCA, PLSCanonical
from nupyml.dummy import DummyClassifier, DummyRegressor
from nupyml.preprocessing import (PowerTransformer, QuantileTransformer,
                                  TargetEncoder, FunctionTransformer)
from nupyml.decomposition import FactorAnalysis, IncrementalPCA, PCA
from nupyml.neighbors import NearestCentroid, RadiusNeighborsClassifier
from nupyml.feature_extraction import (FeatureHasher, DictVectorizer,
                                       HashingVectorizer)
from nupyml.cluster import BisectingKMeans, FeatureAgglomeration


# --- covariance -----------------------------------------------------------

def test_empirical_covariance_matches_numpy():
    X = np.random.RandomState(0).normal(size=(200, 4))
    cov = EmpiricalCovariance().fit(X).covariance_
    assert np.allclose(cov, np.cov(X.T, bias=True), atol=1e-10)


def test_ledoit_wolf_is_invertible_when_sample_covariance_is_not():
    """The whole point of shrinkage: p > n makes the sample covariance singular,
    but the shrunk estimate is always invertible."""
    X = np.random.RandomState(0).normal(size=(30, 50))   # more features than samples
    lw = LedoitWolf().fit(X)
    # the raw sample covariance is singular here
    assert np.linalg.matrix_rank(np.cov(X.T, bias=True)) < 50
    # the shrunk one is full rank and invertible
    inv = np.linalg.inv(lw.covariance_)
    assert np.all(np.isfinite(inv))
    assert 0 < lw.shrinkage_ <= 1


def test_ledoit_wolf_shrinks_more_when_the_estimate_is_less_reliable():
    """The optimal intensity rises when the sample covariance is noisier -- fewer
    samples of genuinely CORRELATED features means a less reliable estimate and
    more shrinkage. (For white noise the identity target is already correct, so
    this only shows up when there is real structure to under-sample.)"""
    rng = np.random.RandomState(0)
    A = rng.normal(size=(20, 20))                # a real covariance structure
    cov_sqrt = A @ A.T
    L = np.linalg.cholesky(cov_sqrt + 1e-6 * np.eye(20))
    few = LedoitWolf().fit(rng.normal(size=(25, 20)) @ L.T).shrinkage_
    many = LedoitWolf().fit(rng.normal(size=(3000, 20)) @ L.T).shrinkage_
    assert few > many


def test_oas_produces_a_valid_shrinkage():
    X = np.random.RandomState(0).normal(size=(40, 30))
    oas = OAS().fit(X)
    assert 0 <= oas.shrinkage_ <= 1
    assert np.all(np.isfinite(np.linalg.inv(oas.covariance_)))


def test_graphical_lasso_gives_a_sparse_precision():
    """The precision matrix should be sparse -- a zero means two variables are
    conditionally independent."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 8))
    gl = GraphicalLasso(alpha=0.3).fit(X)
    off_diag = gl.precision_[~np.eye(8, dtype=bool)]
    assert np.mean(np.abs(off_diag) < 1e-3) > 0.3    # many off-diagonals zeroed


def test_min_cov_det_ignores_outliers():
    """Robust covariance: the estimate should track the clean majority, not the
    outliers a plain covariance would be dragged by."""
    rng = np.random.RandomState(0)
    clean = rng.normal(0, 1, (180, 3))
    outliers = rng.normal(30, 1, (20, 3))
    X = np.vstack([clean, outliers])
    mcd = MinCovDet(random_state=0).fit(X)
    # the robust location sits near the clean centre, not pulled toward 30
    assert np.all(np.abs(mcd.location_) < 2.0)


def test_shrunk_covariance_interpolates():
    X = np.random.RandomState(0).normal(size=(50, 5))
    sample = EmpiricalCovariance().fit(X).covariance_
    heavy = ShrunkCovariance(shrinkage=0.9).fit(X).covariance_
    light = ShrunkCovariance(shrinkage=0.1).fit(X).covariance_
    # heavier shrinkage pulls further from the sample covariance
    assert np.sum((heavy - sample) ** 2) > np.sum((light - sample) ** 2)


def test_mahalanobis_distance_is_larger_for_outliers():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (300, 3))
    cov = EmpiricalCovariance().fit(X)
    d_inlier = cov.mahalanobis(np.zeros((1, 3)))
    d_outlier = cov.mahalanobis(np.full((1, 3), 5.0))
    assert d_outlier[0] > d_inlier[0]


# --- cross-decomposition --------------------------------------------------

@pytest.fixture
def coupled_blocks():
    """Two blocks sharing a rank-2 latent structure, plus noise."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(150, 20))
    Y = X[:, :2] @ rng.normal(size=(2, 3)) + rng.normal(0, 0.1, (150, 3))
    return X, Y


def test_pls_predicts_on_wide_collinear_data(coupled_blocks):
    """PLS should recover the X->Y map even with more features than clean signal."""
    X, Y = coupled_blocks
    pls = PLSRegression(n_components=3).fit(X, Y)
    pred = pls.predict(X)
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    assert r2 > 0.9


def test_pls_components_reduce_dimension(coupled_blocks):
    X, Y = coupled_blocks
    pls = PLSRegression(n_components=3).fit(X, Y)
    assert pls.transform(X).shape == (150, 3)


def test_cca_finds_correlated_directions(coupled_blocks):
    """The shared latent structure should show up as near-perfect canonical
    correlations."""
    X, Y = coupled_blocks
    cca = CCA(n_components=2).fit(X, Y)
    assert cca.correlations_[0] > 0.95           # the shared factors correlate ~1
    # the fitted directions are ordered by correlation
    assert cca.correlations_[0] >= cca.correlations_[1]


def test_cca_transform_projects_both_blocks(coupled_blocks):
    X, Y = coupled_blocks
    cca = CCA(n_components=2).fit(X, Y)
    Xs, Ys = cca.transform(X, Y)
    assert Xs.shape == (150, 2) and Ys.shape == (150, 2)
    # the projected pairs are strongly correlated, by construction
    assert abs(np.corrcoef(Xs[:, 0], Ys[:, 0])[0, 1]) > 0.9


# --- dummies --------------------------------------------------------------

def test_dummy_classifier_predicts_the_majority():
    y = np.array([0] * 90 + [1] * 10)
    X = np.zeros((100, 3))
    dc = DummyClassifier(strategy="most_frequent").fit(X, y)
    assert np.all(dc.predict(X) == 0)
    # it scores 0.9 while learning nothing -- the point of having a baseline
    assert (dc.predict(X) == y).mean() == 0.9


def test_dummy_classifier_prior_probabilities():
    y = np.array([0] * 75 + [1] * 25)
    dc = DummyClassifier().fit(np.zeros((100, 2)), y)
    proba = dc.predict_proba(np.zeros((5, 2)))
    assert np.allclose(proba[0], [0.75, 0.25])


def test_dummy_regressor_predicts_the_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    dr = DummyRegressor(strategy="mean").fit(np.zeros((4, 2)), y)
    assert np.all(dr.predict(np.zeros((3, 2))) == 2.5)


def test_r2_of_the_mean_dummy_is_zero():
    """r^2 is defined as improvement over DummyRegressor(mean), so the dummy
    itself scores exactly 0 -- which is why it is THE reference."""
    from nupyml.metrics import r2_score
    rng = np.random.RandomState(0)
    y = rng.normal(size=200)
    dr = DummyRegressor(strategy="mean").fit(np.zeros((200, 1)), y)
    assert abs(r2_score(y, dr.predict(np.zeros((200, 1))))) < 1e-9


# --- distribution transformers --------------------------------------------

def test_power_transformer_reduces_skew():
    """A skewed feature should come out much closer to symmetric."""
    skewed = np.random.RandomState(0).exponential(2, (500, 1))
    pt = PowerTransformer().fit(skewed)
    out = pt.transform(skewed)
    assert abs(skew(out)[0]) < abs(skew(skewed)[0]) / 3


def test_power_transformer_handles_negatives():
    """Yeo-Johnson (the default) works on negative values, unlike Box-Cox."""
    X = np.random.RandomState(0).normal(0, 2, (200, 1))
    out = PowerTransformer().fit_transform(X)
    assert np.all(np.isfinite(out))


def test_quantile_transformer_makes_data_uniform():
    X = np.random.RandomState(0).exponential(2, (2000, 1))
    out = QuantileTransformer(output_distribution="uniform").fit_transform(X)
    # a uniform distribution has quantiles evenly spread over [0, 1]
    assert abs(np.percentile(out, 25) - 0.25) < 0.05
    assert abs(np.percentile(out, 75) - 0.75) < 0.05


def test_quantile_transformer_makes_data_gaussian():
    X = np.random.RandomState(0).exponential(2, (2000, 1))
    out = QuantileTransformer(output_distribution="normal").fit_transform(X)
    assert abs(skew(out)[0]) < 0.2               # skew removed


def test_target_encoder_reflects_the_target():
    """Categories are replaced by their (smoothed) mean target."""
    X = np.array([["a"]] * 20 + [["b"]] * 20)
    y = np.array([1.0] * 20 + [0.0] * 20)
    te = TargetEncoder(smoothing=5).fit(X, y)
    assert te.transform([["a"]])[0, 0] > te.transform([["b"]])[0, 0]


def test_target_encoder_smooths_rare_categories_toward_the_prior():
    """A singleton category is pulled toward the global mean, not left as its own
    label -- the leak the smoothing bounds."""
    X = np.array([["common"]] * 100 + [["rare"]])
    y = np.array([0.5] * 100 + [10.0])            # the rare row has an extreme y
    te = TargetEncoder(smoothing=10).fit(X, y)
    # 'rare' is pulled well below its own value of 10 toward the prior
    assert te.transform([["rare"]])[0, 0] < 5.0


def test_function_transformer_applies_a_function():
    ft = FunctionTransformer(np.log1p, np.expm1)
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = ft.transform(X)
    assert np.allclose(out, np.log1p(X))
    assert np.allclose(ft.inverse_transform(out), X)   # round-trips


# --- factor analysis and incremental PCA ----------------------------------

def test_factor_analysis_recovers_latent_dimension():
    """Data driven by 2 factors should be well explained by 2 components."""
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(500, 2))
    W = rng.normal(size=(2, 6))
    X = Z @ W + rng.normal(0, 0.3, (500, 6))
    fa = FactorAnalysis(n_components=2).fit(X)
    assert fa.transform(X).shape == (500, 2)
    assert np.isfinite(fa.loglik_)


def test_factor_analysis_separates_noise_from_signal():
    """The property PCA lacks: a purely noisy feature gets a large noise variance,
    not a spurious factor loading."""
    rng = np.random.RandomState(0)
    # a shared factor drives ALL four features equally, but feature 3 carries
    # much larger INDEPENDENT noise on top -- PCA would rotate toward it, FA
    # should attribute it to per-feature noise instead
    Z = rng.normal(size=(800, 1))
    W = np.ones((1, 4))
    noise = np.column_stack([rng.normal(0, 0.2, 800), rng.normal(0, 0.2, 800),
                             rng.normal(0, 0.2, 800), rng.normal(0, 2.0, 800)])
    X = Z @ W + noise
    fa = FactorAnalysis(n_components=1).fit(X)
    # feature 3's noise variance dwarfs the others'
    assert np.argmax(fa.noise_variance_) == 3


def test_incremental_pca_matches_full_pca_at_full_batch():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 10)) @ rng.normal(size=(10, 10))
    ipca = IncrementalPCA(n_components=3).fit(X, batch_size=500)
    pca = PCA(n_components=3).fit(X)
    overlap = np.abs(ipca.components_ @ pca.components_.T).max(axis=1)
    assert np.all(overlap > 0.99)                # a full batch reproduces PCA


def test_incremental_pca_top_components_survive_batching():
    """The leading components are recovered even from small batches; only the
    weakest are approximate."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 10)) @ rng.normal(size=(10, 10))
    ipca = IncrementalPCA(n_components=3).fit(X, batch_size=100)
    pca = PCA(n_components=3).fit(X)
    overlap = np.abs(ipca.components_ @ pca.components_.T).max(axis=1)
    assert overlap[0] > 0.99 and overlap[1] > 0.99


# --- nearest centroid and radius neighbours -------------------------------

def test_nearest_centroid_classifies():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    assert NearestCentroid().fit(Xtr, ytr).score(Xte, yte) > 0.85


def test_nearest_centroid_shrinkage_selects_features():
    """Nearest shrunken centroids: shrinkage pulls each centroid toward the global
    one, so a threshold above zero moves them and can zero out weak features."""
    X, y = load_iris(return_X_y=True)
    nc = NearestCentroid(shrink_threshold=0.5).fit(X, y)
    plain = NearestCentroid().fit(X, y)
    assert not np.allclose(nc.centroids_, plain.centroids_)


def test_radius_neighbors_classifies():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rn = RadiusNeighborsClassifier(radius=2.0).fit(Xtr, ytr)
    assert rn.score(Xte, yte) > 0.8


def test_radius_neighbors_handles_empty_neighbourhoods():
    """A query with no neighbours in the radius must not crash -- the failure mode
    kNN cannot have."""
    X = np.array([[0.0], [0.1], [0.2]])
    y = np.array([0, 0, 1])
    rn = RadiusNeighborsClassifier(radius=0.01).fit(X, y)
    pred = rn.predict(np.array([[100.0]]))       # nothing within radius
    assert pred.shape == (1,)


# --- feature hashing ------------------------------------------------------

def test_feature_hasher_needs_no_vocabulary():
    """The hashing trick: transform works with no fit, on features never seen."""
    fh = FeatureHasher(n_features=32)
    M = fh.transform([{"a": 1, "b": 2}, {"never_seen": 5}])
    assert M.shape == (2, 32)


def test_feature_hasher_is_deterministic():
    fh = FeatureHasher(n_features=64)
    a = fh.transform([{"x": 1, "y": 2}]).toarray()
    b = fh.transform([{"x": 1, "y": 2}]).toarray()
    assert np.array_equal(a, b)


def test_dict_vectorizer_one_hot_encodes_strings():
    dv = DictVectorizer().fit([{"x": 1, "city": "NY"}, {"x": 2, "city": "LA"}])
    assert "city=NY" in dv.feature_names_ and "city=LA" in dv.feature_names_
    out = dv.transform([{"x": 5, "city": "NY"}])
    assert out[0, dv.vocabulary_["x"]] == 5
    assert out[0, dv.vocabulary_["city=NY"]] == 1


def test_hashing_vectorizer_vectorizes_text():
    hv = HashingVectorizer(n_features=64)
    M = hv.transform(["the cat sat", "the dog ran"])
    assert M.shape == (2, 64)
    assert M.nnz > 0


# --- bisecting k-means and feature agglomeration --------------------------

def test_bisecting_kmeans_finds_clusters():
    X, y = load_iris(return_X_y=True)
    bk = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, bk.labels_) > 0.5
    assert len(np.unique(bk.labels_)) == 3


def test_feature_agglomeration_reduces_features():
    """Cluster the columns and merge each group into its mean -- interpretable
    dimensionality reduction."""
    X, _ = load_iris(return_X_y=True)
    fa = FeatureAgglomeration(n_clusters=2).fit(X)
    out = fa.transform(X)
    assert out.shape == (len(X), 2)


def test_feature_agglomeration_groups_correlated_features():
    """Correlated columns should land in the same group."""
    rng = np.random.RandomState(0)
    base = rng.normal(size=(200, 1))
    # columns 0,1 are copies of base; 2,3 are copies of another signal
    other = rng.normal(size=(200, 1))
    X = np.column_stack([base, base + 0.01 * rng.normal(size=(200, 1)),
                         other, other + 0.01 * rng.normal(size=(200, 1))])
    fa = FeatureAgglomeration(n_clusters=2).fit(X)
    # the two copies of each signal share a label
    assert fa.labels_[0] == fa.labels_[1]
    assert fa.labels_[2] == fa.labels_[3]
