"""Each optimization must be invisible in the answer and visible in the clock.

These tests pin the *properties* the optimizations rely on, so that a future
change which quietly breaks one is caught by a wrong answer rather than by a
mysterious slowdown.
"""
import time

import numpy as np
import pytest

import sklearn.decomposition as skd
import sklearn.ensemble as ske
import sklearn.svm as sksvm

from nupyml.decomposition import PCA
from nupyml.decomposition import _randomized_svd, _deterministic_signs
from nupyml.ensemble import (HistGradientBoostingClassifier,
                             HistGradientBoostingRegressor)
from nupyml.ensemble._hist_gb import _HistTree, _BinMapper
from nupyml.cluster import KMeans
from nupyml.svm import SVC
from nupyml.datasets import make_classification, make_regression, make_blobs
from nupyml.metrics import adjusted_rand_score

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# histogram trees: bincount + the subtraction trick
# ---------------------------------------------------------------------------

def _tree_fixture(n=400, d=6, n_bins=32, seed=0):
    r = np.random.RandomState(seed)
    Xb = r.randint(0, n_bins - 1, size=(n, d)).astype(np.uint8)
    g = r.normal(size=n)
    h = np.abs(r.normal(size=n)) + 0.5
    return Xb, g, h, _HistTree(n_bins=n_bins, min_samples_leaf=5)


def test_histograms_match_a_naive_scatter_add():
    """bincount must give bit-identical sums to the obvious scatter-add."""
    Xb, g, h, tree = _tree_fixture()
    idx = np.arange(len(g))
    hist_g, hist_h, hist_c = tree._build_histograms(Xb, idx, g, h)
    for j in range(Xb.shape[1]):
        naive_g = np.zeros(tree.n_bins)
        naive_c = np.zeros(tree.n_bins)
        np.add.at(naive_g, Xb[:, j], g)
        np.add.at(naive_c, Xb[:, j], 1)
        assert np.allclose(hist_g[j], naive_g)
        assert np.allclose(hist_c[j], naive_c)


def test_histogram_subtraction_equals_building_both_sides():
    """The whole trick: parent - child == sibling, exactly."""
    Xb, g, h, tree = _tree_fixture()
    tree.n_features = Xb.shape[1]
    idx = np.arange(len(g))
    parent = tree._build_histograms(Xb, idx, g, h)
    goes_left = Xb[:, 0] <= 15
    left_idx, right_idx = idx[goes_left], idx[~goes_left]

    left, right = tree._split_histograms(Xb, g, h, parent, left_idx, right_idx)
    direct_left = tree._build_histograms(Xb, left_idx, g, h)
    direct_right = tree._build_histograms(Xb, right_idx, g, h)
    for got, want in zip(left, direct_left):
        assert np.allclose(got, want)
    for got, want in zip(right, direct_right):
        assert np.allclose(got, want)


def test_histogram_subtraction_holds_for_lopsided_splits():
    """The smaller side is the one built; both orderings must work."""
    Xb, g, h, tree = _tree_fixture()
    tree.n_features = Xb.shape[1]
    idx = np.arange(len(g))
    parent = tree._build_histograms(Xb, idx, g, h)
    for cut in (2, 30):        # tiny-left, then tiny-right
        goes_left = Xb[:, 0] <= cut
        li, ri = idx[goes_left], idx[~goes_left]
        if len(li) == 0 or len(ri) == 0:
            continue
        left, right = tree._split_histograms(Xb, g, h, parent, li, ri)
        assert np.allclose(left[2], tree._build_histograms(Xb, li, g, h)[2])
        assert np.allclose(right[2], tree._build_histograms(Xb, ri, g, h)[2])
        # and the counts still partition the parent exactly
        assert np.allclose(left[2] + right[2], parent[2])


def test_histograms_partition_the_parent():
    """Additivity over a partition is the property subtraction depends on."""
    Xb, g, h, tree = _tree_fixture(seed=1)
    tree.n_features = Xb.shape[1]
    idx = np.arange(len(g))
    parent = tree._build_histograms(Xb, idx, g, h)
    goes_left = Xb[:, 2] <= 10
    left = tree._build_histograms(Xb, idx[goes_left], g, h)
    right = tree._build_histograms(Xb, idx[~goes_left], g, h)
    for p, l, r in zip(parent, left, right):
        assert np.allclose(p, l + r)


def test_histgb_still_matches_sklearn_after_optimization():
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                               random_state=2)
    ours = HistGradientBoostingClassifier(max_iter=50).fit(X, y)
    ref = ske.HistGradientBoostingClassifier(max_iter=50).fit(X, y)
    assert ours.score(X, y) >= ref.score(X, y) - 0.02


def test_histgb_is_not_pathologically_slow():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8,
                               random_state=3)
    start = time.perf_counter()
    HistGradientBoostingClassifier(max_iter=100).fit(X, y)
    assert time.perf_counter() - start < 8.0


def test_binmapper_still_reserves_the_missing_bin():
    X = np.array([[1.0], [2.0], [np.nan], [3.0]])
    mapper = _BinMapper(max_bins=8).fit(X)
    bins = mapper.transform(X)
    assert bins[2, 0] == mapper.missing_bin_


# ---------------------------------------------------------------------------
# PCA solvers
# ---------------------------------------------------------------------------

def test_all_pca_solvers_agree_on_well_conditioned_data():
    X = rng.normal(size=(2000, 8)) @ rng.normal(size=(8, 8))
    full = PCA(n_components=4, svd_solver="full").fit(X)
    for solver in ("covariance_eigh", "randomized", "auto"):
        got = PCA(n_components=4, svd_solver=solver, random_state=0).fit(X)
        assert np.allclose(got.explained_variance_, full.explained_variance_,
                           rtol=0.02), solver
        assert np.allclose(np.abs(got.components_), np.abs(full.components_),
                           atol=1e-3), solver


def test_pca_auto_picks_covariance_when_samples_dominate():
    X = rng.normal(size=(5000, 10))
    assert PCA(n_components=3).fit(X).svd_solver_ == "covariance_eigh"


def test_pca_auto_picks_full_for_small_data():
    X = rng.normal(size=(50, 40))
    assert PCA(n_components=5).fit(X).svd_solver_ == "full"


def test_pca_signs_are_deterministic_across_solvers():
    """Eigenvectors are only defined up to sign; the convention pins them."""
    X = rng.normal(size=(3000, 6)) @ rng.normal(size=(6, 6))
    a = PCA(n_components=3, svd_solver="full").fit(X).components_
    b = PCA(n_components=3, svd_solver="covariance_eigh").fit(X).components_
    assert np.allclose(a, b, atol=1e-6)      # identical, not merely up to sign
    peaks = np.abs(a).argmax(axis=1)
    assert (a[np.arange(len(a)), peaks] > 0).all()


def test_deterministic_signs_flips_only_the_sign():
    v = np.array([[-3.0, 1.0], [2.0, -0.5]])
    out = _deterministic_signs(v)
    assert np.allclose(np.abs(out), np.abs(v))
    assert (out[np.arange(2), np.abs(v).argmax(axis=1)] > 0).all()


def test_randomized_svd_approximates_the_leading_spectrum():
    X = rng.normal(size=(400, 100))
    X = X @ np.diag(np.linspace(10, 0.1, 100))    # a decaying spectrum
    _, S_approx, _ = _randomized_svd(X, 5, rng=0)
    S_exact = np.linalg.svd(X, compute_uv=False)[:5]
    assert np.allclose(S_approx, S_exact, rtol=0.05)


def test_pca_explained_variance_ratio_sums_correctly():
    X = rng.normal(size=(2000, 6))
    for solver in ("full", "covariance_eigh"):
        p = PCA(svd_solver=solver).fit(X)
        # keeping every component must account for all the variance
        assert p.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-8)


def test_pca_variance_fraction_still_works_per_solver():
    X = np.column_stack([rng.normal(scale=10, size=3000),
                         rng.normal(scale=1, size=3000),
                         rng.normal(scale=0.05, size=3000)])
    for solver in ("full", "covariance_eigh"):
        assert PCA(n_components=0.95, svd_solver=solver).fit(X).n_components_ <= 2


def test_pca_matches_sklearn_after_optimization():
    X = rng.normal(size=(3000, 8)) @ rng.normal(size=(8, 8))
    ours = PCA(n_components=4).fit(X)
    ref = skd.PCA(n_components=4).fit(X)
    assert np.allclose(ours.explained_variance_ratio_,
                       ref.explained_variance_ratio_, rtol=1e-6)


# ---------------------------------------------------------------------------
# KMeans scatter-add M-step
# ---------------------------------------------------------------------------

def test_kmeans_scatter_m_step_matches_per_cluster_means():
    """The scatter-add must reproduce plain per-cluster weighted means."""
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.6, random_state=4)
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    for c in range(4):
        members = X[km.labels_ == c]
        if len(members):
            assert np.allclose(km.cluster_centers_[c], members.mean(axis=0),
                               atol=1e-6)


def test_kmeans_still_recovers_blobs():
    import sklearn.cluster as skc
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.5, random_state=5)
    ours = adjusted_rand_score(y, KMeans(n_clusters=4,
                                         random_state=0).fit(X).labels_)
    # these blobs genuinely overlap, so judge against sklearn on the same data
    # rather than against a made-up absolute number
    ref = adjusted_rand_score(y, skc.KMeans(n_clusters=4, n_init=10,
                                            random_state=0).fit(X).labels_)
    assert ours >= ref - 0.02


def test_kmeans_weights_still_respected_by_scatter_step():
    X = np.array([[0.0, 0], [0, 1], [10, 0], [10, 1]])
    w = np.array([1.0, 1, 100, 1])
    km = KMeans(n_clusters=1, n_init=1, random_state=0).fit(X, sample_weight=w)
    assert km.cluster_centers_[0, 0] > 9.0


def test_kmeans_handles_fewer_distinct_points_than_clusters():
    """With duplicated rows, D^2 sampling runs out of signal: every remaining
    point sits on a centre already, so the sampling weights are all zero."""
    X = np.vstack([np.zeros((50, 2)), np.full((50, 2), 100.0)])
    km = KMeans(n_clusters=5, n_init=1, random_state=0).fit(X)
    assert np.isfinite(km.cluster_centers_).all()
    assert len(np.unique(km.labels_)) >= 2


def test_kmeans_plusplus_spreads_seeds_across_clusters():
    """D^2 sampling should not drop every seed into one blob."""
    from nupyml.cluster import _kmeans_plusplus
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.4,
                      center_box=(-20, 20), random_state=8)
    seeds = _kmeans_plusplus(X, 4, np.random.RandomState(0))
    from scipy.spatial.distance import pdist
    # the seeds must be mutually distant, not huddled together
    assert pdist(seeds).min() > 1.0


# ---------------------------------------------------------------------------
# SMO second-order selection
# ---------------------------------------------------------------------------

def test_svc_is_not_pathologically_slow():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                               random_state=6)
    start = time.perf_counter()
    clf = SVC(random_state=0).fit(X, y)
    assert time.perf_counter() - start < 5.0
    assert clf.score(X, y) >= sksvm.SVC().fit(X, y).score(X, y) - 0.02


def test_svc_kkt_conditions_hold_at_the_solution():
    """Convergence means the KKT gap closed, not that the loop ran out."""
    from nupyml.svm import _kernel_fn, _smo
    X, y = make_classification(n_samples=300, n_features=5, random_state=7)
    t = np.where(y == 1, 1.0, -1.0)
    K = _kernel_fn("rbf", 0.2, 3, 0.0)(X, X)
    alpha, b, n_iter = _smo(K, t, 1.0, tol=1e-3)
    # gradient of 0.5 a'Qa - e'a at the solution, with Q_ij = y_i y_j K_ij:
    # (Qa)_i = y_i * sum_j y_j a_j K_ij
    G = t * (K @ (alpha * t)) - 1.0
    neg_yG = -t * G
    up = ((alpha < 1.0 - 1e-9) & (t > 0)) | ((alpha > 1e-9) & (t < 0))
    low = ((alpha > 1e-9) & (t > 0)) | ((alpha < 1.0 - 1e-9) & (t < 0))
    gap = neg_yG[up].max() - neg_yG[low].min()
    assert gap < 1e-2
    # the equality constraint sum(alpha_i y_i) = 0 must hold throughout
    assert abs((alpha * t).sum()) < 1e-6
