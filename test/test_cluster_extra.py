import numpy as np
import pytest

import sklearn.cluster as skc
import sklearn.mixture as skm
import sklearn.manifold as skman

from nupyml.cluster import (Birch, OPTICS, AffinityPropagation, HDBSCAN,
                            KMeans, DBSCAN)
from nupyml.mixture import BayesianGaussianMixture, GaussianMixture
from nupyml.manifold import SpectralEmbedding, TSNEBarnesHut, TSNE
from nupyml.metrics import adjusted_rand_score as ari
from nupyml.datasets import make_blobs, make_moons, make_circles

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# Birch
# ---------------------------------------------------------------------------

def test_birch_recovers_blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    b = Birch(n_clusters=3, threshold=1.0).fit(X)
    assert ari(y, b.labels_) > 0.95
    assert np.array_equal(b.predict(X), b.labels_)


def test_birch_threshold_controls_subclusters():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=1)
    coarse = Birch(threshold=2.0, n_clusters=3).fit(X)
    fine = Birch(threshold=0.3, n_clusters=3).fit(X)
    assert len(fine.subcluster_centers_) > len(coarse.subcluster_centers_)


def test_birch_transform_gives_subcluster_distances():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=2)
    b = Birch(n_clusters=3, threshold=1.0).fit(X)
    D = b.transform(X)
    assert D.shape == (100, len(b.subcluster_centers_))
    assert (D >= 0).all()


def test_birch_without_global_clustering():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=3)
    b = Birch(n_clusters=None, threshold=1.0).fit(X)
    assert len(np.unique(b.labels_)) == len(b.subcluster_centers_)


# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------

def test_optics_orders_all_points():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=4)
    o = OPTICS(min_samples=5).fit(X)
    assert sorted(o.ordering_) == list(range(150))
    assert len(o.reachability_) == 150
    assert len(o.core_distances_) == 150


def test_optics_finds_clusters():
    # explicit centers: randomly drawn ones can overlap, and then no eps
    # separates them (sklearn's OPTICS merges them identically)
    centers = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    X, y = make_blobs(n_samples=180, centers=centers, cluster_std=0.5,
                      random_state=5)
    o = OPTICS(min_samples=5, eps=2.0).fit(X)
    core = o.labels_ != -1
    assert core.sum() == 180
    assert len(np.unique(o.labels_[core])) == 3
    assert ari(y[core], o.labels_[core]) > 0.95


def test_optics_matches_sklearn_dbscan_extraction():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=5)
    ours = OPTICS(min_samples=5, eps=1.0).fit(X)
    ref = skc.OPTICS(min_samples=5, cluster_method="dbscan", eps=1.0).fit(X)
    assert ari(ours.labels_, ref.labels_) > 0.95


def test_optics_reachability_is_low_within_clusters():
    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.3,
                      center_box=(-10, 10), random_state=6)
    o = OPTICS(min_samples=5).fit(X)
    finite = o.reachability_[np.isfinite(o.reachability_)]
    # a well-separated dataset produces a few tall spikes and many low values
    assert np.median(finite) < np.max(finite) / 3


# ---------------------------------------------------------------------------
# AffinityPropagation
# ---------------------------------------------------------------------------

def test_affinity_propagation_finds_exemplars():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=7)
    ap = AffinityPropagation(preference=-50, max_iter=400).fit(X)
    assert ap.converged_
    assert ari(y, ap.labels_) > 0.9
    assert len(ap.cluster_centers_indices_) == 3
    # exemplars are actual data points
    for pos, i in enumerate(ap.cluster_centers_indices_):
        assert np.allclose(X[i], ap.cluster_centers_[pos])


def test_affinity_propagation_preference_controls_cluster_count():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=8)
    # a higher (less negative) preference makes exemplars cheaper -> more of them
    few = AffinityPropagation(preference=-50, max_iter=400).fit(X)
    many = AffinityPropagation(preference=-5, max_iter=400).fit(X)
    assert few.converged_ and many.converged_
    assert len(many.cluster_centers_indices_) > len(few.cluster_centers_indices_)


def test_affinity_propagation_warns_when_not_converged():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=8)
    with pytest.warns(UserWarning, match="did not converge"):
        ap = AffinityPropagation(preference=-200, max_iter=50).fit(X)
    assert not ap.converged_


def test_affinity_propagation_predict_new_points():
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.4, random_state=9)
    ap = AffinityPropagation(preference=-50).fit(X)
    assert np.array_equal(ap.predict(X), ap.labels_)


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def test_hdbscan_recovers_blobs():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=10)
    h = HDBSCAN(min_cluster_size=10).fit(X)
    core = h.labels_ != -1
    assert core.sum() > 150
    assert ari(y[core], h.labels_[core]) > 0.9


def test_hdbscan_handles_varying_density():
    # DBSCAN with one global eps cannot fit both a tight and a loose cluster
    r = np.random.RandomState(11)
    tight = r.normal([0, 0], 0.15, size=(100, 2))
    loose = r.normal([6, 6], 1.2, size=(100, 2))
    X = np.vstack([tight, loose])
    y = np.r_[np.zeros(100), np.ones(100)]
    h = HDBSCAN(min_cluster_size=15).fit(X)
    core = h.labels_ != -1
    assert ari(y[core], h.labels_[core]) > 0.9


def test_hdbscan_marks_noise():
    X, _ = make_blobs(n_samples=150, centers=2, cluster_std=0.3, random_state=12)
    X = np.vstack([X, [[30.0, 30.0], [-30.0, -30.0]]])
    h = HDBSCAN(min_cluster_size=10).fit(X)
    assert (h.labels_[-2:] == -1).all()


# ---------------------------------------------------------------------------
# BayesianGaussianMixture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prior_type", ["dirichlet_process",
                                        "dirichlet_distribution"])
def test_bgmm_prunes_unneeded_components(prior_type):
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)
    bgmm = BayesianGaussianMixture(
        n_components=6, weight_concentration_prior_type=prior_type,
        random_state=0).fit(X)
    assert ari(y, bgmm.predict(X)) > 0.95
    # n_components is an upper bound: the extra components get ~zero weight
    assert (bgmm.weights_ > 0.05).sum() == 3
    assert np.allclose(bgmm.weights_.sum(), 1.0)


def test_bgmm_matches_sklearn_weight_profile():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)
    ours = BayesianGaussianMixture(n_components=6, random_state=0).fit(X)
    ref = skm.BayesianGaussianMixture(n_components=6, random_state=0).fit(X)
    assert ari(ours.predict(X), ref.predict(X)) > 0.95
    assert np.allclose(np.sort(ours.weights_)[::-1][:3],
                       np.sort(ref.weights_)[::-1][:3], atol=0.05)


def test_bgmm_lower_bound_improves_with_more_iterations():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=13)
    bounds = [BayesianGaussianMixture(n_components=4, max_iter=m, tol=0.0,
                                      random_state=0).fit(X).lower_bound_
              for m in (1, 5, 20)]
    # the bound omits terms constant in the parameters (as sklearn's does), so
    # it is not guaranteed monotone step-by-step, but must improve overall
    assert bounds[-1] > bounds[0]


def test_bgmm_tol_on_lower_bound_not_likelihood():
    """The mean log-likelihood plateaus while the weights are still sharpening,
    so convergence must be judged on the bound."""
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)
    b = BayesianGaussianMixture(n_components=6, tol=1e-3, random_state=0).fit(X)
    assert b.n_iter_ > 10
    assert ari(y, b.predict(X)) > 0.95


def test_bgmm_predict_proba_and_score():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=14)
    b = BayesianGaussianMixture(n_components=5, random_state=0).fit(X)
    proba = b.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.isfinite(b.score(X))
    assert len(b.score_samples(X)) == 150


def test_bgmm_concentration_prior_controls_sparsity():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=15)
    sparse = BayesianGaussianMixture(n_components=8,
                                     weight_concentration_prior=1e-3,
                                     random_state=0).fit(X)
    dense = BayesianGaussianMixture(n_components=8,
                                    weight_concentration_prior=1e3,
                                    random_state=0).fit(X)
    assert (sparse.weights_ > 0.05).sum() <= (dense.weights_ > 0.05).sum()


# ---------------------------------------------------------------------------
# SpectralEmbedding
# ---------------------------------------------------------------------------

def test_spectral_embedding_unfolds_moons():
    X, y = make_moons(200, noise=0.05, random_state=16)
    emb = SpectralEmbedding(n_components=2, n_neighbors=10).fit_transform(X)
    assert emb.shape == (200, 2)
    # the leading eigenvector separates the two moons
    assert ari(y, (emb[:, 0] > np.median(emb[:, 0])).astype(int)) > 0.8


def test_spectral_embedding_rbf_affinity():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.4, random_state=17)
    emb = SpectralEmbedding(n_components=3, affinity="rbf",
                            gamma=0.1).fit_transform(X)
    assert emb.shape == (150, 3)
    labels = KMeans(n_clusters=3, random_state=0).fit(emb).labels_
    assert ari(y, labels) > 0.9


def test_spectral_embedding_beats_kmeans_on_circles():
    X, y = make_circles(200, noise=0.05, factor=0.4, random_state=18)
    emb = SpectralEmbedding(n_components=2, n_neighbors=10).fit_transform(X)
    spectral = ari(y, KMeans(n_clusters=2, random_state=0).fit(emb).labels_)
    raw = ari(y, KMeans(n_clusters=2, random_state=0).fit(X).labels_)
    assert spectral > raw


# ---------------------------------------------------------------------------
# Barnes-Hut t-SNE
# ---------------------------------------------------------------------------

def test_barnes_hut_tsne_separates_blobs():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=19)
    emb = TSNEBarnesHut(perplexity=15, max_iter=800,
                        random_state=0).fit_transform(X)
    assert emb.shape == (200, 2)
    labels = KMeans(n_clusters=3, random_state=0).fit(emb).labels_
    assert ari(y, labels) > 0.85


def test_barnes_hut_tsne_rejects_non_2d():
    X, _ = make_blobs(n_samples=50, centers=2, random_state=20)
    with pytest.raises(ValueError, match="2-D only"):
        TSNEBarnesHut(n_components=3).fit_transform(X)


def test_barnes_hut_small_angle_is_more_accurate():
    """angle trades accuracy for speed: a smaller theta means fewer cells are
    collapsed into their center of mass, so the forces are closer to exact."""
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.5, random_state=21)
    scores = {}
    for angle in (0.2, 0.8):
        emb = TSNEBarnesHut(perplexity=10, max_iter=600, angle=angle,
                            random_state=0).fit_transform(X)
        labels = KMeans(n_clusters=3, random_state=0).fit(emb).labels_
        scores[angle] = ari(y, labels)
    assert scores[0.2] > 0.8
    assert scores[0.2] >= scores[0.8]


def test_barnes_hut_matches_exact_tsne_quality():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=22)
    bh = TSNEBarnesHut(perplexity=15, max_iter=600, random_state=0).fit_transform(X)
    exact = TSNE(perplexity=15, max_iter=600, random_state=0).fit_transform(X)
    bh_ari = ari(y, KMeans(n_clusters=3, random_state=0).fit(bh).labels_)
    ex_ari = ari(y, KMeans(n_clusters=3, random_state=0).fit(exact).labels_)
    assert bh_ari > ex_ari - 0.15
