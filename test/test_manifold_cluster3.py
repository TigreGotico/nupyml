"""I9: manifold + clustering v3 -- diffusion maps, PHATE, DP-means, possibilistic
c-means, sparse subspace clustering.

The manifold methods must produce an embedding that separates well-separated
blobs; DP-means must discover the right number of clusters from a distance
penalty; possibilistic c-means must give a gross outlier near-zero typicality;
SSC must group points by their subspace even when the subspaces cross.
"""
import numpy as np
import pytest

from nupyml.manifold import DiffusionMap, PHATE
from nupyml.cluster import (KMeans, DPMeans, PossibilisticCMeans,
                            SparseSubspaceClustering)
from nupyml.metrics import adjusted_rand_score


def _blobs(rng):
    X = np.vstack([rng.randn(40, 2) + c for c in ([0, 0], [8, 8], [0, 8])])
    y = np.repeat([0, 1, 2], 40)
    return X, y


def test_diffusion_map_embedding_separates_blobs():
    rng = np.random.RandomState(0)
    X, y = _blobs(rng)
    emb = DiffusionMap(n_components=2, t=2).fit_transform(X)
    assert emb.shape == (120, 2)
    labels = KMeans(3, random_state=0).fit_predict(emb)
    assert adjusted_rand_score(y, labels) > 0.9


def test_phate_embedding_separates_blobs():
    rng = np.random.RandomState(0)
    X, y = _blobs(rng)
    emb = PHATE(n_components=2, t=3, knn=5).fit_transform(X)
    assert emb.shape == (120, 2)
    labels = KMeans(3, random_state=0).fit_predict(emb)
    assert adjusted_rand_score(y, labels) > 0.9


def test_dpmeans_discovers_cluster_count_and_lambda_controls_it():
    rng = np.random.RandomState(0)
    X, y = _blobs(rng)
    dp = DPMeans(lam=30.0, random_state=0).fit(X)
    assert dp.n_clusters_ == 3                        # found k without being told
    assert adjusted_rand_score(y, dp.labels_) > 0.9
    # a much larger penalty yields fewer clusters (never more)
    coarse = DPMeans(lam=500.0, random_state=0).fit(X)
    assert coarse.n_clusters_ <= dp.n_clusters_


def test_possibilistic_cmeans_gives_outlier_near_zero_typicality():
    rng = np.random.RandomState(0)
    X, y = _blobs(rng)
    pcm = PossibilisticCMeans(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, pcm.labels_) > 0.9
    # an extreme outlier belongs to nothing -- its max typicality is ~0
    Xout = np.vstack([X, [[100.0, 100.0]]])
    pcm2 = PossibilisticCMeans(n_clusters=3, random_state=0).fit(Xout)
    assert pcm2.typicalities_[-1].max() < 0.05


def test_sparse_subspace_clustering_separates_crossing_lines():
    rng = np.random.RandomState(0)
    t = np.linspace(-1, 1, 40)
    line1 = np.c_[t, 2 * t] + 0.01 * rng.randn(40, 2)
    line2 = np.c_[t, -0.5 * t] + 0.01 * rng.randn(40, 2)
    X = np.vstack([line1, line2])
    y = np.repeat([0, 1], 40)
    ssc = SparseSubspaceClustering(n_clusters=2, n_nonzero=4,
                                   random_state=0).fit(X)
    # the two lines cross at the origin, where distance-based clustering fails
    assert adjusted_rand_score(y, ssc.labels_) > 0.85
