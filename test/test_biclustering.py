"""F5: co-clustering, density-peak, and consensus clustering.

Co-clustering must recover a planted row-and-column block structure; density-peak
and consensus must recover well-separated blobs; consensus must additionally
produce a near-binary co-association matrix when the structure is stable.
"""
import numpy as np
import pytest

from nupyml.cluster import (SpectralCoclustering, DensityPeakClustering,
                           ConsensusClustering)
from nupyml.datasets import make_blobs
from nupyml.metrics import adjusted_rand_score


def _block_matrix(seed=0):
    rng = np.random.RandomState(seed)
    A = np.zeros((60, 45))
    for b in range(3):
        A[b * 20:(b + 1) * 20, b * 15:(b + 1) * 15] = rng.rand(20, 15) + 1.0
    pr, pc = rng.permutation(60), rng.permutation(45)
    true_r = np.repeat([0, 1, 2], 20)[pr]
    true_c = np.repeat([0, 1, 2], 15)[pc]
    return A[pr][:, pc], true_r, true_c


def test_coclustering_recovers_row_and_column_blocks():
    A, true_r, true_c = _block_matrix()
    cc = SpectralCoclustering(n_clusters=3, random_state=0).fit(A)
    assert adjusted_rand_score(true_r, cc.row_labels_) > 0.9
    assert adjusted_rand_score(true_c, cc.column_labels_) > 0.9


def test_density_peak_recovers_blobs():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
    dp = DensityPeakClustering(n_clusters=4).fit(X)
    assert adjusted_rand_score(y, dp.labels_) > 0.85
    assert len(dp.centres_) == 4


def test_density_peak_centres_are_distinct_points():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=1)
    dp = DensityPeakClustering(n_clusters=3).fit(X)
    assert len(set(dp.centres_.tolist())) == 3      # three different data points


def test_consensus_recovers_blobs_and_is_stable():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
    cc = ConsensusClustering(n_clusters=4, n_runs=15, random_state=0).fit(X)
    assert adjusted_rand_score(y, cc.labels_) > 0.85
    # co-association is a probability matrix; stable structure -> near 0/1 values
    M = cc.coassociation_
    assert M.shape == (300, 300)
    assert (M >= 0).all() and (M <= 1).all()
    off = M[~np.eye(300, dtype=bool)]
    near_binary = np.mean((off < 0.1) | (off > 0.9))
    assert near_binary > 0.8


def test_consensus_diagonal_is_one():
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.7, random_state=2)
    cc = ConsensusClustering(n_clusters=3, n_runs=10, random_state=0).fit(X)
    # a point is always co-clustered with itself when co-sampled
    assert np.allclose(np.diag(cc.coassociation_), 1.0)
