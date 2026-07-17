"""F6: LODA, feature bagging, half-space trees, Mahalanobis, PCA-reconstruction,
and automatic thresholding.

Detectors are scored by ROC-AUC against hidden outlier labels on data suited to
each: the projection/ensemble/tree/PCA detectors on two clusters with scattered
outliers, and Mahalanobis on a single Gaussian blob (its unimodal assumption).
Thresholding functions must flag planted extremes and spare the bulk.
"""
import numpy as np
import pytest

from nupyml.anomaly import (LODA, FeatureBaggingDetector, HalfSpaceTrees,
                           MahalanobisDetector, PCAReconstructionDetector,
                           threshold_iqr, threshold_mad, threshold_gesd)
from nupyml.metrics import roc_auc_score


def _clustered(seed=0, d=8):
    rng = np.random.RandomState(seed)
    nin = 475
    inl = np.vstack([rng.randn(nin // 2, d) + 3, rng.randn(nin - nin // 2, d) - 3])
    out = rng.uniform(-9, 9, (25, d))
    X = np.vstack([inl, out])
    y = np.r_[np.zeros(nin), np.ones(25)]
    p = rng.permutation(len(y))
    return X[p], y[p]


def _single_blob(seed=1, d=6):
    rng = np.random.RandomState(seed)
    inl = rng.randn(480, d)
    out = rng.randn(20, d) * 1.0 + 8.0
    X = np.vstack([inl, out])
    y = np.r_[np.zeros(480), np.ones(20)]
    p = rng.permutation(len(y))
    return X[p], y[p]


@pytest.mark.parametrize("make_det", [
    lambda: LODA(n_projections=100, random_state=0),
    lambda: FeatureBaggingDetector(n_estimators=8, random_state=0),
    lambda: HalfSpaceTrees(n_estimators=25, random_state=0),
    lambda: PCAReconstructionDetector(n_components=3),
], ids=["loda", "feature_bagging", "half_space_trees", "pca_recon"])
def test_detector_ranks_outliers_above_inliers(make_det):
    X, y = _clustered()
    det = make_det().fit(X)
    assert roc_auc_score(y, det.decision_function(X)) > 0.9


def test_mahalanobis_flags_far_points_on_a_single_blob():
    X, y = _single_blob()
    det = MahalanobisDetector().fit(X)
    assert roc_auc_score(y, det.decision_function(X)) > 0.95


def test_predict_labels_use_the_contamination_threshold():
    X, y = _clustered()
    det = LODA(contamination=0.05, random_state=0).fit(X)
    pred = det.predict(X)
    assert set(np.unique(pred)).issubset({-1, 1})
    # about the contamination fraction are called outliers (-1)
    assert 0.02 < np.mean(pred == -1) < 0.1


def test_half_space_trees_handles_a_constant_feature():
    """A zero-range feature must not break the random-threshold split."""
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.randn(200), np.ones(200)])   # 2nd feature constant
    det = HalfSpaceTrees(n_estimators=10, random_state=0).fit(X)
    s = det.decision_function(X)
    assert np.all(np.isfinite(s))


def test_pca_reconstruction_error_is_zero_on_the_subspace():
    """Points that lie exactly on the principal subspace reconstruct perfectly."""
    rng = np.random.RandomState(0)
    # data spanning a 2-D subspace embedded in 5-D
    basis = rng.randn(2, 5)
    X = rng.randn(300, 2) @ basis
    det = PCAReconstructionDetector(n_components=2).fit(X)
    assert det.decision_function(X).max() < 1e-6


# --- thresholding ---------------------------------------------------------

@pytest.fixture
def scores_with_extremes():
    rng = np.random.RandomState(0)
    return np.r_[rng.randn(100), rng.randn(5) + 8]   # 5 clear high outliers


def test_thresholds_flag_the_planted_extremes(scores_with_extremes):
    s = scores_with_extremes
    for fn in (threshold_iqr, threshold_mad, threshold_gesd):
        mask = fn(s)
        # the 5 extremes (indices 100..104) are all flagged...
        assert mask[100:].all(), fn.__name__
        # ...and not too many of the bulk
        assert mask[:100].sum() <= 5, fn.__name__


def test_mad_threshold_is_robust_to_a_few_extremes(scores_with_extremes):
    """MAD uses the median, so a handful of huge scores don't inflate the scale
    and mask themselves (unlike a mean/std z-score would)."""
    s = scores_with_extremes.copy()
    s[100:] = 100.0                                   # make them enormous
    assert threshold_mad(s)[100:].all()
