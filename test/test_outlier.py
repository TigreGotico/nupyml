import numpy as np
import pytest

import sklearn.ensemble as ske
import sklearn.neighbors as sknn
import sklearn.svm as sksvm
import sklearn.covariance as skcov

from nupyml.outlier import (IsolationForest, LocalOutlierFactor, OneClassSVM,
                            EllipticEnvelope)
from nupyml.metrics import roc_auc_score

rng = np.random.RandomState(0)


def _inliers_with_outliers(n_in=200, n_out=20, seed=0):
    r = np.random.RandomState(seed)
    inliers = r.normal(0, 1, size=(n_in, 2))
    outliers = r.uniform(-8, 8, size=(n_out, 2))
    # keep only genuinely distant outliers so the ground truth is unambiguous
    outliers = outliers[np.linalg.norm(outliers, axis=1) > 5][:n_out]
    X = np.vstack([inliers, outliers])
    is_outlier = np.r_[np.zeros(len(inliers)), np.ones(len(outliers))]
    return X, is_outlier


def _detects_outliers(scores, is_outlier, min_auc=0.9):
    """scores: higher = more anomalous."""
    return roc_auc_score(is_outlier, scores) > min_auc


def test_isolation_forest_detects_outliers():
    X, is_outlier = _inliers_with_outliers()
    iso = IsolationForest(n_estimators=100, random_state=0).fit(X)
    assert _detects_outliers(-iso.score_samples(X), is_outlier)
    pred = iso.predict(X)
    assert set(np.unique(pred)) <= {-1, 1}
    # most flagged points should be true outliers
    assert (is_outlier[pred == -1]).mean() > 0.5


def test_isolation_forest_comparable_to_sklearn():
    X, is_outlier = _inliers_with_outliers(seed=1)
    ours = roc_auc_score(is_outlier, -IsolationForest(
        n_estimators=100, random_state=0).fit(X).score_samples(X))
    ref = roc_auc_score(is_outlier, -ske.IsolationForest(
        n_estimators=100, random_state=0).fit(X).score_samples(X))
    assert ours >= ref - 0.1


def test_isolation_forest_contamination_controls_flag_rate():
    X, _ = _inliers_with_outliers(seed=2)
    for c in (0.05, 0.2):
        iso = IsolationForest(contamination=c, random_state=0).fit(X)
        flagged = (iso.predict(X) == -1).mean()
        assert abs(flagged - c) < 0.05


def test_isolation_forest_average_path_length():
    # closed form must match the harmonic-number definition for small n
    iso = IsolationForest()
    assert iso._average_path_length(np.array([1]))[0] == 0.0
    assert iso._average_path_length(np.array([2]))[0] == 1.0
    assert iso._average_path_length(np.array([10]))[0] > 3.0


def test_local_outlier_factor_detects_outliers():
    X, is_outlier = _inliers_with_outliers(seed=3)
    lof = LocalOutlierFactor(n_neighbors=20).fit(X)
    assert _detects_outliers(-lof.negative_outlier_factor_, is_outlier)
    pred = lof.fit_predict(X)
    assert (is_outlier[pred == -1]).mean() > 0.5


def test_local_outlier_factor_matches_sklearn_scores():
    X, _ = _inliers_with_outliers(seed=4)
    ours = LocalOutlierFactor(n_neighbors=20).fit(X)
    ref = sknn.LocalOutlierFactor(n_neighbors=20).fit(X)
    assert np.corrcoef(ours.negative_outlier_factor_,
                       ref.negative_outlier_factor_)[0, 1] > 0.95


def test_lof_novelty_mode():
    X, _ = _inliers_with_outliers(seed=5)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X)
    new_normal = np.zeros((5, 2))
    new_weird = np.full((5, 2), 20.0)
    assert (lof.predict(new_normal) == 1).all()
    assert (lof.predict(new_weird) == -1).all()


def test_lof_score_samples_requires_novelty():
    X, _ = _inliers_with_outliers(seed=6)
    lof = LocalOutlierFactor(novelty=False).fit(X)
    with pytest.raises(AttributeError, match="novelty=True"):
        lof.score_samples(X)


def test_one_class_svm_detects_outliers():
    X, is_outlier = _inliers_with_outliers(seed=7)
    oc = OneClassSVM(nu=0.1, gamma=0.1).fit(X)
    assert _detects_outliers(-oc.score_samples(X), is_outlier, min_auc=0.85)
    assert len(oc.support_vectors_) > 0


def test_one_class_svm_comparable_to_sklearn():
    X, is_outlier = _inliers_with_outliers(seed=8)
    ours = roc_auc_score(is_outlier, -OneClassSVM(nu=0.1, gamma=0.1)
                         .fit(X).score_samples(X))
    ref = roc_auc_score(is_outlier, -sksvm.OneClassSVM(nu=0.1, gamma=0.1)
                        .fit(X).score_samples(X))
    assert ours >= ref - 0.1


def test_one_class_svm_novelty():
    X = rng.normal(0, 1, size=(200, 2))
    oc = OneClassSVM(nu=0.1, gamma=0.2).fit(X)
    assert (oc.predict(np.zeros((3, 2))) == 1).all()
    assert (oc.predict(np.full((3, 2), 15.0)) == -1).all()


def test_elliptic_envelope_detects_outliers():
    X, is_outlier = _inliers_with_outliers(seed=9)
    ee = EllipticEnvelope(random_state=0).fit(X)
    assert _detects_outliers(ee.mahalanobis(X), is_outlier)
    assert (is_outlier[ee.predict(X) == -1]).mean() > 0.5


def test_elliptic_envelope_robust_location():
    # MCD must ignore the contamination when estimating the center
    r = np.random.RandomState(10)
    X = np.vstack([r.normal(0, 1, size=(200, 2)),
                   r.normal(30, 0.5, size=(30, 2))])
    ee = EllipticEnvelope(contamination=0.15, random_state=0).fit(X)
    assert np.linalg.norm(ee.location_) < 1.0     # not dragged toward 30
    plain_mean = X.mean(axis=0)
    assert np.linalg.norm(plain_mean) > 2.0       # the naive mean is dragged


def test_elliptic_envelope_matches_sklearn_location():
    X, _ = _inliers_with_outliers(seed=11)
    ours = EllipticEnvelope(random_state=0).fit(X)
    ref = skcov.EllipticEnvelope(random_state=0).fit(X)
    assert np.allclose(ours.location_, ref.location_, atol=0.5)


def test_all_detectors_share_the_sign_convention():
    """-1 = outlier, 1 = inlier; decision_function < 0 exactly when flagged."""
    X, _ = _inliers_with_outliers(seed=12)
    for est in [IsolationForest(random_state=0),
                OneClassSVM(nu=0.1, gamma=0.1),
                EllipticEnvelope(random_state=0),
                LocalOutlierFactor(novelty=True)]:
        est.fit(X)
        pred = est.predict(X)
        dec = est.decision_function(X)
        assert set(np.unique(pred)) <= {-1, 1}
        assert np.array_equal(pred == -1, dec < 0)


def test_fit_predict_available():
    X, _ = _inliers_with_outliers(seed=13)
    for est in [IsolationForest(random_state=0), LocalOutlierFactor(),
                OneClassSVM(nu=0.1, gamma=0.1), EllipticEnvelope(random_state=0)]:
        pred = est.fit_predict(X)
        assert pred.shape == (len(X),)
        assert set(np.unique(pred)) <= {-1, 1}
