"""I11: anomaly / OOD v2 -- extended isolation forest, isolation kernel, Deep
SVDD, energy score.

The detectors are held to ranking a held-out set of outliers above inliers (AUC);
the isolation kernel to being sharper within a dense cluster than across to an
outlier; the energy score to being lower for a confident (in-distribution) logit
vector than a flat (OOD) one.
"""
import numpy as np
import pytest

from nupyml.anomaly import (ExtendedIsolationForest, IsolationKernel, DeepSVDD,
                            energy_score)
from nupyml.metrics import roc_auc_score


def _inliers_outliers(rng):
    Xin = rng.randn(200, 2)
    Xout = rng.uniform(-8, 8, (20, 2))
    X = np.vstack([Xin, Xout])
    y = np.r_[np.zeros(200), np.ones(20)]
    return Xin, X, y


def test_extended_isolation_forest_ranks_outliers():
    rng = np.random.RandomState(0)
    Xin, X, y = _inliers_outliers(rng)
    eif = ExtendedIsolationForest(n_estimators=100, random_state=0).fit(Xin)
    assert roc_auc_score(y, eif.decision_function(X)) > 0.9


def test_isolation_kernel_is_data_dependent():
    rng = np.random.RandomState(0)
    Xin, _, _ = _inliers_outliers(rng)
    Xout = rng.uniform(-8, 8, (5, 2))
    ik = IsolationKernel(n_estimators=200, psi=8, random_state=0).fit(Xin)
    sim_in = ik.similarity(Xin[:5], Xin[5:10]).mean()
    sim_out = ik.similarity(Xin[:5], Xout).mean()
    # points inside the dense cluster are more similar to each other than to an outlier
    assert sim_in > sim_out
    # the feature map's self-similarity is 1 (unit-norm binary embedding)
    self_sim = np.diag(ik.similarity(Xin[:5]))
    assert np.allclose(self_sim, 1.0, atol=1e-6)


def test_deep_svdd_packs_normal_data_and_ranks_outliers():
    rng = np.random.RandomState(0)
    Xin, X, y = _inliers_outliers(rng)
    dv = DeepSVDD(hidden=8, epochs=60, random_state=0).fit(Xin)
    assert roc_auc_score(y, dv.decision_function(X)) > 0.85
    # normal points sit closer to the learned centre than outliers do
    assert dv.decision_function(Xin).mean() < dv.decision_function(X[200:]).mean()


def test_energy_score_low_for_confident_high_for_flat():
    confident = np.array([[10., 0, 0], [0, 9, 0]])
    flat = np.array([[0.1, 0.0, 0.1], [0.0, 0.0, 0.0]])
    e_conf = energy_score(confident)
    e_flat = energy_score(flat)
    # a peaked (in-distribution) logit vector has lower free energy than a flat one
    assert np.all(e_conf < e_flat)


def test_energy_score_matches_definition():
    logits = np.array([[1.0, 2.0, 3.0]])
    expected = -np.log(np.exp([1.0, 2.0, 3.0]).sum())
    assert energy_score(logits)[0] == pytest.approx(expected, abs=1e-6)
