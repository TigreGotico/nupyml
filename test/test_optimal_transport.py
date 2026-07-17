"""Optimal transport, kernel two-sample distances, and advanced metric learning.

OT tests use cases with known answers (Wasserstein between shifted Gaussians, the
barycenter of two bumps); the metric learners are held to the kNN improvement
they promise on noise-dominated data.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification
from nupyml.model_selection import train_test_split
from nupyml.neighbors import KNeighborsClassifier
from nupyml.optimal_transport import (sinkhorn, wasserstein_distance, barycenter,
                                     ot_domain_adaptation, mmd, energy_distance)
from nupyml.metric_learning import ITML, LFDA, RCA


# --- optimal transport ----------------------------------------------------

def test_wasserstein_1d_matches_the_shift():
    """In 1D, W1 between two Gaussians differing only in mean equals the mean
    gap -- the closed-form sort-and-match answer."""
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(3, 1, 2000)
    assert wasserstein_distance(a, b, p=1) == pytest.approx(3.0, abs=0.2)


def test_wasserstein_is_zero_for_the_same_distribution():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    assert wasserstein_distance(a, b, p=1) < 0.15


def test_sinkhorn_plan_matches_the_marginals():
    """The transport plan must move exactly the prescribed mass from and to each
    point -- its row and column sums are the input marginals."""
    cost = np.abs(np.subtract.outer(np.arange(5), np.arange(5))).astype(float)
    a = np.array([0.2, 0.3, 0.5, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.5, 0.3, 0.2])
    plan, cost_val = sinkhorn(a, b, cost, reg=0.1)
    assert np.allclose(plan.sum(axis=1), a, atol=0.01)
    assert np.allclose(plan.sum(axis=0), b, atol=0.01)
    assert cost_val > 0


def test_sinkhorn_cost_falls_with_less_regularization():
    """Less entropic blur -> a sharper plan -> closer to the true (lower) optimal
    transport cost."""
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, (20, 2))
    y = rng.normal(2, 1, (20, 2))
    from scipy.spatial.distance import cdist
    cost = cdist(x, y) ** 2
    a = np.full(20, 1 / 20)
    _, c_blur = sinkhorn(a, a, cost, reg=1.0)
    _, c_sharp = sinkhorn(a, a, cost, reg=0.05)
    assert c_sharp < c_blur


def test_barycenter_of_two_bumps_lies_between_them():
    """The shape-aware average: the Wasserstein barycenter of two separated bumps
    is a single bump in the MIDDLE, not two half-bumps as a pointwise mean gives."""
    d1 = np.zeros(20); d1[3] = 1.0
    d2 = np.zeros(20); d2[15] = 1.0
    bary = barycenter([d1, d2], reg=0.5, n_iter=200)
    peak = np.argmax(bary)
    assert 7 <= peak <= 11                        # near the midpoint (9)


def test_ot_domain_adaptation_moves_source_onto_target():
    """Transporting the labelled source onto the target distribution shifts its
    mean to the target's, so a source-trained model can work on the target."""
    rng = np.random.RandomState(0)
    X_source = rng.normal(0, 1, (150, 2))
    y_source = (X_source[:, 0] > 0).astype(int)
    X_target = rng.normal(0, 1, (150, 2)) + np.array([2.0, 2.0])
    adapted = ot_domain_adaptation(X_source, y_source, X_target, reg=0.5)
    assert np.allclose(adapted.mean(axis=0), [2.0, 2.0], atol=0.3)


def test_mmd_distinguishes_distributions():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (200, 3))
    Y = rng.normal(0, 1, (200, 3))
    Z = rng.normal(3, 1, (200, 3))
    assert mmd(X, Z) > mmd(X, Y)
    assert mmd(X, Y) < 0.05                       # same distribution -> ~0


def test_energy_distance_distinguishes_distributions():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (200, 3))
    Y = rng.normal(0, 1, (200, 3))
    Z = rng.normal(3, 1, (200, 3))
    assert energy_distance(X, Z) > energy_distance(X, Y)
    assert energy_distance(X, Y) < 0.5


def test_mmd_and_energy_are_nonnegative():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 2))
    Y = rng.normal(size=(100, 2))
    assert mmd(X, Y) >= 0
    assert energy_distance(X, Y) >= 0


# --- advanced metric learning ---------------------------------------------

@pytest.fixture
def noise_dominated():
    """Two informative features drowned by six large-scale noise ones -- so
    Euclidean kNN struggles and a learned metric should help."""
    rng = np.random.RandomState(1)
    X_sig, y = make_classification(n_samples=400, n_features=2, n_informative=2,
                                   random_state=1)
    noise = rng.normal(0, 5, (400, 6))
    X = np.hstack([X_sig, noise])
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.mark.parametrize("learner_factory", [
    lambda: ITML(random_state=0),
    lambda: RCA(),
    lambda: LFDA(n_components=2),
])
def test_metric_learner_improves_knn(learner_factory, noise_dominated):
    Xtr, Xte, ytr, yte = noise_dominated
    raw = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).score(Xte, yte)
    ml = learner_factory().fit(Xtr, ytr)
    learned = KNeighborsClassifier(n_neighbors=5).fit(
        ml.transform(Xtr), ytr).score(ml.transform(Xte), yte)
    assert learned > raw


def test_itml_produces_a_valid_metric(noise_dominated):
    """ITML's LogDet objective keeps M positive semidefinite -- a real distance."""
    Xtr, _, ytr, _ = noise_dominated
    itml = ITML(random_state=0).fit(Xtr, ytr)
    eigvals = np.linalg.eigvalsh(itml.M_)
    assert np.all(eigvals > -1e-8)


def test_lfda_reduces_dimension(noise_dominated):
    Xtr, _, ytr, _ = noise_dominated
    lfda = LFDA(n_components=2).fit(Xtr, ytr)
    assert lfda.transform(Xtr).shape == (len(Xtr), 2)


def test_rca_whitens_within_class_variation():
    """RCA shrinks the directions where same-class points vary; after transform,
    within-class scatter should be smaller relative to between-class."""
    rng = np.random.RandomState(0)
    # class variation is large along axis 1 (irrelevant), separation along axis 0
    X = np.vstack([rng.normal([0, 0], [0.3, 5], (100, 2)),
                   rng.normal([4, 0], [0.3, 5], (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    rca = RCA().fit(X, y)
    Xt = rca.transform(X)
    # after whitening, the within-class spread on the noisy axis is tamed
    before_ratio = X[y == 0, 1].std() / (X[y == 0, 0].std() + 1e-9)
    after_within = np.std(Xt[y == 0], axis=0)
    assert after_within.max() / after_within.min() < before_ratio
