"""F8: unbalanced OT, sliced Wasserstein, Gromov-Wasserstein, and OT mapping.

Sliced Wasserstein must behave like a distance (grow with separation, ~0 for
identical); unbalanced OT must leave mass unmatched rather than chase an outlier;
Gromov-Wasserstein must preserve the RELATIONAL distance structure across domains
(the property that survives its isometry ambiguity); OT mapping must push source
points onto the target distribution.
"""
import numpy as np
import pytest

from scipy.spatial.distance import cdist

from nupyml.optimal_transport import (sinkhorn_unbalanced, sliced_wasserstein,
                                      gromov_wasserstein, ot_mapping)


# --- sliced Wasserstein ---------------------------------------------------

def test_sliced_wasserstein_grows_with_separation():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    Y_close = rng.randn(200, 4)
    Y_far = rng.randn(200, 4) + 3.0
    d_close = sliced_wasserstein(X, Y_close, random_state=0)
    d_far = sliced_wasserstein(X, Y_far, random_state=0)
    assert d_far > d_close
    assert d_far > 2.5                               # ~ the shift of 3


def test_sliced_wasserstein_near_zero_for_same_distribution():
    rng = np.random.RandomState(1)
    X = rng.randn(500, 3)
    assert sliced_wasserstein(X, X.copy(), random_state=0) < 0.05


# --- unbalanced OT --------------------------------------------------------

def test_unbalanced_ot_recovers_balanced_when_penalty_is_huge():
    """With a very large marginal penalty, unbalanced OT must respect the
    marginals (it reduces to balanced OT)."""
    rng = np.random.RandomState(0)
    a = np.full(5, 0.2); b = np.full(4, 0.25)
    C = cdist(rng.randn(5, 2), rng.randn(4, 2))
    T = sinkhorn_unbalanced(a, b, C, reg=0.05, reg_marginal=50.0)
    assert np.allclose(T.sum(axis=1), a, atol=0.05)
    assert np.allclose(T.sum(axis=0), b, atol=0.05)


def test_unbalanced_ot_leaves_outlier_mass_unmatched():
    """A soft penalty lets the plan ignore an outlier source bin instead of
    forcing its mass across a huge distance."""
    a = np.array([0.33, 0.33, 0.34])
    b = np.array([0.5, 0.5])
    # bins 0,1 are near b; bin 2 is an outlier, far from everything
    C = np.array([[0.1, 0.2], [0.2, 0.1], [50.0, 50.0]])
    T = sinkhorn_unbalanced(a, b, C, reg=0.1, reg_marginal=1.0)
    # the outlier row transports far less than its nominal mass
    assert T[2].sum() < 0.5 * a[2]
    assert T[0].sum() > T[2].sum()


# --- Gromov-Wasserstein ---------------------------------------------------

def test_gromov_wasserstein_preserves_relational_structure():
    """GW is defined only up to isometry, so we don't test the exact labels -- we
    test that the coupling it finds preserves pairwise distances: matched pairs in
    domain 1 have near-equal distances in domain 2."""
    rng = np.random.RandomState(0)
    pts = rng.randn(8, 3)                            # asymmetric point cloud
    D1 = cdist(pts, pts)
    perm = rng.permutation(8)
    D2 = D1[perm][:, perm]                           # same structure, relabelled
    T = gromov_wasserstein(D1, D2, reg=0.02, n_iter=300)
    match = T.argmax(axis=1)
    # reconstruct domain-2 distances via the matching and compare to domain 1
    D2_matched = D2[np.ix_(match, match)]
    err = np.abs(D1 - D2_matched).mean()
    assert err < 0.5 * D1.mean()                     # structure broadly preserved


def test_gromov_wasserstein_coupling_is_valid():
    rng = np.random.RandomState(2)
    D1 = cdist(rng.randn(6, 2), rng.randn(6, 2))
    D2 = cdist(rng.randn(6, 2), rng.randn(6, 2))
    T = gromov_wasserstein(D1, D2, reg=0.05)
    assert np.allclose(T.sum(), 1.0, atol=1e-2)      # a valid coupling
    assert (T >= 0).all()


# --- OT mapping -----------------------------------------------------------

def test_ot_mapping_pushes_source_onto_target():
    rng = np.random.RandomState(0)
    Xs = rng.randn(150, 3)
    Xt = rng.randn(150, 3) + 5.0                      # target is shifted
    mapped = ot_mapping(Xs, Xt, reg=0.5)
    assert mapped.shape == Xs.shape
    # the mapped source now sits on the target distribution
    assert np.allclose(mapped.mean(axis=0), Xt.mean(axis=0), atol=0.7)
    # and it moved away from the original source
    assert np.linalg.norm(mapped.mean(axis=0) - Xs.mean(axis=0)) > 3.0
