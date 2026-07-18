"""K8: numerical optimization v3 -- trust-region Newton-CG, interior-point QP,
SPSA, OWL-QN, basin-hopping.

TR-Newton-CG reaches the Rosenbrock optimum; interior-point solves a nonnegativity-
constrained QP; SPSA minimises a noisy quadratic from two evals per step; OWL-QN
recovers a SPARSE solution (exact zeros); basin-hopping escapes a local minimum.
"""
import numpy as np
import pytest

from nupyml.optimize import (trust_region_newton_cg, interior_point_qp, spsa,
                             owlqn, basin_hopping, nelder_mead)


def test_trust_region_newton_cg_solves_rosenbrock():
    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def g(x):
        return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                         200 * (x[1] - x[0] ** 2)])

    def hessp(x, v):
        H = np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]],
                      [-400 * x[0], 200.0]])
        return H @ v

    x = trust_region_newton_cg(f, g, hessp, np.array([-1.2, 1.0]))
    assert np.allclose(x, [1.0, 1.0], atol=1e-3)


def test_interior_point_qp_respects_constraints():
    # min 0.5||x||^2 - b.x  s.t.  x >= 0  ->  x = max(b, 0)
    b = np.array([1.0, -0.5, 2.0])
    P = np.eye(3); q = -b
    G = -np.eye(3); h = np.zeros(3)                      # -x <= 0  ==  x >= 0
    x = interior_point_qp(P, q, G, h)
    assert np.allclose(x, np.maximum(b, 0), atol=1e-2)
    assert np.all(x >= -1e-6)                           # feasible


def test_spsa_minimises_noisy_quadratic():
    rng = np.random.RandomState(0)

    def noisy(x):
        return np.sum((x - 1.0) ** 2) + 0.01 * rng.randn()

    x = spsa(noisy, np.array([5.0, -3.0, 2.0]), a=0.3, c=0.1, max_iter=2000,
             random_state=0)
    assert np.allclose(x, [1.0, 1.0, 1.0], atol=0.2)


def test_owlqn_recovers_sparse_solution():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    w_true = np.array([3., 0, 0, -2, 0, 0, 0, 1, 0, 0])
    y = X @ w_true

    def loss(w):
        return 0.5 * np.mean((X @ w - y) ** 2)

    def gloss(w):
        return X.T @ (X @ w - y) / len(y)

    w = owlqn(loss, gloss, np.zeros(10), l1=0.05, max_iter=300)
    assert np.sum(np.abs(w) < 1e-6) >= 6                 # the 7 true zeros
    # the nonzero weights point the right way
    assert w[0] > 1 and w[3] < -1 and w[7] > 0.3


def test_basin_hopping_escapes_local_minimum():
    def multi(x):
        return float(np.sum(x ** 2) + 10 * np.sum(np.sin(3 * x) ** 2))

    single = nelder_mead(multi, np.array([4.0]), max_iter=200)[0]
    hopped = basin_hopping(multi, np.array([4.0]), step_size=2.0, n_iter=60,
                           random_state=0)
    assert multi(hopped) < multi(single)                # found a deeper basin
    assert multi(hopped) < 1.0                           # near the global minimum
