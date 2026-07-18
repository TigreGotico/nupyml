"""L8: numerical optimization v4 -- SLSQP, augmented Lagrangian, SVRG/SAGA, mirror
descent, proximal Newton.

SLSQP and the augmented Lagrangian solve equality-constrained problems to the known
optimum; SVRG and SAGA reach the least-squares solution of a finite sum; mirror
descent stays on the simplex and finds the target; proximal Newton recovers a sparse
solution.
"""
import numpy as np
import pytest

from nupyml.optimize import (slsqp, augmented_lagrangian, svrg, saga,
                             mirror_descent, proximal_newton)


def test_slsqp_equality_constrained():
    f = lambda x: x @ x
    g = lambda x: 2 * x
    x = slsqp(f, g, np.array([2.0, -1.0]),
              eq_constraint=lambda x: np.array([x[0] + x[1] - 1]),
              eq_jac=lambda x: np.array([[1.0, 1.0]]))
    assert np.allclose(x, [0.5, 0.5], atol=1e-4)


def test_augmented_lagrangian_equality_constrained():
    f = lambda x: x @ x
    g = lambda x: 2 * x
    x = augmented_lagrangian(f, g, np.array([0.0, 0.0]),
                             constraint=lambda x: np.array([x[0] + 2 * x[1] - 4]),
                             constraint_jac=lambda x: np.array([[1.0, 2.0]]),
                             max_inner=300)
    assert np.allclose(x, [0.8, 1.6], atol=1e-2)          # min ||x||^2 on the line


def test_svrg_and_saga_reach_least_squares():
    rng = np.random.RandomState(0)
    A = rng.randn(50, 5); w = rng.randn(5); b = A @ w
    grad_i = lambda x, i: 2 * A[i] * (A[i] @ x - b[i])
    ols = np.linalg.lstsq(A, b, rcond=None)[0]
    xs = svrg(grad_i, np.zeros(5), 50, lr=0.02, epochs=60, random_state=0)
    xa = saga(grad_i, np.zeros(5), 50, lr=0.02, epochs=60, random_state=0)
    assert np.linalg.norm(xs - ols) < 1e-3
    assert np.linalg.norm(xa - ols) < 1e-3


def test_mirror_descent_stays_on_simplex():
    target = np.array([0.6, 0.3, 0.1])
    x = mirror_descent(lambda x: 2 * (x - target), np.ones(3) / 3, eta=0.5,
                       max_iter=1000)
    assert abs(x.sum() - 1.0) < 1e-6 and np.all(x >= 0)   # feasible on the simplex
    assert np.allclose(x, target, atol=0.02)


def test_proximal_newton_recovers_sparse():
    rng = np.random.RandomState(0)
    A = rng.randn(80, 10)
    w_true = np.array([3., 0, 0, -2, 0, 0, 0, 1, 0, 0])
    b = A @ w_true
    sg = lambda w: A.T @ (A @ w - b) / len(b)
    sh = lambda w: A.T @ A / len(b)
    w = proximal_newton(sg, sh, np.zeros(10), l1=0.05)
    assert (np.abs(w) < 1e-6).sum() >= 6                  # the true zeros
    assert w[0] > 1 and w[3] < -1
