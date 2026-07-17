"""G3: mathematical optimization solvers.

Each solver is checked against a problem with a known answer: an LP against its
vertex optimum, the QP against a box-clipped unconstrained solution, CG against a
direct solve, the proximal lasso solvers against the true sparse support, and the
derivative-free methods against Rosenbrock / a shifted sphere.
"""
import numpy as np
import pytest

from nupyml.optimize import (linprog_simplex, quadratic_program,
                            conjugate_gradient, fista, soft_threshold,
                            lasso_fista, admm_lasso, nelder_mead,
                            simulated_annealing, particle_swarm)


def test_linprog_finds_the_vertex_optimum():
    # max x+y  (min -x-y)  s.t. x+2y<=14, -3x+y<=0, x-y<=2, x,y>=0  ->  (6,4)
    x, val = linprog_simplex([-1, -1], [[1, 2], [-3, 1], [1, -1]], [14, 0, 2])
    assert np.allclose(x, [6, 4], atol=1e-6)
    assert val == pytest.approx(-10.0)
    # the solution is feasible
    assert np.all(np.array([[1, 2], [-3, 1], [1, -1]]) @ x <= np.array([14, 0, 2]) + 1e-6)


def test_linprog_detects_unboundedness():
    with pytest.raises(ValueError):
        linprog_simplex([-1.0], [[-1.0]], [0.0])     # min -x s.t. -x<=0 -> unbounded


def test_quadratic_program_respects_constraints():
    # min 0.5||x||^2 - [1,2]x ; unconstrained optimum is (1,2), box-capped at 0.5
    P = np.eye(2); q = [-1.0, -2.0]
    x = quadratic_program(P, q, G=[[1, 0], [0, 1]], h=[0.5, 0.5])
    assert np.allclose(x, [0.5, 0.5], atol=1e-6)
    # an unconstrained QP hits the analytic solution
    x2 = quadratic_program(P, q)
    assert np.allclose(x2, [1.0, 2.0], atol=1e-8)


def test_conjugate_gradient_solves_spd_system():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x = conjugate_gradient(A, b)
    assert np.allclose(A @ x, b, atol=1e-8)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8)


def test_soft_threshold_shrinks_and_clamps():
    assert soft_threshold(np.array([5.0, -0.3, 0.8]), 1.0).tolist() == [4.0, 0.0, 0.0]


@pytest.mark.parametrize("solver", [lasso_fista, admm_lasso],
                         ids=["fista", "admm"])
def test_proximal_lasso_recovers_sparse_support(solver):
    rng = np.random.RandomState(0)
    X = rng.randn(100, 20)
    w_true = np.zeros(20); w_true[[1, 5, 10]] = [3.0, -2.0, 1.0]
    y = X @ w_true + 0.1 * rng.randn(100)
    w = solver(X, y, alpha=5.0)
    support = set(np.where(np.abs(w) > 0.1)[0])
    assert support == {1, 5, 10}                     # exact support recovery


def test_fista_matches_a_smooth_quadratic_minimum():
    # min 0.5||x - t||^2 with no penalty (prox = identity) -> x == t
    t = np.array([3.0, -1.0, 2.0])
    x = fista(grad_f=lambda x: x - t, prox=lambda x, s: x, x0=np.zeros(3),
              L=1.0, n_iter=200)
    assert np.allclose(x, t, atol=1e-3)


def test_nelder_mead_minimises_rosenbrock():
    rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    x, f = nelder_mead(rosen, [-1.0, 1.0], max_iter=2000)
    assert np.allclose(x, [1.0, 1.0], atol=1e-2)
    assert f < 1e-4


def test_simulated_annealing_finds_shifted_sphere_minimum():
    f = lambda x: np.sum((x - np.array([2.0, -3.0])) ** 2)
    x, val = simulated_annealing(f, [0.0, 0.0], bounds=(-5, 5), random_state=0)
    assert val < 0.05
    assert np.allclose(x, [2.0, -3.0], atol=0.3)


def test_particle_swarm_escapes_to_global_minimum():
    # a multi-modal function where greedy descent would get stuck
    f = lambda x: np.sum(x ** 2) - 5 * np.cos(2 * x[0]) - 5 * np.cos(2 * x[1])
    x, val = particle_swarm(f, ([-5, -5], [5, 5]), random_state=0)
    assert np.allclose(x, [0.0, 0.0], atol=0.2)      # global min at the origin
