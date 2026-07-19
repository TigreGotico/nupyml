"""M8: optimization v5 -- nonlinear CG, subgradient, pattern search,
Douglas-Rachford, Chambolle-Pock.

Nonlinear CG solves Rosenbrock; the subgradient method minimises a sum of absolute
values; pattern search finds a shifted sphere's optimum with no gradient; both
splitting methods recover the closed-form lasso (soft-threshold) solution.
"""
import numpy as np

from nupyml.optimize import (nonlinear_conjugate_gradient, subgradient_method,
                            pattern_search, douglas_rachford, chambolle_pock)


def _rosen(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def _rosen_grad(x):
    g = np.zeros_like(x); xm = x[1:-1]
    g[1:-1] = 200 * (xm - x[:-2] ** 2) - 400 * (x[2:] - xm ** 2) * xm - 2 * (1 - xm)
    g[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    g[-1] = 200 * (x[-1] - x[-2] ** 2)
    return g


def test_nonlinear_cg_solves_rosenbrock():
    for method in ("fletcher-reeves", "polak-ribiere"):
        x = nonlinear_conjugate_gradient(_rosen, _rosen_grad, np.array([-1.2, 1.0]),
                                         method=method, max_iter=2000)
        assert np.allclose(x, [1.0, 1.0], atol=1e-2)


def test_subgradient_minimises_nonsmooth():
    # |x-3| + |x+1| is flat-minimised on [-1, 3] with value 4
    f = lambda x: abs(x[0] - 3) + abs(x[0] + 1)
    sg = lambda x: np.array([np.sign(x[0] - 3) + np.sign(x[0] + 1)])
    x = subgradient_method(sg, np.array([10.0]), step="1/k", step_size=2.0,
                           max_iter=3000, f=f)
    assert abs(f(x) - 4.0) < 1e-2
    assert -1.01 <= x[0] <= 3.01


def test_pattern_search_finds_shifted_sphere():
    target = np.array([1.0, -2.0, 0.5])
    x = pattern_search(lambda x: np.sum((x - target) ** 2), np.zeros(3),
                       step=1.0, max_iter=5000)
    assert np.allclose(x, target, atol=1e-3)


def _lasso_setup():
    a = np.array([3.0, -0.3, 1.5, 0.05]); lam = 0.5
    expected = np.sign(a) * np.maximum(np.abs(a) - lam, 0)   # soft-threshold
    return a, lam, expected


def test_douglas_rachford_recovers_lasso():
    a, lam, expected = _lasso_setup()
    prox_f = lambda v, g: (v + g * a) / (1 + g)             # prox of 0.5||x-a||^2
    prox_g = lambda v, g: np.sign(v) * np.maximum(np.abs(v) - g * lam, 0)
    x = douglas_rachford(prox_f, prox_g, np.zeros(4), gamma=1.0, max_iter=2000)
    assert np.allclose(x, expected, atol=1e-2)


def test_chambolle_pock_recovers_lasso():
    a, lam, expected = _lasso_setup()
    K = lambda x: x; Kt = lambda y: y
    prox_fs = lambda y, s: np.clip(y, -lam, lam)            # prox of (lam||.||_1)*
    prox_g = lambda v, t: (v + t * a) / (1 + t)
    x = chambolle_pock(K, Kt, prox_fs, prox_g, np.zeros(4), sigma=0.9, tau=0.9,
                       L=1.0, max_iter=2000)
    assert np.allclose(x, expected, atol=1e-2)
