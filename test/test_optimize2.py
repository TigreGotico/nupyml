"""J9: numerical optimization v2 -- L-BFGS, Levenberg-Marquardt, Frank-Wolfe,
Powell, cross-entropy method.

L-BFGS must reach the Rosenbrock minimum; LM must recover a nonlinear curve fit;
Frank-Wolfe must stay on the probability simplex; Powell and CEM (both gradient-
free) must find a quadratic bowl's minimum.
"""
import numpy as np
import pytest

from nupyml.optimize import (lbfgs, levenberg_marquardt, frank_wolfe, powell,
                             cross_entropy_method)


def test_lbfgs_solves_rosenbrock():
    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def g(x):
        return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                         200 * (x[1] - x[0] ** 2)])

    x = lbfgs(f, g, np.array([-1.2, 1.0]), max_iter=200)
    assert np.allclose(x, [1.0, 1.0], atol=1e-3)


def test_levenberg_marquardt_fits_exponential():
    rng = np.random.RandomState(0)
    xs = np.linspace(0, 2, 30)
    ys = 2.5 * np.exp(0.8 * xs) + 0.01 * rng.randn(30)

    def res(p):
        return p[0] * np.exp(p[1] * xs) - ys

    def jac(p):
        return np.column_stack([np.exp(p[1] * xs), p[0] * xs * np.exp(p[1] * xs)])

    p = levenberg_marquardt(res, jac, np.array([1.0, 0.1]))
    assert np.allclose(p, [2.5, 0.8], atol=0.05)


def test_frank_wolfe_stays_on_simplex():
    c = np.array([0.6, 0.9, -0.2])

    def g(x):
        return 2 * (x - c)

    def oracle(grad):                                     # a vertex of the simplex
        s = np.zeros(3); s[np.argmin(grad)] = 1.0; return s

    x = frank_wolfe(g, oracle, np.array([1 / 3, 1 / 3, 1 / 3]), max_iter=500)
    assert abs(x.sum() - 1.0) < 1e-6 and np.all(x >= -1e-9)   # feasible throughout
    # the projection of c onto the simplex zeroes the negative coordinate
    assert x[2] == pytest.approx(0.0, abs=1e-3)


def test_powell_minimises_without_gradient():
    def bowl(x):
        return (x[0] - 3) ** 2 + (x[1] + 1) ** 2 + 0.5 * (x[0] - 3) * (x[1] + 1)
    x = powell(bowl, np.array([0.0, 0.0]))
    assert np.allclose(x, [3.0, -1.0], atol=1e-3)


def test_cross_entropy_method_finds_minimum():
    def bowl(x):
        return (x[0] - 3) ** 2 + (x[1] + 1) ** 2
    x = cross_entropy_method(bowl, np.array([0.0, 0.0]), sigma0=3.0,
                             random_state=0)
    assert np.allclose(x, [3.0, -1.0], atol=0.1)
