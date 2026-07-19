"""Gradient descent in the RIGHT geometry -- here the simplex (Nemirovski, 1983)."""
import numpy as np


def mirror_descent(grad, x0, eta=0.1, max_iter=500, tol=1e-9):
    """Gradient descent in the RIGHT geometry -- here the simplex (Nemirovski, 1983).

    Ordinary gradient descent assumes a Euclidean space, which is wrong when the
    variable lives on the probability SIMPLEX (weights that must be non-negative and
    sum to one). Mirror descent replaces the Euclidean step with one measured by a
    "mirror map" -- for the simplex, the negative entropy -- yielding the EXPONENTIATED
    GRADIENT update ``x *= exp(-eta·grad); x /= sum(x)``. It stays on the simplex
    automatically and its convergence depends on the geometry's diameter, not the
    ambient dimension, which is why it excels in high-dimensional online learning.
    Starts from ``x0`` on the simplex.
    """
    x = np.asarray(x0, float).copy()
    x = x / x.sum()
    for _ in range(max_iter):
        g = grad(x)
        x_new = x * np.exp(-eta * g)                      # entropic mirror step
        x_new /= x_new.sum()
        if np.linalg.norm(x_new - x) < tol:
            x = x_new; break
        x = x_new
    return x


__all__ = ["mirror_descent"]
