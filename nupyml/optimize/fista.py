"""FISTA: accelerated proximal gradient for ``min f(x) + g(x)`` (Beck-Teboulle)."""
import numpy as np


def fista(grad_f, prox, x0, L, n_iter=200):
    """FISTA: accelerated proximal gradient for ``min f(x) + g(x)`` (Beck-Teboulle).

    When the objective splits into a SMOOTH ``f`` (you have its gradient) and a
    non-smooth ``g`` (you have its proximal operator -- e.g. soft-thresholding for
    an L1 penalty), proximal gradient alternates a gradient step on ``f`` with a
    prox step on ``g``. FISTA adds Nesterov MOMENTUM via an extrapolated point,
    lifting the convergence rate from ``O(1/k)`` to ``O(1/k^2)`` at no extra cost
    per iteration. ``L`` is a Lipschitz constant of ``grad_f`` (the step is 1/L).
    """
    x = np.array(x0, float)
    y = x.copy()
    t = 1.0
    for _ in range(n_iter):
        x_new = prox(y - grad_f(y) / L, 1.0 / L)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)  # momentum extrapolation
        x, t = x_new, t_new
    return x


__all__ = ["fista"]
