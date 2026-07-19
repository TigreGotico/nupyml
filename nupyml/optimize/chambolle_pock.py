"""Chambolle-Pock: a primal-dual splitting for ``min_x f(Kx) + g(x)``."""
import numpy as np


def chambolle_pock(K, Kt, prox_fs, prox_g, x0, sigma=None, tau=None, theta=1.0,
                   L=None, max_iter=300):
    """Solve a saddle-point problem by PRIMAL-DUAL splitting (Chambolle & Pock, 2011).

    Imaging problems have the shape ``min_x f(Kx) + g(x)`` -- a term acting through a
    linear operator ``K`` (a gradient, a blur) plus a term on ``x`` itself. Inverting
    ``K`` is expensive, so Chambolle-Pock rewrites the problem as a SADDLE point over a
    primal ``x`` and a dual ``y``, and takes cheap alternating steps: a dual proximal
    ascent (using ``prox`` of ``f*``, the convex conjugate), a primal proximal descent,
    and an over-relaxation that couples them. Each step needs only ``K``, its adjoint
    ``Kt``, and two proximal operators -- no matrix inverse -- yet it converges at the
    optimal rate for this class. ``prox_fs`` is the prox of the conjugate ``f*``.
    """
    x = np.array(x0, float)
    y = K(x) * 0.0
    if L is None:
        L = 1.0
    if sigma is None:
        sigma = 1.0 / L
    if tau is None:
        tau = 1.0 / L
    x_bar = x.copy()
    for _ in range(max_iter):
        y = prox_fs(y + sigma * K(x_bar), sigma)           # dual ascent
        x_new = prox_g(x - tau * Kt(y), tau)               # primal descent
        x_bar = x_new + theta * (x_new - x)                # over-relaxation
        x = x_new
    return x


__all__ = ["chambolle_pock"]
