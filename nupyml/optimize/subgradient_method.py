"""Subgradient method: minimise a non-differentiable convex function."""
import numpy as np


def subgradient_method(subgrad, x0, step="1/k", step_size=0.1, max_iter=1000,
                       f=None):
    """Minimise a CONVEX but non-smooth function using any subgradient (Shor, 1985).

    Gradient descent needs a gradient, but many convex objectives -- an L1 penalty, a
    hinge loss, a max of linear pieces -- have kinks where none exists. At a kink there
    is instead a SET of subgradients (any supporting slope), and moving opposite ANY of
    them still makes progress on average. The catch: the objective is not guaranteed to
    drop every step, so the method tracks the BEST point seen, and its step size must
    shrink (``1/k``) for convergence. Simple, general, and the theoretical backbone of
    non-smooth convex optimisation. ``subgrad(x)`` returns any subgradient; pass ``f``
    to track the best objective.
    """
    x = np.array(x0, float)
    best_x = x.copy()
    best_f = f(x) if f is not None else np.inf
    for k in range(1, max_iter + 1):
        g = np.asarray(subgrad(x), float)
        if step == "1/k":
            alpha = step_size / k                          # diminishing step
        elif step == "1/sqrt(k)":
            alpha = step_size / np.sqrt(k)
        else:
            alpha = step_size                              # constant step
        x = x - alpha * g
        if f is not None:
            fx = f(x)
            if fx < best_f:                                # non-monotone: keep the best
                best_f, best_x = fx, x.copy()
    return best_x if f is not None else x


__all__ = ["subgradient_method"]
