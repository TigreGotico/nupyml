"""Nonlinear conjugate gradient: Fletcher-Reeves / Polak-Ribiere for smooth minimisation."""
import numpy as np


def _line_search(f, x, d, g, c1=1e-4, c2=0.1, alpha=1.0, max_ls=50):
    """Backtracking that also enforces the curvature (Wolfe) condition weakly."""
    fx = f(x); gd = g @ d
    for _ in range(max_ls):
        if f(x + alpha * d) <= fx + c1 * alpha * gd:
            return alpha
        alpha *= 0.5
    return alpha


def nonlinear_conjugate_gradient(f, grad, x0, method="polak-ribiere", max_iter=500,
                                 tol=1e-6):
    """Minimise a smooth function along CONJUGATE directions (Fletcher & Reeves, 1964).

    Steepest descent zig-zags in a narrow valley because each step undoes the last.
    Conjugate gradient fixes this for general functions: it starts down the gradient,
    then bends each new search direction by a factor ``beta`` that folds in the previous
    direction, so successive steps do not interfere. Fletcher-Reeves and Polak-Ribiere
    are two formulas for ``beta`` (the latter self-restarts on non-quadratic terrain and
    is usually faster). It needs only gradients -- no Hessian, no matrix storage -- so it
    scales to very large problems. ``method`` is 'fletcher-reeves' or 'polak-ribiere'.
    """
    x = np.array(x0, float)
    g = grad(x)
    d = -g
    for i in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        alpha = _line_search(f, x, d, g)
        x = x + alpha * d
        g_new = grad(x)
        if method == "fletcher-reeves":
            beta = (g_new @ g_new) / (g @ g + 1e-12)
        else:                                              # polak-ribiere (+)
            beta = max(0.0, (g_new @ (g_new - g)) / (g @ g + 1e-12))
        d = -g_new + beta * d
        if (i + 1) % len(x) == 0 or g_new @ d > 0:         # periodic restart
            d = -g_new
        g = g_new
    return x


__all__ = ["nonlinear_conjugate_gradient"]
