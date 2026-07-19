"""Newton-like steps from a LIMITED MEMORY of past gradients (Nocedal, 1980)."""
import numpy as np


def _wolfe_line_search(f, grad, x, d, g, c1, c2, max_ls=25):
    step, lo, hi = 1.0, 0.0, np.inf
    f0 = f(x); gd0 = g @ d
    for _ in range(max_ls):
        xn = x + step * d
        if f(xn) > f0 + c1 * step * gd0:                 # Armijo (sufficient decrease)
            hi = step; step = 0.5 * (lo + hi)
        elif grad(xn) @ d < c2 * gd0:                    # curvature condition
            lo = step; step = 2 * lo if hi == np.inf else 0.5 * (lo + hi)
        else:
            return step
    return step


def lbfgs(f, grad, x0, m=10, max_iter=200, tol=1e-6, c1=1e-4, c2=0.9):
    """Newton-like steps from a LIMITED MEMORY of past gradients (Nocedal, 1980).

    Newton's method needs the inverse Hessian -- ``O(n^2)`` storage and ``O(n^3)``
    to invert, impossible for large ``n``. L-BFGS never forms it: it reconstructs
    the Hessian's ACTION on the gradient from just the last ``m`` steps and gradient
    changes (the two-loop recursion), giving curvature-aware steps with ``O(mn)``
    memory. It is the default batch optimizer for smooth problems (logistic
    regression, CRFs, many MLE fits) precisely because it scales. A Wolfe line
    search keeps each step stable.
    """
    x = np.asarray(x0, float).copy()
    g = grad(x)
    s_list, y_list, rho_list = [], [], []
    for it in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        q = g.copy()                                     # two-loop recursion
        alphas = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * (s @ q); alphas.append(a); q = q - a * y
        if y_list:
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        else:
            gamma = 1.0
        z = gamma * q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            beta = rho * (y @ z); z = z + (a - beta) * s
        d = -z                                           # search direction
        step = _wolfe_line_search(f, grad, x, d, g, c1, c2)
        x_new = x + step * d
        g_new = grad(x_new)
        s, y = x_new - x, g_new - g
        if s @ y > 1e-10:                                # keep positive curvature
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s); y_list.append(y); rho_list.append(1.0 / (s @ y))
        x, g = x_new, g_new
    return x


__all__ = ["lbfgs"]
