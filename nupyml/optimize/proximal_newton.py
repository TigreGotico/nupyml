"""Second-order curvature on the smooth part of an L1 problem (Lee, 2014)."""
import numpy as np


def _soft(v, thr):
    return np.sign(v) * max(abs(v) - thr, 0.0)


def proximal_newton(smooth_grad, smooth_hess, x0, l1=0.1, max_iter=100, tol=1e-8,
                    inner_iter=200):
    """Second-order curvature on the smooth part of an L1 problem (Lee, 2014).

    Proximal GRADIENT (ISTA/FISTA) uses only the gradient of the smooth loss, so it
    inherits gradient descent's slow convergence on ill-conditioned problems. Proximal
    NEWTON scales the step by the smooth part's HESSIAN: at each iterate it minimises a
    second-order model of the loss plus the exact L1 term (a lasso subproblem, solved
    by coordinate descent), giving Newton-fast convergence to a sparse solution. Ideal
    when the Hessian is available and the problem is badly conditioned. ``smooth_grad``
    and ``smooth_hess`` are of the SMOOTH loss only; the L1 term is handled internally.
    """
    x = np.asarray(x0, float).copy()
    n = len(x)
    for _ in range(max_iter):
        g = smooth_grad(x)
        H = np.atleast_2d(smooth_hess(x))
        # solve min 0.5 (z-x)'H(z-x) + g'(z-x) + l1||z||_1 by coordinate descent
        z = x.copy()
        Hd = np.diag(H)
        for _ in range(inner_iter):
            for j in range(n):
                # gradient of the quadratic model wrt z_j, excluding z_j's own term
                rj = g[j] + H[j] @ (z - x) - Hd[j] * (z[j] - x[j])
                z[j] = _soft(x[j] - rj / (Hd[j] + 1e-12), l1 / (Hd[j] + 1e-12))
        if np.linalg.norm(z - x) < tol:
            x = z; break
        x = z
    return x


__all__ = ["proximal_newton"]
