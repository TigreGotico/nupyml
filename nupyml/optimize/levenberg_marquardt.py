"""Interpolate between gradient descent and Gauss-Newton (Levenberg 1944; ...)."""
import numpy as np


def levenberg_marquardt(residuals, jacobian, x0, max_iter=100, tol=1e-8,
                        lam0=1e-3):
    """Interpolate between gradient descent and Gauss-Newton (Levenberg 1944; ...).

    For a sum of squared residuals, Gauss-Newton is fast near the solution but can
    diverge far from it; gradient descent is safe but slow. Levenberg-Marquardt
    blends them with a damping ``lambda``: solve ``(JᵀJ + lambda·I) dx = -Jᵀr``.
    Large ``lambda`` -> a small, safe gradient step; small ``lambda`` -> the fast
    Gauss-Newton step. After a successful step ``lambda`` shrinks (trust the
    quadratic model more); after a failed one it grows. This adaptivity is why LM is
    the default for nonlinear least squares and curve fitting.
    """
    x = np.asarray(x0, float).copy()
    lam = lam0
    r = np.asarray(residuals(x), float)
    cost = r @ r
    for _ in range(max_iter):
        J = np.asarray(jacobian(x), float)
        JtJ = J.T @ J
        g = J.T @ r
        while True:
            try:
                dx = np.linalg.solve(JtJ + lam * np.eye(len(x)), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            r_new = np.asarray(residuals(x + dx), float)
            if r_new @ r_new < cost:                     # step accepted
                x = x + dx; r = r_new; cost = r @ r
                lam = max(lam / 10, 1e-12)               # trust the model more
                break
            lam *= 10                                    # step rejected: damp harder
            if lam > 1e12:
                return x
        if np.linalg.norm(g) < tol:
            break
    return x


__all__ = ["levenberg_marquardt"]
