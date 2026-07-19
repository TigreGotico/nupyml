"""Solve ``A x = b`` for symmetric positive-definite ``A`` (Hestenes-Stiefel)."""
import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """Solve ``A x = b`` for symmetric positive-definite ``A`` (Hestenes-Stiefel).

    CG minimises the quadratic ``0.5 x'Ax - b'x`` (whose minimum solves the
    system) along a sequence of A-CONJUGATE directions, so each step's progress is
    never undone by the next -- unlike steepest descent, which zig-zags. It
    reaches the exact solution in at most ``n`` steps and, crucially, touches ``A``
    only through matrix-VECTOR products, so it scales to huge sparse systems where
    factorising ``A`` is impossible.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, float)
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    for _ in range(max_iter or len(b)):
        Ap = A @ p
        alpha = rs / (p @ Ap + 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p                    # conjugate direction update
        rs = rs_new
    return x


# --- proximal methods ------------------------------------------------------


__all__ = ["conjugate_gradient"]
