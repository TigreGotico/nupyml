"""Solve an equality-constrained problem by a sequence of QPs (Kraft, 1988)."""
import numpy as np


def slsqp(f, grad, x0, eq_constraint, eq_jac, max_iter=100, tol=1e-8):
    """Solve an equality-constrained problem by a sequence of QPs (Kraft, 1988).

    Minimise ``f(x)`` subject to ``c(x) = 0``. SLSQP works like Newton's method for the
    KKT conditions: at each iterate it builds a QUADRATIC model of the Lagrangian and a
    LINEAR model of the constraints, and solves that equality-constrained QP for the
    step by its KKT linear system. A BFGS update keeps a positive-definite Hessian
    approximation so the QP is well posed without computing second derivatives. It is
    scipy's default for smooth constrained optimisation. ``eq_constraint(x)`` returns
    the constraint residual vector and ``eq_jac(x)`` its Jacobian.
    """
    x = np.asarray(x0, float).copy()
    n = len(x)
    B = np.eye(n)
    g = grad(x)
    for _ in range(max_iter):
        c = np.atleast_1d(eq_constraint(x))
        J = np.atleast_2d(eq_jac(x))
        m = len(c)
        # KKT system for min 0.5 d'B d + g'd  s.t.  J d = -c
        KKT = np.block([[B, J.T], [J, np.zeros((m, m))]])
        rhs = np.concatenate([-g, -c])
        sol = np.linalg.solve(KKT, rhs)
        d = sol[:n]
        if np.linalg.norm(d) < tol:
            break
        x_new = x + d
        g_new = grad(x_new)
        s, yv = d, g_new - g                              # BFGS on the Lagrangian curvature
        if s @ yv > 1e-10:
            Bs = B @ s
            B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(yv, yv) / (yv @ s)
        x, g = x_new, g_new
    return x


__all__ = ["slsqp"]
