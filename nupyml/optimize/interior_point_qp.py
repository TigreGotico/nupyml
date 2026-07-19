"""Solve a constrained QP by walking down the MIDDLE of the feasible set"""
import numpy as np


def interior_point_qp(P, q, G=None, h=None, max_iter=100, tol=1e-8):
    """Solve a constrained QP by walking down the MIDDLE of the feasible set
    (Karmarkar lineage).

    Minimise ``0.5 xᵀP x + qᵀx`` subject to ``G x <= h``. The active-set approach
    guesses which constraints bind; interior-point methods avoid the guess entirely.
    They add a LOG-BARRIER that pushes the iterate away from every constraint
    boundary, then relax the barrier toward zero while taking Newton steps on the
    perturbed optimality (KKT) conditions -- so the iterate glides through the strict
    interior and homes in on the solution, active constraints and all, in a
    predictable number of steps. Returns the optimal ``x``.
    """
    P = np.asarray(P, float); q = np.asarray(q, float)
    n = len(q)
    if G is None:
        return np.linalg.solve(P, -q)                    # unconstrained
    G = np.asarray(G, float); h = np.asarray(h, float)
    m = len(h)
    x = np.zeros(n)
    s = np.ones(m)                                       # slacks: h - Gx = s > 0
    lam = np.ones(m)                                     # multipliers
    # ensure a strictly feasible start
    while np.any(h - G @ x <= 0):
        x -= 0.1 * G.sum(axis=0)
    s = h - G @ x
    for _ in range(max_iter):
        mu = (s @ lam) / m
        if mu < tol:
            break
        t = 0.1 * mu                                     # barrier parameter
        # residuals of the perturbed KKT system
        r_dual = P @ x + q + G.T @ lam
        r_cent = s * lam - t
        Sinv = 1.0 / s
        # Schur complement on x
        H = P + G.T @ np.diag(lam * Sinv) @ G
        rhs = -(r_dual + G.T @ (Sinv * (r_cent - lam * (h - G @ x - s))))
        dx = np.linalg.solve(H + 1e-9 * np.eye(n), rhs)
        ds = -(G @ dx)                                   # from h - G x = s
        dlam = -Sinv * (r_cent + lam * ds)
        # step length keeping s, lam > 0
        step = 1.0
        for arr, darr in ((s, ds), (lam, dlam)):
            neg = darr < 0
            if np.any(neg):
                step = min(step, 0.99 * np.min(-arr[neg] / darr[neg]))
        x += step * dx; s += step * ds; lam += step * dlam
    return x


__all__ = ["interior_point_qp"]
