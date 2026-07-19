"""Minimise ``0.5 x'Px + q'x`` s.t. ``Gx <= h`` by the ACTIVE-SET method."""
import numpy as np


def quadratic_program(P, q, G=None, h=None, max_iter=100):
    """Minimise ``0.5 x'Px + q'x`` s.t. ``Gx <= h`` by the ACTIVE-SET method.

    THE ACTIVE-SET IDEA
    -------------------
    At the solution some inequality constraints hold with EQUALITY (active) and
    the rest are slack. If you knew which, the problem is a simple equality-
    constrained QP with a closed-form KKT solution. You don't, so active-set
    guesses a working set and repairs it: solve the equality QP on the current
    set; if the step is blocked by an inactive constraint, ADD it; if a
    constraint's Lagrange multiplier goes negative (it is pulling the wrong way),
    DROP it. Each move strictly helps, so it converges in finitely many steps for
    a strictly convex ``P``.

    Requires ``P`` positive definite and a feasible start (``x=0`` when ``h>=0``).
    Returns the minimiser.
    """
    P = np.asarray(P, float)
    q = np.asarray(q, float)
    n = len(q)
    if G is None:
        return np.linalg.solve(P, -q)                # unconstrained
    G = np.asarray(G, float)
    h = np.asarray(h, float)
    x = np.zeros(n)
    if np.any(G @ x > h + 1e-9):
        raise ValueError("x=0 infeasible; provide a feasible start")
    working = [i for i in range(len(h)) if abs(G[i] @ x - h[i]) < 1e-9]
    for _ in range(max_iter):
        g = P @ x + q
        Gw = G[working] if working else np.zeros((0, n))
        # solve KKT for the step p that minimises the QP on the working set
        if len(working):
            KKT = np.block([[P, Gw.T], [Gw, np.zeros((len(working),) * 2)]])
            rhs = np.concatenate([-g, np.zeros(len(working))])
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            p, lam = sol[:n], sol[n:]
        else:
            p, lam = np.linalg.solve(P, -g), np.array([])
        if np.linalg.norm(p) < 1e-8:
            if len(lam) == 0 or np.all(lam >= -1e-9):
                return x                             # KKT satisfied -> optimum
            j = working[int(np.argmin(lam))]         # drop wrong-sign constraint
            working.remove(j)
        else:
            # largest step in [0,1] before an inactive constraint blocks
            alpha, block = 1.0, None
            for i in range(len(h)):
                if i in working:
                    continue
                denom = G[i] @ p
                if denom > 1e-12:
                    t = (h[i] - G[i] @ x) / denom
                    if t < alpha:
                        alpha, block = t, i
            x = x + alpha * p
            if block is not None:
                working.append(block)
    return x


# --- iterative linear solve ------------------------------------------------


__all__ = ["quadratic_program"]
