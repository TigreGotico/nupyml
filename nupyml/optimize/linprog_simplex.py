"""Minimise ``c·x`` subject to ``A_ub x <= b_ub`` and ``x >= 0`` (b_ub >= 0)."""
import numpy as np


def linprog_simplex(c, A_ub, b_ub, max_iter=1000):
    """Minimise ``c·x`` subject to ``A_ub x <= b_ub`` and ``x >= 0`` (b_ub >= 0).

    THE SIMPLEX IDEA
    ----------------
    The feasible region of an LP is a polytope, and the optimum (if finite) sits
    at a VERTEX. Simplex walks from vertex to adjacent vertex, each step along an
    edge that lowers the objective, until no improving edge remains. Mechanically
    it pivots a tableau: pick an entering variable with negative reduced cost
    (Bland's rule here, to avoid cycling), a leaving variable by the min-ratio
    test (the first constraint that binds), and pivot. Slack variables turn the
    ``<=`` rows into equalities, and ``x = 0`` is the starting vertex.

    Returns (x, optimal_value). Handles the standard ``<=``, non-negative form;
    equality constraints would need a phase-1 to find a starting vertex.
    """
    c = np.asarray(c, float)
    A = np.asarray(A_ub, float)
    b = np.asarray(b_ub, float)
    m, n = A.shape
    # tableau: [A | I | b] on top, [c | 0 | 0] as the objective row
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = c
    basis = list(range(n, n + m))                    # start with slacks basic
    for _ in range(max_iter):
        # Bland's rule: lowest-index column with negative reduced cost enters
        cols = np.where(T[-1, :-1] < -1e-9)[0]
        if len(cols) == 0:
            break                                    # optimal
        j = cols[0]
        ratios = np.where(T[:m, j] > 1e-9, T[:m, -1] / T[:m, j], np.inf)
        if np.all(~np.isfinite(ratios)):
            raise ValueError("LP is unbounded")
        i = int(np.argmin(ratios))
        T[i] /= T[i, j]                              # pivot
        for k in range(m + 1):
            if k != i:
                T[k] -= T[k, j] * T[i]
        basis[i] = j
    x = np.zeros(n + m)
    for i, bvar in enumerate(basis):
        x[bvar] = T[i, -1]
    return x[:n], float(c @ x[:n])


# --- quadratic programming -------------------------------------------------


__all__ = ["linprog_simplex"]
