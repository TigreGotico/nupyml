"""Optimize over a constraint set WITHOUT ever projecting (Frank & Wolfe, 1956)."""
import numpy as np


def frank_wolfe(grad, linear_oracle, x0, max_iter=200, tol=1e-8):
    """Optimize over a constraint set WITHOUT ever projecting (Frank & Wolfe, 1956).

    Projected gradient needs a projection onto the feasible set, which can be as
    hard as the original problem. Frank-Wolfe (conditional gradient) instead calls a
    LINEAR oracle: minimise the current linear approximation over the set -- often a
    cheap closed form (a vertex of a simplex, the top singular vector of a nuclear-
    norm ball). It then steps toward that vertex by ``2/(t+2)``. Every iterate stays
    feasible as a convex combination of vertices, which also makes the solution
    SPARSE. ``linear_oracle(g)`` returns argmin over the set of ``g·s``.
    """
    x = np.asarray(x0, float).copy()
    for t in range(max_iter):
        g = grad(x)
        s = np.asarray(linear_oracle(g), float)          # linear minimisation oracle
        gap = g @ (x - s)                                # Frank-Wolfe duality gap
        if gap < tol:
            break
        gamma = 2.0 / (t + 2.0)                          # step toward the vertex
        x = x + gamma * (s - x)
    return x


__all__ = ["frank_wolfe"]
