"""Generalised pattern search: derivative-free minimisation on a shrinking mesh."""
import numpy as np


def pattern_search(f, x0, step=1.0, shrink=0.5, expand=1.0, tol=1e-8,
                   max_iter=1000):
    """Minimise using only function values, on a MESH that adapts (Torczon, 1997).

    A robust derivative-free method with a convergence guarantee that Nelder-Mead
    lacks. It probes the objective at the current point PLUS and MINUS a step along each
    coordinate (the "pattern"). If any probe improves, it moves there and optionally
    grows the step (an ambitious stride); if none does, the current point is a mesh
    local optimum at this resolution, so it SHRINKS the step and looks again on the finer
    mesh. Because it only ever compares function values on a structured mesh, it tolerates
    noise and non-smoothness while still provably converging. ``step`` is the initial
    mesh size.
    """
    x = np.array(x0, float)
    n = len(x)
    fx = f(x)
    directions = np.vstack([np.eye(n), -np.eye(n)])        # +/- each coordinate
    while step > tol and max_iter > 0:
        improved = False
        for d in directions:
            trial = x + step * d
            ft = f(trial)
            max_iter -= 1
            if ft < fx:
                x, fx = trial, ft                          # poll success: move
                step *= (1 + expand) if expand else 1.0
                improved = True
                break
        if not improved:
            step *= shrink                                 # refine the mesh
    return x


__all__ = ["pattern_search"]
