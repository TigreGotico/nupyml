"""Nelder-Mead: minimise using only function VALUES, no gradient."""
import numpy as np


def nelder_mead(f, x0, step=0.5, max_iter=1000, tol=1e-8):
    """Nelder-Mead: minimise using only function VALUES, no gradient.

    Keeps a simplex of ``n+1`` points and reshapes it toward lower values by four
    moves -- reflect the worst point through the centroid, EXPAND if that is very
    good, CONTRACT if it is poor, and SHRINK the whole simplex if nothing helps.
    The go-to when the objective is noisy, non-differentiable, or a black box
    (hyperparameters, simulation outputs), though it has no convergence guarantee
    in high dimensions.
    """
    x0 = np.array(x0, float)
    n = len(x0)
    simplex = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    simplex = np.array(simplex)
    fvals = np.array([f(x) for x in simplex])
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        if np.std(fvals) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)         # exclude the worst
        worst = simplex[-1]
        refl = centroid + (centroid - worst)         # reflection
        fr = f(refl)
        if fr < fvals[0]:
            exp = centroid + 2 * (centroid - worst)  # expansion
            fe = f(exp)
            simplex[-1], fvals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = refl, fr
        else:
            cont = centroid + 0.5 * (worst - centroid)   # contraction
            fc = f(cont)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = cont, fc
            else:                                    # shrink toward the best
                simplex = simplex[0] + 0.5 * (simplex - simplex[0])
                fvals = np.array([f(x) for x in simplex])
    i = int(np.argmin(fvals))
    return simplex[i], float(fvals[i])


__all__ = ["nelder_mead"]
