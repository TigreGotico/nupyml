"""Hop between BASINS to escape local minima (Wales & Doye, 1997)."""
import numpy as np
from .. utils import check_random_state


def basin_hopping(f, x0, step_size=1.0, n_iter=100, T=1.0, local=None,
                  random_state=None):
    """Hop between BASINS to escape local minima (Wales & Doye, 1997).

    Local optimisation only ever finds the nearest valley. Basin-hopping wraps it in
    a global search: from the current local minimum, take a random PERTURBATION,
    re-optimise to the bottom of whatever basin you land in, and accept the new
    minimum by a Metropolis rule (always if better, sometimes if worse, governed by
    ``T``). Because it explores the discrete set of local minima rather than the raw
    landscape, it is remarkably effective on rugged, many-minima problems (it was
    invented for molecular energy landscapes). ``local`` is a local minimiser
    ``x -> x_min`` (Nelder-Mead by default).
    """
    from .nelder_mead import nelder_mead
    rng = check_random_state(random_state)
    if local is None:
        local = lambda x: np.asarray(nelder_mead(f, x, max_iter=200)[0])
    x = local(np.asarray(x0, float))
    fx = f(x)
    best, best_f = x.copy(), fx
    for _ in range(n_iter):
        trial = local(x + rng.uniform(-step_size, step_size, size=len(x)))
        ft = f(trial)
        if ft < fx or rng.rand() < np.exp(-(ft - fx) / T):   # Metropolis accept
            x, fx = trial, ft
        if ft < best_f:
            best, best_f = trial.copy(), ft
    return best


__all__ = ["basin_hopping"]
