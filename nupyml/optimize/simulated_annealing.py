"""Simulated annealing: escape local minima by accepting UPHILL moves early."""
import numpy as np


def simulated_annealing(f, x0, bounds=None, T0=1.0, cooling=0.995,
                        max_iter=5000, random_state=None):
    """Simulated annealing: escape local minima by accepting UPHILL moves early.

    Propose a random nearby point; always accept it if it is better, and accept it
    even if WORSE with probability ``exp(-Delta/T)``. The temperature ``T`` starts
    high (almost any move accepted -- broad exploration) and cools geometrically
    (only improving moves survive -- local refinement). The occasional uphill move
    is what lets it climb out of a local basin, unlike greedy descent. Inspired by
    annealing in metallurgy.
    """
    from ..utils import check_random_state
    rng = check_random_state(random_state)
    x = np.array(x0, float)
    fx = f(x)
    best, fbest = x.copy(), fx
    T = T0
    for _ in range(max_iter):
        cand = x + rng.normal(0, T + 0.05, size=x.shape)
        if bounds is not None:
            cand = np.clip(cand, bounds[0], bounds[1])
        fc = f(cand)
        if fc < fx or rng.rand() < np.exp(-(fc - fx) / max(T, 1e-12)):
            x, fx = cand, fc                         # accept (maybe uphill)
            if fc < fbest:
                best, fbest = cand.copy(), fc
        T *= cooling                                 # cool down
    return best, float(fbest)


__all__ = ["simulated_annealing"]
