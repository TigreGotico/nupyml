"""Continuous optimisation modelled on wolf-pack hunting (Mirjalili, 2014)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class GreyWolfOptimizer(BaseEstimator):
    """Continuous optimisation modelled on wolf-pack hunting (Mirjalili, 2014).

    A swarm metaheuristic with an unusually clean rule. The three best solutions are
    named ALPHA, BETA and DELTA -- the pack leaders -- and every other wolf updates
    its position by moving toward a blend of the three, with a coefficient that
    shrinks over time from EXPLORATION (wolves spread out and search) to
    EXPLOITATION (they converge on the prey). No gradients, few parameters, and it
    is competitive with PSO/DE on multimodal functions. Minimises ``fitness`` over a
    box.
    """

    def __init__(self, fitness, n_var, bounds, n_wolves=20, max_iter=200,
                 random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pos = rng.uniform(lo, hi, (self.n_wolves, self.n_var))
        for it in range(self.max_iter):
            fits = np.array([self.fitness(w) for w in pos])
            order = np.argsort(fits)
            alpha, beta, delta = pos[order[0]], pos[order[1]], pos[order[2]]
            a = 2 - 2 * it / self.max_iter                # explore -> exploit
            new = np.empty_like(pos)
            for i in range(self.n_wolves):
                moves = []
                for leader in (alpha, beta, delta):
                    r1, r2 = rng.rand(self.n_var), rng.rand(self.n_var)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - pos[i])
                    moves.append(leader - A * D)
                new[i] = np.clip(np.mean(moves, axis=0), lo, hi)
            pos = new
        fits = np.array([self.fitness(w) for w in pos])
        self.best_ = pos[fits.argmin()]
        self.best_fitness_ = fits.min()
        return self.best_


__all__ = ["GreyWolfOptimizer"]
