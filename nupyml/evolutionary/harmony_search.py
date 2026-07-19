"""Improvise solutions like a musician tuning a chord (Geem et al., 2001)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class HarmonySearch(BaseEstimator):
    """Improvise solutions like a musician tuning a chord (Geem et al., 2001).

    Harmony search draws its metaphor from a band searching for a pleasing harmony.
    It keeps a MEMORY of good solutions, and each new candidate is composed
    note-by-note (dimension by dimension): with high probability take that note from
    a remembered solution (and maybe nudge its PITCH slightly), otherwise play a
    random note. If the new harmony is better than the worst in memory, it replaces
    it. The mix of memory, small adjustments, and occasional randomness balances
    exploitation and exploration with very few parameters. Minimises ``fitness`` over
    a box.
    """

    def __init__(self, fitness, n_var, bounds, memory_size=30, hmcr=0.9, par=0.3,
                 bandwidth=0.05, iterations=2000, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.memory_size = memory_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.iterations = iterations
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        HM = rng.uniform(lo, hi, (self.memory_size, self.n_var))
        fit = np.array([self.fitness(h) for h in HM])
        span = hi - lo
        for _ in range(self.iterations):
            new = np.empty(self.n_var)
            for j in range(self.n_var):
                if rng.rand() < self.hmcr:                 # memory consideration
                    new[j] = HM[rng.randint(self.memory_size), j]
                    if rng.rand() < self.par:              # pitch adjustment
                        new[j] += rng.uniform(-1, 1) * self.bandwidth * span
                else:
                    new[j] = rng.uniform(lo, hi)           # random note
            new = np.clip(new, lo, hi)
            f = self.fitness(new)
            worst = fit.argmax()
            if f < fit[worst]:                             # replace the worst harmony
                HM[worst] = new; fit[worst] = f
        self.best_ = HM[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


__all__ = ["HarmonySearch"]
