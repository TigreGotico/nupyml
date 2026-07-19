"""A GA that also does LOCAL SEARCH (Moscato, 1989)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class MemeticAlgorithm(BaseEstimator):
    """A GA that also does LOCAL SEARCH (Moscato, 1989).

    A genetic algorithm explores globally but converges slowly to a precise optimum;
    local search refines precisely but gets stuck. A memetic algorithm combines them:
    after the usual selection/crossover/mutation, every offspring is polished by a few
    steps of LOCAL SEARCH before it competes -- Lamarckian, since the improved genotype
    is kept and inherited. The GA supplies exploration and the local search supplies
    exploitation, so it reaches better optima in fewer generations than either alone.
    Minimises ``fitness`` over a box.
    """

    def __init__(self, fitness, n_var, bounds, pop_size=30, generations=50,
                 local_steps=10, local_sigma=0.1, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.local_steps = local_steps
        self.local_sigma = local_sigma
        self.random_state = random_state

    def _local(self, x, rng):
        fx = self.fitness(x)
        for _ in range(self.local_steps):                  # random-restart hill climb
            cand = np.clip(x + rng.normal(0, self.local_sigma, self.n_var),
                           *self.bounds)
            fc = self.fitness(cand)
            if fc < fx:
                x, fx = cand, fc
        return x, fx

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        fit = np.empty(self.pop_size)
        for i in range(self.pop_size):
            pop[i], fit[i] = self._local(pop[i], rng)
        for _ in range(self.generations):
            order = np.argsort(fit)
            elite = pop[order[:max(2, self.pop_size // 4)]]
            new, newfit = list(elite), list(fit[order[:len(elite)]])
            while len(new) < self.pop_size:
                a, b = elite[rng.randint(len(elite))], elite[rng.randint(len(elite))]
                mask = rng.rand(self.n_var) < 0.5
                child = np.where(mask, a, b)               # uniform crossover
                child = np.clip(child + rng.normal(0, 0.1, self.n_var), lo, hi)
                child, fc = self._local(child, rng)        # Lamarckian refinement
                new.append(child); newfit.append(fc)
            pop, fit = np.array(new), np.array(newfit)
        self.best_ = pop[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


__all__ = ["MemeticAlgorithm"]
