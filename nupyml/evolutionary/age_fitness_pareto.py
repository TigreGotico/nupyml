"""Age-Fitness Pareto optimisation: age as a second objective to preserve novelty."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class AgeFitnessParetoOptimization(BaseEstimator):
    """Fight premature convergence by making AGE a second objective (Schmidt & Lipson, 2011).

    Elitist EAs collapse onto one lineage and stall in a local optimum. This adds a
    second objective -- AGE, the number of generations since an individual's oldest
    ancestor entered the population -- and selects on the Pareto front of (fitness, age).
    A young individual survives even if mediocre, because nothing yet dominates it on
    BOTH objectives, so fresh lineages get time to mature instead of being crushed by
    incumbents. Each generation also injects a brand-new age-0 random individual, giving
    a continual supply of novelty. Robust global search with almost no tuning. Minimises
    ``fitness`` over a box.
    """

    def __init__(self, fitness, n_var, bounds, pop_size=50, max_iter=100,
                 mutation=0.2, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation = mutation
        self.random_state = random_state

    @staticmethod
    def _pareto(objs):
        # indices not dominated on both (fitness, age) -- lower is better on both
        n = len(objs)
        keep = np.ones(n, bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    keep[i] = False
                    break
        return np.where(keep)[0]

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.broadcast_to(self.bounds[0], self.n_var), \
            np.broadcast_to(self.bounds[1], self.n_var)
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        age = np.zeros(self.pop_size)
        best, best_fit = None, np.inf
        for _ in range(self.max_iter):
            age += 1
            # one fresh age-0 individual keeps injecting novelty
            pop = np.vstack([pop, rng.uniform(lo, hi, self.n_var)])
            age = np.append(age, 0)
            # breed: mutate random survivors
            children = np.clip(pop + rng.randn(*pop.shape) * self.mutation * (hi - lo),
                               lo, hi)
            child_age = age.copy()
            pop = np.vstack([pop, children]); age = np.append(age, child_age)
            fits = np.array([self.fitness(x) for x in pop])
            if fits.min() < best_fit:
                best_fit = fits.min(); best = pop[fits.argmin()].copy()
            # select the (fitness, age) Pareto front, then fill by fitness
            objs = np.column_stack([fits, age])
            front = self._pareto(objs)
            keep = list(front)
            if len(keep) < self.pop_size:
                rest = [i for i in np.argsort(fits) if i not in set(keep)]
                keep += rest[:self.pop_size - len(keep)]
            keep = np.array(keep[:self.pop_size])
            pop, age = pop[keep], age[keep]
        self.best_ = best
        self.best_fitness_ = best_fit
        return self.best_


__all__ = ["AgeFitnessParetoOptimization"]
