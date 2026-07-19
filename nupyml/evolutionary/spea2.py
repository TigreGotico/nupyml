"""Strength-Pareto EA 2: rank by dominance AND density (Zitzler, 2001)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class SPEA2(BaseEstimator):
    """Strength-Pareto EA 2: rank by dominance AND density (Zitzler, 2001).

    NSGA-II ranks by non-domination fronts; SPEA2 uses a finer fitness. Each
    solution's STRENGTH is how many others it dominates; its raw fitness is the sum
    of the strengths of everything that dominates IT (so non-dominated solutions
    score 0). A DENSITY term -- the distance to the k-th nearest neighbour -- breaks
    ties toward isolated solutions, spreading the population along the front. An
    external ARCHIVE of the best-so-far is truncated to keep that spread. Minimises a
    vector ``objectives``.
    """

    def __init__(self, objectives, n_var, bounds, pop_size=50, archive_size=50,
                 generations=100, random_state=None):
        self.objectives = objectives
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.generations = generations
        self.random_state = random_state

    def _fitness(self, F):
        n = len(F)
        dom = np.zeros((n, n), bool)
        for i in range(n):
            dom[i] = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        strength = dom.sum(axis=1)
        raw = np.array([strength[dom[:, i]].sum() for i in range(n)])
        from scipy.spatial.distance import cdist
        D = np.sort(cdist(F, F), axis=1)
        k = int(np.sqrt(n))
        density = 1.0 / (D[:, k] + 2.0)
        return raw + density

    def fit(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        archive = np.empty((0, self.n_var))
        for _ in range(self.generations):
            union = np.vstack([pop, archive]) if len(archive) else pop
            F = np.array([self.objectives(x) for x in union])
            fit = self._fitness(F)
            nd = np.where(fit < 1.0)[0]                    # non-dominated
            if len(nd) <= self.archive_size:
                order = np.argsort(fit)[:self.archive_size]
            else:
                order = nd[:self.archive_size]
            archive = union[order]
            # binary tournament + gaussian mutation to make offspring
            offspring = []
            afit = fit[order]
            while len(offspring) < self.pop_size:
                i, j = rng.randint(len(archive), size=2)
                parent = archive[i] if afit[i] <= afit[j] else archive[j]
                child = np.clip(parent + rng.normal(0, 0.1, self.n_var), lo, hi)
                offspring.append(child)
            pop = np.array(offspring)
        self.archive_ = archive
        self.objectives_ = np.array([self.objectives(x) for x in archive])
        return self


__all__ = ["SPEA2"]
