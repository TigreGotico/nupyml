"""Multi-objective by DECOMPOSITION into scalar subproblems (Zhang & Li, 2007)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class MOEAD(BaseEstimator):
    """Multi-objective by DECOMPOSITION into scalar subproblems (Zhang & Li, 2007).

    NSGA-II ranks the whole population by dominance. MOEA/D takes a different route:
    spread ``n`` weight vectors over the objective space, turning the problem into
    ``n`` SCALAR subproblems (here via the Tchebycheff aggregation), and evolve them
    together -- each subproblem improved using solutions from its NEIGHBOURS in
    weight space. Because neighbouring weights have similar optima, information
    shares cheaply, and the population sweeps out the Pareto front. ``objectives``
    returns a vector to be minimised.
    """

    def __init__(self, objectives, n_var, bounds, pop_size=50, generations=100,
                 n_neighbors=10, F=0.5, random_state=None):
        self.objectives = objectives
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.n_neighbors = n_neighbors
        self.F = F
        self.random_state = random_state

    def fit(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        # evenly spaced weight vectors on the 2-objective simplex
        w = np.linspace(0, 1, self.pop_size)
        W = np.column_stack([w, 1 - w]) + 1e-6
        B = np.argsort(np.abs(w[:, None] - w[None, :]), axis=1)[:, :self.n_neighbors]
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        F = np.array([self.objectives(x) for x in pop])
        z = F.min(axis=0)                                # ideal point
        for _ in range(self.generations):
            for i in range(self.pop_size):
                a, b, c = pop[rng.choice(B[i], 3, replace=True)]
                trial = a + self.F * (b - c)                 # DE-style offspring
                mut = rng.rand(self.n_var) < 1.0 / self.n_var
                trial[mut] += rng.normal(0, 0.1, mut.sum())  # exploration mutation
                trial = np.clip(trial, lo, hi)
                ft = self.objectives(trial)
                z = np.minimum(z, ft)
                for j in B[i]:                            # update neighbours it improves
                    g_new = np.max(W[j] * np.abs(ft - z))       # Tchebycheff
                    g_old = np.max(W[j] * np.abs(F[j] - z))
                    if g_new <= g_old:
                        pop[j] = trial; F[j] = ft
        self.solutions_ = pop
        self.objectives_ = F
        return self


__all__ = ["MOEAD"]
