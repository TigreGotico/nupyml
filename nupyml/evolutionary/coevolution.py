"""Cooperative coevolution: evolve subcomponents in separate populations."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class CooperativeCoevolution(BaseEstimator):
    """Divide a hard problem into subcomponents evolved SEPARATELY (Potter & De Jong, 1994).

    A high-dimensional search is exponentially harder than several low-dimensional ones.
    Cooperative coevolution splits the solution vector into blocks, gives each block its
    OWN population, and evolves them in parallel. The catch is evaluation: a block is
    meaningless alone, so each candidate is scored by plugging it into a COMPLETE
    solution built from the current best of every other population (its "collaborators").
    Populations thus co-adapt -- each specialises on its block while the others hold
    context. It scales evolutionary search to many variables. Minimises ``fitness`` over
    a box; ``n_subcomponents`` splits the variables.
    """

    def __init__(self, fitness, n_var, bounds, n_subcomponents=2, pop_size=30,
                 max_iter=100, mutation=0.2, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.n_subcomponents = n_subcomponents
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation = mutation
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.broadcast_to(self.bounds[0], self.n_var), \
            np.broadcast_to(self.bounds[1], self.n_var)
        blocks = np.array_split(np.arange(self.n_var), self.n_subcomponents)
        pops = [rng.uniform(lo[b], hi[b], (self.pop_size, len(b))) for b in blocks]
        best = np.array([rng.uniform(lo[i], hi[i]) for i in range(self.n_var)])

        def assemble(block_i, candidate):
            x = best.copy()
            x[blocks[block_i]] = candidate
            return x

        for _ in range(self.max_iter):
            for bi, b in enumerate(blocks):
                # score each candidate in the context of the other blocks' best
                fits = np.array([self.fitness(assemble(bi, c)) for c in pops[bi]])
                order = np.argsort(fits)
                best[b] = pops[bi][order[0]]               # adopt this block's winner
                # breed the next population: mutate the top half
                elite = pops[bi][order[:self.pop_size // 2]]
                children = []
                for _ in range(self.pop_size):
                    parent = elite[rng.randint(len(elite))].copy()
                    step = rng.randn(len(b)) * self.mutation * (hi[b] - lo[b])
                    children.append(np.clip(parent + step, lo[b], hi[b]))
                pops[bi] = np.array(children)
                pops[bi][0] = best[b]                       # elitism
        self.best_ = best
        self.best_fitness_ = self.fitness(best)
        return self.best_


__all__ = ["CooperativeCoevolution"]
