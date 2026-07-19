"""Evolve a DISTRIBUTION over good solutions, not a population (Muhlenbein, 1996)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class UMDA(BaseEstimator):
    """Evolve a DISTRIBUTION over good solutions, not a population (Muhlenbein, 1996).

    A genetic algorithm recombines individuals and hopes structure survives. An
    estimation-of-distribution algorithm makes the structure EXPLICIT: it keeps a
    probability model, samples a population from it, selects the best, and re-fits
    the model to them. UMDA is the simplest -- one independent Bernoulli probability
    PER BIT. Each generation, the selected winners' bit-frequencies become the new
    probabilities, so the model marches toward the optimum. It assumes the bits are
    independent, which is exactly its limitation and the reason richer EDAs exist.
    Maximises a binary ``fitness``.
    """

    def __init__(self, n_bits, pop_size=100, select_frac=0.5, generations=100,
                 random_state=None):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.select_frac = select_frac
        self.generations = generations
        self.random_state = random_state

    def optimize(self, fitness):
        rng = check_random_state(self.random_state)
        p = np.full(self.n_bits, 0.5)
        n_sel = max(1, int(self.pop_size * self.select_frac))
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            pop = (rng.rand(self.pop_size, self.n_bits) < p).astype(int)
            fits = np.array([fitness(ind) for ind in pop])
            elite = pop[np.argsort(fits)[::-1][:n_sel]]
            p = np.clip(elite.mean(axis=0), 0.02, 0.98)   # re-fit the model
            if fits.max() > best_fit:
                best_fit = fits.max(); best = pop[fits.argmax()].copy()
        self.prob_ = p
        self.best_, self.best_fitness_ = best, best_fit
        return best


__all__ = ["UMDA"]
