"""A genetic algorithm compressed to a PROBABILITY VECTOR (Harik, 1999)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class CompactGA(BaseEstimator):
    """A genetic algorithm compressed to a PROBABILITY VECTOR (Harik, 1999).

    The compact GA is an EDA that mimics a GA with a tiny memory footprint: instead
    of a population it stores one probability per bit. Each step it samples TWO
    individuals, lets them compete, and nudges each bit's probability a small step
    (``1/n``) toward the winner's value -- so over many tournaments the vector drifts
    to the better allele on each bit, reproducing a GA's selection pressure using
    only ``O(n)`` memory. Ideal when a full population will not fit.
    """

    def __init__(self, n_bits, virtual_pop=50, max_iter=5000, random_state=None):
        self.n_bits = n_bits
        self.virtual_pop = virtual_pop
        self.max_iter = max_iter
        self.random_state = random_state

    def optimize(self, fitness):
        rng = check_random_state(self.random_state)
        p = np.full(self.n_bits, 0.5)
        step = 1.0 / self.virtual_pop
        for _ in range(self.max_iter):
            a = (rng.rand(self.n_bits) < p).astype(int)
            b = (rng.rand(self.n_bits) < p).astype(int)
            win, lose = (a, b) if fitness(a) >= fitness(b) else (b, a)
            move = win != lose                            # only bits that differ
            # shift each differing bit 1/N toward the winner's allele
            p[move] += step * (2 * win[move] - 1)
            p = np.clip(p, 0.0, 1.0)
            if np.all((p == 0) | (p == 1)):
                break
        self.prob_ = p
        self.best_ = (p >= 0.5).astype(int)
        self.best_fitness_ = fitness(self.best_)
        return self.best_


__all__ = ["CompactGA"]
