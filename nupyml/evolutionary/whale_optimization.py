"""Whale optimisation: encircling prey and bubble-net spiral hunting."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class WhaleOptimization(BaseEstimator):
    """Continuous optimisation modelled on humpback BUBBLE-NET hunting (Mirjalili, 2016).

    Humpback whales corral prey by swimming up a shrinking SPIRAL while blowing a net
    of bubbles. The whale optimiser turns that into a search rule: each whale either
    ENCIRCLES the current best (a contracting move, like Grey Wolf's), or spirals in
    toward it along a logarithmic path -- chosen 50/50 each step. Early on a coefficient
    stays large so some whales instead swim toward a RANDOM peer, keeping exploration
    alive; it shrinks over time into pure exploitation. Few parameters, strong on
    multimodal landscapes. Minimises ``fitness`` over a box.
    """

    def __init__(self, fitness, n_var, bounds, n_whales=30, max_iter=200, b=1.0,
                 random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.b = b
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pos = rng.uniform(lo, hi, (self.n_whales, self.n_var))
        fits = np.array([self.fitness(w) for w in pos])
        best = pos[fits.argmin()].copy(); best_fit = fits.min()
        for it in range(self.max_iter):
            a = 2 - 2 * it / self.max_iter                 # explore -> exploit
            for i in range(self.n_whales):
                if rng.rand() < 0.5:
                    A = 2 * a * rng.rand(self.n_var) - a
                    C = 2 * rng.rand(self.n_var)
                    if np.abs(A).mean() < 1:               # encircle the best
                        D = np.abs(C * best - pos[i])
                        pos[i] = best - A * D
                    else:                                  # search toward a random peer
                        rand = pos[rng.randint(self.n_whales)]
                        D = np.abs(C * rand - pos[i])
                        pos[i] = rand - A * D
                else:                                      # bubble-net spiral to best
                    l = rng.uniform(-1, 1, self.n_var)
                    D = np.abs(best - pos[i])
                    pos[i] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best
                pos[i] = np.clip(pos[i], lo, hi)
            fits = np.array([self.fitness(w) for w in pos])
            if fits.min() < best_fit:
                best = pos[fits.argmin()].copy(); best_fit = fits.min()
        self.best_ = best
        self.best_fitness_ = best_fit
        return self.best_


__all__ = ["WhaleOptimization"]
