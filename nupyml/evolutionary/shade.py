"""Differential evolution that ADAPTS its own knobs (Tanabe & Fukunaga, 2013)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


def _cauchy(rng, loc, scale):
    return loc + scale * np.tan(np.pi * (rng.rand() - 0.5))


class SHADE(BaseEstimator):
    """Differential evolution that ADAPTS its own knobs (Tanabe & Fukunaga, 2013).

    Differential evolution's performance hinges on two constants -- the scale ``F`` and
    crossover rate ``CR`` -- and the best values change during the run and across
    problems. SHADE removes the guesswork with SUCCESS-HISTORY ADAPTATION: it keeps a
    memory of the ``F``/``CR`` values that recently produced IMPROVEMENTS, samples new
    trials' parameters from that memory, and updates it toward what keeps working
    (weighted Lehmer mean). Combined with current-to-pbest mutation and an archive of
    replaced solutions, it is a top-tier black-box optimiser with essentially no tuning.
    Minimises ``fitness``.
    """

    def __init__(self, fitness, n_var, bounds, pop_size=50, generations=200,
                 memory_size=10, p_best=0.1, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.memory_size = memory_size
        self.p_best = p_best
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        fit = np.array([self.fitness(x) for x in pop])
        MF = np.full(self.memory_size, 0.5)
        MCR = np.full(self.memory_size, 0.5)
        mem_pos = 0
        archive = []
        for _ in range(self.generations):
            sF, sCR, weights = [], [], []
            n_best = max(2, int(self.p_best * self.pop_size))
            best_idx = np.argsort(fit)[:n_best]
            for i in range(self.pop_size):
                r = rng.randint(self.memory_size)
                CR = np.clip(rng.normal(MCR[r], 0.1), 0, 1)
                F = -1
                while F <= 0:
                    F = min(_cauchy(rng, MF[r], 0.1), 1.0)
                pbest = pop[rng.choice(best_idx)]
                a = pop[rng.randint(self.pop_size)]
                pool = np.vstack([pop] + ([np.array(archive)] if archive else []))
                b = pool[rng.randint(len(pool))]
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)
                cross = rng.rand(self.n_var) < CR
                cross[rng.randint(self.n_var)] = True
                trial = np.clip(np.where(cross, mutant, pop[i]), lo, hi)
                ft = self.fitness(trial)
                if ft < fit[i]:                            # success -> record F, CR
                    archive.append(pop[i].copy())
                    sF.append(F); sCR.append(CR); weights.append(fit[i] - ft)
                    pop[i] = trial; fit[i] = ft
            if len(archive) > self.pop_size:
                archive = [archive[k] for k in rng.choice(len(archive),
                                                          self.pop_size, replace=False)]
            if sF:                                          # update the memories
                w = np.array(weights); w /= w.sum()
                MF[mem_pos] = np.sum(w * np.array(sF) ** 2) / np.sum(w * np.array(sF))
                MCR[mem_pos] = np.sum(w * np.array(sCR) ** 2) / (
                    np.sum(w * np.array(sCR)) + 1e-12)
                mem_pos = (mem_pos + 1) % self.memory_size
        self.best_ = pop[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


__all__ = ["SHADE"]
