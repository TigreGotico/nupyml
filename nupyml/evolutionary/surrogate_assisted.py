"""Surrogate-assisted evolution: let a cheap learned model stand in for a costly one."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class SurrogateAssistedEA(BaseEstimator):
    """Replace an EXPENSIVE fitness with a learned model most of the time (Jin, 2011).

    When each evaluation is a wind-tunnel run or a day-long simulation, an evolutionary
    algorithm's thousands of fitness calls are unaffordable. A surrogate-assisted EA
    trains a cheap regression model on the evaluations it HAS made, then screens each new
    generation on the surrogate and spends a real evaluation only on the few most
    promising candidates. Those true values retrain the surrogate, which sharpens where
    it matters. The model manages the exploration/accuracy trade-off, cutting expensive
    calls by an order of magnitude. ``surrogate`` is any regressor with fit/predict;
    ``eval_fraction`` is how much of each generation gets a true evaluation.
    """

    def __init__(self, fitness, n_var, bounds, surrogate=None, pop_size=40,
                 max_iter=30, eval_fraction=0.3, mutation=0.2, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.surrogate = surrogate
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.eval_fraction = eval_fraction
        self.mutation = mutation
        self.random_state = random_state

    def optimize(self):
        from ..neighbors import KNeighborsRegressor
        rng = check_random_state(self.random_state)
        lo, hi = np.broadcast_to(self.bounds[0], self.n_var), \
            np.broadcast_to(self.bounds[1], self.n_var)
        model = self.surrogate if self.surrogate is not None \
            else KNeighborsRegressor(n_neighbors=5)
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        X_seen = pop.copy()
        y_seen = np.array([self.fitness(x) for x in pop])   # bootstrap the surrogate
        self.n_true_evals_ = len(y_seen)
        n_eval = max(1, int(self.eval_fraction * self.pop_size))
        for _ in range(self.max_iter):
            model.fit(X_seen, y_seen)
            pred = model.predict(pop)
            # truly evaluate only the surrogate's most promising candidates
            promising = np.argsort(pred)[:n_eval]
            for i in promising:
                y = self.fitness(pop[i])
                X_seen = np.vstack([X_seen, pop[i]]); y_seen = np.append(y_seen, y)
                self.n_true_evals_ += 1
            # breed next generation from the best (by true where known, else surrogate)
            score = model.predict(pop); score[promising] = y_seen[-n_eval:]
            elite = pop[np.argsort(score)[:self.pop_size // 2]]
            children = []
            for _ in range(self.pop_size):
                parent = elite[rng.randint(len(elite))].copy()
                children.append(np.clip(parent + rng.randn(self.n_var)
                                        * self.mutation * (hi - lo), lo, hi))
            pop = np.array(children)
        self.best_ = X_seen[y_seen.argmin()]
        self.best_fitness_ = y_seen.min()
        return self.best_


__all__ = ["SurrogateAssistedEA"]
