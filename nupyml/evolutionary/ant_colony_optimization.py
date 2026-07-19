"""Build tours from a shared PHEROMONE trail (Dorigo, 1992)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class AntColonyOptimization(BaseEstimator):
    """Build tours from a shared PHEROMONE trail (Dorigo, 1992).

    Ants find short paths without a map: each lays PHEROMONE on the route it takes,
    shorter routes get reinforced faster (less evaporation time), and later ants
    prefer strong trails -- a positive feedback that converges on good solutions.
    ACO copies this for combinatorial problems like the TSP: each iteration, ants
    build tours choosing the next city by pheromone^alpha * (1/distance)^beta, then
    deposit pheromone inversely proportional to tour length, with evaporation to
    forget bad early choices. Returns the best tour found.
    """

    def __init__(self, n_ants=20, alpha=1.0, beta=3.0, evaporation=0.5,
                 iterations=100, random_state=None):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, distances):
        D = np.asarray(distances, float)
        n = len(D)
        rng = check_random_state(self.random_state)
        tau = np.ones((n, n))                             # pheromone
        eta = 1.0 / (D + np.eye(n) * 1e9)                 # desirability = 1/distance
        best_tour, best_len = None, np.inf
        for _ in range(self.iterations):
            all_tours, all_lens = [], []
            for _ant in range(self.n_ants):
                start = rng.randint(n)
                tour = [start]; unvisited = set(range(n)) - {start}
                while unvisited:
                    cur = tour[-1]
                    cities = list(unvisited)
                    w = (tau[cur, cities] ** self.alpha) * (eta[cur, cities] ** self.beta)
                    w = w / w.sum()
                    nxt = cities[rng.choice(len(cities), p=w)]
                    tour.append(nxt); unvisited.discard(nxt)
                length = sum(D[tour[i], tour[i + 1]] for i in range(n - 1)) + D[tour[-1], tour[0]]
                all_tours.append(tour); all_lens.append(length)
                if length < best_len:
                    best_len, best_tour = length, tour
            tau *= (1 - self.evaporation)                 # evaporate
            for tour, length in zip(all_tours, all_lens):
                for i in range(n):                        # deposit on used edges
                    a, b = tour[i], tour[(i + 1) % n]
                    tau[a, b] += 1.0 / length; tau[b, a] += 1.0 / length
        self.best_tour_ = best_tour
        self.best_length_ = best_len
        return self


__all__ = ["AntColonyOptimization"]
