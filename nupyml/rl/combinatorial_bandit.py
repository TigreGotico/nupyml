"""Choose a SUBSET of arms each round (semi-bandit feedback)."""
import numpy as np
from ..utils import check_random_state


class CombinatorialBandit:
    """Choose a SUBSET of arms each round (semi-bandit feedback).

    In many settings you pick SEVERAL arms at once -- a slate of ads, a set of
    sensors -- and observe each chosen arm's reward (semi-bandit feedback).
    CombinatorialBandit runs a UCB per arm and selects the top-``k`` by upper
    confidence bound, then updates every arm it played. The per-arm optimism drives
    exploration across the combinatorial action space without enumerating its
    exponentially many subsets.
    """

    def __init__(self, n_arms, k, random_state=None):
        self.n_arms = n_arms
        self.k = k
        self._rng = check_random_state(random_state)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self):
        self.t += 1
        ucb = np.where(self.counts > 0,
                       self.values + np.sqrt(2 * np.log(self.t + 1)
                                             / (self.counts + 1e-9)),
                       np.inf)                          # try each arm once first
        return np.argsort(-ucb)[:self.k]                # the top-k arms

    def update(self, arms, rewards):
        for a, r in zip(arms, rewards):
            self.counts[a] += 1
            self.values[a] += (r - self.values[a]) / self.counts[a]   # running mean


__all__ = ["CombinatorialBandit"]
