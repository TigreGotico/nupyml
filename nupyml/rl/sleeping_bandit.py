"""A bandit where only a SUBSET of arms is AVAILABLE each round."""
import numpy as np
from ..utils import check_random_state


class SleepingBandit:
    """A bandit where only a SUBSET of arms is AVAILABLE each round.

    Products go out of stock, ads exhaust their budget -- the action set changes
    over time. A sleeping bandit picks the best-UCB arm among only the AVAILABLE
    ("awake") ones each round, so an arm's statistics persist across the rounds it
    sleeps and are used whenever it reappears. Ordinary bandits assume a fixed arm
    set and break here.
    """

    def __init__(self, n_arms, random_state=None):
        self.n_arms = n_arms
        self._rng = check_random_state(random_state)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self, available):
        self.t += 1
        ucb = np.full(self.n_arms, -np.inf)
        for a in available:
            ucb[a] = (self.values[a] + np.sqrt(2 * np.log(self.t + 1)
                                               / (self.counts[a] + 1e-9))
                      if self.counts[a] > 0 else np.inf)
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


__all__ = ["SleepingBandit"]
