"""Dyna-Q: interleave real experience with PLANNING on a learned model"""
import numpy as np
from ..utils import check_random_state


class DynaQ:
    """Dyna-Q: interleave real experience with PLANNING on a learned model
    (Sutton, 1990).

    Model-free Q-learning learns only from real transitions -- data-hungry when
    interaction is expensive. Dyna-Q also LEARNS A MODEL of the environment
    (remembering observed ``(s, a) -> (r, s')``) and, after each real step, runs
    ``n_planning`` simulated Q-updates from random remembered transitions. Those
    imagined updates propagate value across the state space far faster than real
    experience alone, so Dyna-Q reaches a good policy in a fraction of the
    environment interactions -- the essence of model-based RL. Discrete states and
    actions.
    """

    def __init__(self, n_states, n_actions, gamma=0.95, lr=0.1, epsilon=0.1,
                 n_planning=10, random_state=None):
        self.n_actions = n_actions
        self.gamma, self.lr, self.epsilon = gamma, lr, epsilon
        self.n_planning = n_planning
        self._rng = check_random_state(random_state)
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}                                # (s,a) -> (r, s2)

    def act(self, s):
        if self._rng.rand() < self.epsilon:
            return self._rng.randint(self.n_actions)
        return int(self.Q[s].argmax())

    def update(self, s, a, r, s2):
        self._q_update(s, a, r, s2)
        self.model[(s, a)] = (r, s2)                   # remember the transition
        keys = list(self.model.keys())
        for _ in range(self.n_planning):               # plan on remembered ones
            ps, pa = keys[self._rng.randint(len(keys))]
            pr, ps2 = self.model[(ps, pa)]
            self._q_update(ps, pa, pr, ps2)

    def _q_update(self, s, a, r, s2):
        target = r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.lr * (target - self.Q[s, a])


__all__ = ["DynaQ"]
