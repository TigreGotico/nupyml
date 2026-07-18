"""Bandit and planning expansion: linear Thompson sampling, combinatorial and
sleeping bandits, and Dyna-Q.

The base bandits (Eps-greedy/UCB1/Thompson/LinUCB/EXP3) pull ONE arm per round
from a FIXED set. These handle contextual posterior sampling, choosing a SUBSET,
a changing available set, and model-based planning.
"""
import numpy as np

from ..utils import check_random_state


class LinearThompsonSampling:
    """Contextual bandit by POSTERIOR SAMPLING of a linear reward model
    (Agrawal & Goyal, 2013).

    LinUCB adds an optimism BONUS to each arm's predicted reward; linear Thompson
    sampling instead maintains a Bayesian posterior over the reward weights and
    SAMPLES a weight vector from it each round, acting greedily w.r.t. the sample.
    The sampling itself provides exploration -- arms the posterior is uncertain
    about occasionally get a high sampled value and are tried. It often matches or
    beats LinUCB and needs no bonus to tune; the Gaussian posterior over ``theta``
    has a closed-form update per observed (context, reward).
    """

    def __init__(self, n_features, alpha=1.0, v=1.0, random_state=None):
        self.d = n_features
        self.alpha = alpha
        self.v = v
        self._rng = check_random_state(random_state)
        self.A = np.eye(n_features) / alpha           # precision
        self.b = np.zeros(n_features)

    def select(self, contexts):
        """contexts: (n_arms, n_features). Return the chosen arm index."""
        contexts = np.asarray(contexts, float)
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        theta = self._rng.multivariate_normal(mu, self.v ** 2 * A_inv)  # sample
        return int(np.argmax(contexts @ theta))

    def update(self, context, reward):
        x = np.asarray(context, float)
        self.A += np.outer(x, x)                       # Bayesian posterior update
        self.b += reward * x


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


__all__ = ["LinearThompsonSampling", "CombinatorialBandit", "SleepingBandit",
           "DynaQ"]
