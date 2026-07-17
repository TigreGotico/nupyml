"""Tabular RL: the Bellman equation, learned from experience.

THE BELLMAN EQUATION
--------------------
The value of a state-action is the immediate reward plus the discounted value of
where you end up::

    Q(s, a) = reward + gamma * (value of the next state)

That recursion is the whole of value-based RL. It is a consistency condition the
true values satisfy, and every algorithm here is a way of forcing an estimate to
satisfy it -- either by sweeping the known dynamics (``value_iteration``,
``policy_iteration``) or by nudging toward it from sampled experience
(``QLearning``, ``SARSA``).

``gamma`` (0 to 1) is how much the future matters. Near 0 is myopic -- grab the
nearest reward; near 1 is far-sighted -- a distant payoff is worth almost as much
as an immediate one. It also keeps the sum of rewards finite over an infinite
horizon, which is the other reason it is there.

PLANNING vs LEARNING
--------------------
* If you KNOW the dynamics (which state each action leads to, and its reward),
  you can PLAN: ``value_iteration`` and ``policy_iteration`` compute the optimal
  policy by sweeping the equation to convergence. No experience needed.
* If you do NOT know the dynamics -- the usual case -- you must LEARN from
  sampled transitions: ``QLearning`` and ``SARSA`` update the estimate one
  experienced step at a time.

The learning methods BOOTSTRAP: they update an estimate toward a target that is
itself an estimate (reward plus current Q of the next state). That is what lets
them learn before an episode ends, and also what makes them subtle -- an estimate
chasing a moving estimate.

THE DISTINCTION THAT MATTERS: ON- vs OFF-POLICY
-----------------------------------------------
``QLearning`` and ``SARSA`` differ in one symbol of their update, and it changes
their character completely -- see their docstrings. It is the single most
important idea in this file.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


def value_iteration(P, R, gamma=0.9, tol=1e-6, max_iter=1000):
    """Plan the optimal values when the dynamics are known.

    ``P[s, a, s']`` is the probability of landing in ``s'``, ``R[s, a]`` the
    reward. Sweep the Bellman OPTIMALITY equation -- value of a state is the best
    action's expected reward-plus-discounted-value -- until it stops changing::

        V(s) <- max_a [ R(s,a) + gamma * sum_s' P(s,a,s') V(s') ]

    Each sweep is a contraction by factor ``gamma``, so it converges geometrically
    to the unique optimal values. This is dynamic programming in the original
    Bellman sense, and the learning methods below are its sample-based shadows.
    """
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    for _ in range(max_iter):
        Q = R + gamma * (P @ V)            # (states, actions), vectorised sweep
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = np.argmax(R + gamma * (P @ V), axis=1)
    return V, policy


def policy_iteration(P, R, gamma=0.9, max_iter=1000):
    """Plan by alternating exact evaluation and greedy improvement.

    Two steps repeated: EVALUATE the current policy (solve for its values
    exactly, a linear system), then IMPROVE it (act greedily w.r.t. those
    values). Each improvement is provably no worse, and with finitely many
    policies it reaches the optimum in finitely many steps -- often FAR fewer
    sweeps than value iteration, because the exact evaluation does more work per
    step. The trade is exactly that: fewer, more expensive iterations.
    """
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)
    for _ in range(max_iter):
        # EVALUATE: solve V = R_pi + gamma P_pi V exactly (a linear system)
        P_pi = P[np.arange(n_states), policy]
        R_pi = R[np.arange(n_states), policy]
        V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)
        # IMPROVE: act greedily with respect to the freshly evaluated V
        new_policy = np.argmax(R + gamma * (P @ V), axis=1)
        if np.array_equal(new_policy, policy):
            break                          # stable => optimal
        policy = new_policy
    return V, policy


class _TabularTD(BaseEstimator):
    """Shared machinery for the temporal-difference control methods."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 epsilon_decay=1.0, random_state=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.random_state = random_state

    def reset(self):
        self._rng = check_random_state(self.random_state)
        self.Q_ = np.zeros((self.n_states, self.n_actions))
        self._eps = self.epsilon
        return self

    def act(self, state, greedy=False):
        """Epsilon-greedy action selection -- the exploration policy."""
        if not hasattr(self, "Q_"):
            self.reset()
        if not greedy and self._rng.uniform() < self._eps:
            return self._rng.randint(self.n_actions)
        return int(np.argmax(self.Q_[state]))

    def policy(self):
        """The greedy policy implied by the current Q-table."""
        return np.argmax(self.Q_, axis=1)


class QLearning(_TabularTD):
    """Off-policy TD control: learn the value of the GREEDY policy while exploring.

    THE UPDATE
    ----------
    ::

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * MAX_a' Q(s',a') - Q(s,a) ]

    The bracket is the TD error: the gap between the current estimate and a better
    one built from the observed reward. The ``max`` is the crux -- the target
    assumes the BEST next action will be taken, regardless of what the agent
    actually does next.

    That makes Q-learning OFF-POLICY: it learns the optimal value function even
    while behaving with random exploration. It can drink from any stream of
    experience -- old data, a human's demonstrations, another policy's rollouts --
    and still converge to the optimum. That flexibility is why Q-learning
    underlies DQN and most of deep RL.

    THE PRICE
    ---------
    Because the target ignores the exploration it will actually do, Q-learning can
    learn a policy that walks along a cliff edge -- optimal if executed perfectly,
    disastrous if the epsilon-greedy behaviour occasionally steps off. SARSA,
    which accounts for its own exploration, is the cautious counterpart.

    Watkins (1989).
    """

    def update(self, s, a, r, s_next, done=False):
        best_next = 0.0 if done else np.max(self.Q_[s_next])
        # the max over next actions is what makes this OFF-policy
        target = r + self.gamma * best_next
        self.Q_[s, a] += self.alpha * (target - self.Q_[s, a])

    def end_episode(self):
        self._eps *= self.epsilon_decay


class SARSA(_TabularTD):
    """On-policy TD control: learn the value of the policy you ACTUALLY follow.

    THE UPDATE
    ----------
    ::

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s', a') - Q(s,a) ]

    The one difference from Q-learning is ``Q(s', a')`` instead of
    ``max_a' Q(s', a')``: the target uses the action ``a'`` the agent will
    genuinely take next, exploration and all -- which is why the name is the
    tuple (State, Action, Reward, State, Action) the update touches.

    That makes SARSA ON-POLICY: it learns the value of the behaviour it is
    actually running. The consequence is famous -- on the cliff-walking problem
    SARSA learns to walk a SAFE path away from the edge, because its own updates
    account for the chance that exploration sends it over. Q-learning learns the
    shorter cliff-edge path and falls off during exploration. Neither is "more
    correct"; they optimise different things, and choosing between them is a real
    modelling decision about whether the exploration is part of the deployed
    behaviour.

    Rummery & Niranjan (1994).
    """

    def update(self, s, a, r, s_next, a_next, done=False):
        next_q = 0.0 if done else self.Q_[s_next, a_next]
        # the ACTUAL next action, not the best one -- this is ON-policy
        target = r + self.gamma * next_q
        self.Q_[s, a] += self.alpha * (target - self.Q_[s, a])

    def end_episode(self):
        self._eps *= self.epsilon_decay


class ExpectedSARSA(_TabularTD):
    """SARSA using the EXPECTED next value under the policy, not a sample of it.

    Replace ``Q(s', a')`` with the expectation over the policy's action
    distribution::

        target = r + gamma * sum_a' pi(a'|s') Q(s', a')

    Same on-policy target in expectation, but with the sampling noise of "which
    action did we happen to pick next" removed. Lower variance updates, so it
    often learns faster and more stably than plain SARSA, at the cost of a sum
    over actions per step. A small, clean illustration that averaging out an
    unnecessary sample is almost always worth it.
    """

    def update(self, s, a, r, s_next, done=False):
        if done:
            expected = 0.0
        else:
            # the epsilon-greedy action distribution at s_next
            pi = np.full(self.n_actions, self._eps / self.n_actions)
            pi[np.argmax(self.Q_[s_next])] += 1 - self._eps
            expected = pi @ self.Q_[s_next]        # average, not a sample
        target = r + self.gamma * expected
        self.Q_[s, a] += self.alpha * (target - self.Q_[s, a])

    def end_episode(self):
        self._eps *= self.epsilon_decay


__all__ = ["QLearning", "SARSA", "ExpectedSARSA", "value_iteration",
           "policy_iteration"]
