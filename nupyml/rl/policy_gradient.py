"""Policy gradients: optimize the policy directly, on the autograd engine.

THE OTHER WAY TO DO RL
----------------------
Value methods learn how good each action is, then act greedily. Policy-gradient
methods skip the values and adjust the POLICY itself -- a parameterised
distribution over actions -- by gradient ascent on expected reward. Two reasons
this matters:

* **Continuous actions.** "Steer 13.4 degrees" has no ``max_a`` to take; you
  cannot argmax over a continuum. A policy that outputs the parameters of a
  distribution handles it directly.
* **Stochastic optimal policies.** Sometimes the best policy is genuinely random
  (bluffing in poker, breaking symmetry). A value-greedy policy is deterministic
  and cannot represent that; a policy gradient can.

THE POLICY GRADIENT THEOREM
---------------------------
You cannot backprop through the environment -- it is not differentiable, and you
do not have its equations. The theorem sidesteps this beautifully::

    grad E[reward] = E[ reward * grad log pi(action | state) ]

Read it as: push up the log-probability of actions that led to high reward, push
down those that led to low. The gradient of the ENVIRONMENT never appears --
only the gradient of your own policy, which the autograd engine gives you. That
is the whole trick, and it is why this file is short.

THE PROBLEM AND THE FIX: VARIANCE
---------------------------------
The reward in that expectation is enormously noisy -- one lucky episode swamps the
signal -- so the raw gradient is nearly useless. The standard remedy is a
BASELINE: subtract an estimate of the average reward, and multiply by the
ADVANTAGE (how much better than average this action did) instead::

    grad = E[ (reward - baseline) * grad log pi ]

Subtracting a baseline leaves the gradient UNBIASED (it has zero mean under the
policy) while slashing its variance. An action that beat the average is
reinforced; one that merely did "well, but below average" is now correctly
discouraged. ``ActorCritic`` makes the baseline a learned value function; that is
the entire step from REINFORCE to actor-critic.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd.functional import softmax, log_softmax
from ..base import BaseEstimator
from ..nn import Adam, Linear, Module, Sequential, ReLU
from ..utils import check_random_state


class _PolicyNet(Module):
    """A small network mapping a state to action logits."""

    def __init__(self, n_features, n_actions, hidden=(32,), rng=None):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [Linear(prev, h, rng=rng), ReLU()]
            prev = h
        layers.append(Linear(prev, n_actions, rng=rng))
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(Tensor._wrap(x))


class REINFORCE(BaseEstimator):
    """Monte-Carlo policy gradient: the theorem, applied episode by episode.

    THE ALGORITHM
    -------------
    Run a whole episode under the current policy. Compute each step's RETURN (the
    discounted sum of rewards that FOLLOWED it -- credit for what came after, not
    before). Then push up the log-probability of each action in proportion to its
    return. Repeat.

    "Monte-Carlo" because it waits for the full episode and uses the ACTUAL
    return, not a bootstrapped estimate. That makes it unbiased but high-variance:
    the true return is noisy, and REINFORCE feels every bit of that noise. A
    baseline (subtracting the mean return) is the cheap variance reduction applied
    here; a learned baseline is ``ActorCritic``.

    ``normalize_returns`` standardises the returns within each batch -- a crude but
    remarkably effective baseline-and-scaling that stabilises training more than
    its simplicity suggests.

    Williams (1992).
    """

    def __init__(self, n_features, n_actions, hidden=(32,), gamma=0.99,
                 learning_rate=1e-2, normalize_returns=True, random_state=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden = hidden
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.normalize_returns = normalize_returns
        self.random_state = random_state

    def _build(self):
        self._rng = check_random_state(self.random_state)
        self.policy_ = _PolicyNet(self.n_features, self.n_actions, self.hidden,
                                  rng=self._rng)
        self._opt = Adam(self.policy_.parameters(), lr=self.learning_rate)

    def act(self, state):
        """Sample an action from the policy -- stochastic, which is the point."""
        if not hasattr(self, "policy_"):
            self._build()
        logits = self.policy_(np.asarray(state, float)[None]).data[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(self._rng.choice(self.n_actions, p=probs))

    def _returns(self, rewards):
        """Discounted return FOLLOWING each step -- the credit-assignment core.

        Walking backward accumulates ``G_t = r_t + gamma * G_{t+1}`` in one pass,
        so each action is credited with everything that came after it and nothing
        before. Crediting the future, not the past, is the whole idea of a return.
        """
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            G[t] = running
        return G

    def update(self, states, actions, rewards):
        """One policy-gradient step from a batch of complete episodes' data."""
        if not hasattr(self, "policy_"):
            self._build()
        returns = self._returns(rewards)
        if self.normalize_returns and len(returns) > 1:
            # the baseline-and-scale: centre and normalise the returns, which
            # both subtracts a baseline and keeps the step size sane
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = np.asarray(states, float)
        actions = np.asarray(actions)
        logits = self.policy_(states)
        logp = log_softmax(logits, axis=1)
        # gather log pi(a|s) for the actions actually taken
        chosen = logp * Tensor(np.eye(self.n_actions)[actions])
        chosen = chosen.sum(axis=1)
        # ascend expected return => descend its negative, weighted by the return
        loss = -(chosen * Tensor(returns)).mean()

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return float(loss.item())


class ActorCritic(BaseEstimator):
    """Two heads: an actor that acts, a critic that judges -- a LEARNED baseline.

    THE STEP UP FROM REINFORCE
    --------------------------
    REINFORCE's baseline is the batch mean return -- one number for everything.
    Actor-critic learns a VALUE FUNCTION ``V(s)`` and uses it as a per-state
    baseline, so the advantage becomes::

        advantage = reward + gamma * V(s') - V(s)

    -- the TD error again, now measuring "was this action better than my critic
    expected FROM THIS STATE". A state-specific baseline cancels far more variance
    than a global constant, because most of a return's variance is about WHICH
    STATE you were in, not which action you took.

    THE TWO HALVES
    --------------
    * the ACTOR (the policy) is pushed by the advantage, exactly as in REINFORCE;
    * the CRITIC (the value function) is trained to predict the return, by
      regression against the TD target.

    They co-adapt: a better critic gives the actor a cleaner signal, and a better
    actor gives the critic a more stationary target. This actor-critic template
    -- policy plus learned value baseline -- is the skeleton of the modern methods
    (A2C, PPO); the elaborations are mostly about making that co-adaptation
    stable.
    """

    def __init__(self, n_features, n_actions, hidden=(32,), gamma=0.99,
                 learning_rate=1e-2, value_coef=0.5, random_state=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden = hidden
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.value_coef = value_coef
        self.random_state = random_state

    def _build(self):
        self._rng = check_random_state(self.random_state)
        self.actor_ = _PolicyNet(self.n_features, self.n_actions, self.hidden,
                                 rng=self._rng)
        self.critic_ = _PolicyNet(self.n_features, 1, self.hidden, rng=self._rng)
        self._opt = Adam(list(self.actor_.parameters())
                         + list(self.critic_.parameters()), lr=self.learning_rate)

    def act(self, state):
        if not hasattr(self, "actor_"):
            self._build()
        logits = self.actor_(np.asarray(state, float)[None]).data[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(self._rng.choice(self.n_actions, p=probs))

    def _returns(self, rewards):
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            G[t] = running
        return G

    def update(self, states, actions, rewards):
        if not hasattr(self, "actor_"):
            self._build()
        states = np.asarray(states, float)
        actions = np.asarray(actions)
        returns = Tensor(self._returns(rewards))

        values = self.critic_(states).reshape(-1)
        # advantage = return - critic's estimate; detached so the actor's update
        # treats the baseline as a constant, not something to push around
        advantage = (returns - values).detach()

        logp = log_softmax(self.actor_(states), axis=1)
        chosen = (logp * Tensor(np.eye(self.n_actions)[actions])).sum(axis=1)
        actor_loss = -(chosen * advantage).mean()
        # the critic is regressed onto the actual returns
        critic_loss = ((values - returns) ** 2).mean()
        loss = actor_loss + self.value_coef * critic_loss

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return float(loss.item())


__all__ = ["REINFORCE", "ActorCritic"]
