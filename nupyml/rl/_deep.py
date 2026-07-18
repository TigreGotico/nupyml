"""Value- and policy-based deep RL with function approximation: DQN and PPO.

These join the tabular methods (Q-learning/SARSA) and policy gradients
(REINFORCE/ActorCritic). They use nupyml's autograd MLPs so they scale to
continuous state spaces where a Q-TABLE is impossible. Environments follow the
usual ``reset() -> state`` / ``step(action) -> (state, reward, done)`` protocol.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..nn import Sequential, Linear, ReLU, Adam
from ..utils import check_random_state


def _mlp(n_in, n_out, hidden, rng):
    dims = [n_in, *hidden, n_out]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [Linear(a, b, rng=rng), ReLU()]
    return Sequential(*layers[:-1])                   # drop trailing ReLU


class DQN:
    """Deep Q-Network: a neural Q-function with replay and a target net (Mnih, 2015).

    THE TWO STABILISERS
    -------------------
    Tabular Q-learning cannot handle continuous states; replacing the table with a
    network is obvious but UNSTABLE, for two reasons DQN fixes:

    * **Experience replay** -- store transitions in a buffer and train on random
      minibatches, breaking the strong temporal correlation of consecutive steps
      that would otherwise make the network chase its own tail.
    * **Target network** -- compute the TD target with a FROZEN copy of the network
      that is only synced occasionally, so the target does not move every step
      (bootstrapping off a moving target diverges).

    Acts epsilon-greedily; ``remember`` stores a transition; ``update`` does one
    minibatch gradient step on the squared TD error and periodically syncs the
    target.
    """

    def __init__(self, n_features, n_actions, hidden=(64,), gamma=0.99,
                 lr=1e-3, buffer_size=10000, batch_size=64, target_sync=100,
                 random_state=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self._rng = check_random_state(random_state)
        self.q = _mlp(n_features, n_actions, hidden, self._rng)
        self.target = _mlp(n_features, n_actions, hidden, self._rng)
        self._sync()
        self._opt = Adam(self.q.parameters(), lr=lr)
        self._buffer = []
        self._buffer_size = buffer_size
        self._steps = 0

    def _sync(self):
        for tp, qp in zip(self.target.parameters(), self.q.parameters()):
            tp.data[...] = qp.data

    def act(self, state, epsilon=0.1):
        if self._rng.rand() < epsilon:
            return int(self._rng.randint(self.n_actions))
        q = self.q(np.asarray(state, float)[None]).data[0]
        return int(q.argmax())

    def remember(self, s, a, r, s2, done):
        self._buffer.append((np.asarray(s, float), a, r, np.asarray(s2, float), done))
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)

    def update(self):
        if len(self._buffer) < self.batch_size:
            return None
        idx = self._rng.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in idx]
        s = np.array([b[0] for b in batch])
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch], float)
        s2 = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch], float)
        # TD target from the FROZEN target network (no gradient through it)
        q_next = self.target(s2).data.max(axis=1)
        target = r + self.gamma * q_next * (1 - done)
        q_sa = self.q(s)[np.arange(self.batch_size), a]
        loss = ((q_sa - Tensor(target)) ** 2).mean()
        self._opt.zero_grad(); loss.backward(); self._opt.step()
        self._steps += 1
        if self._steps % self.target_sync == 0:
            self._sync()                              # periodic target refresh
        return float(loss.item())


class PPO:
    """Proximal Policy Optimization -- take the biggest SAFE policy step
    (Schulman, 2017).

    THE CLIPPED OBJECTIVE
    ---------------------
    Policy gradients are fragile: one too-large step can collapse a good policy and
    it never recovers. PPO lets you reuse a batch of experience for several
    epochs while forbidding the policy from moving too far. It maximises the
    probability ratio ``r = pi_new/pi_old`` times the advantage, but CLIPS that
    ratio to ``[1-eps, 1+eps]`` and takes the pessimistic minimum::

        L = min( r * A,  clip(r, 1-eps, 1+eps) * A )

    so once the new policy has moved ``eps`` in the helpful direction the objective
    flattens -- removing the incentive to overshoot. A critic estimates the
    advantages. This clip is the whole reason PPO is the default deep-RL algorithm:
    the stability of trust-region methods with first-order simplicity.

    ``act`` samples an action; ``update`` runs several clipped epochs over a batch
    of transitions.
    """

    def __init__(self, n_features, n_actions, hidden=(64,), gamma=0.99,
                 lr=3e-3, clip_eps=0.2, epochs=6, random_state=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self._rng = check_random_state(random_state)
        self.actor = _mlp(n_features, n_actions, hidden, self._rng)
        self.critic = _mlp(n_features, 1, hidden, self._rng)
        self._opt = Adam(list(self.actor.parameters())
                         + list(self.critic.parameters()), lr=lr)

    def act(self, state):
        logits = self.actor(np.asarray(state, float)[None]).data[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        return int(self._rng.choice(self.n_actions, p=p))

    def _returns(self, rewards, dones):
        out = np.zeros(len(rewards))
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.gamma * g * (1 - dones[t])
            out[t] = g
        return out

    def update(self, states, actions, rewards, dones):
        s = np.asarray(states, float)
        a = np.asarray(actions)
        returns = self._returns(np.asarray(rewards, float), np.asarray(dones, float))
        idx = np.arange(len(a))
        # advantage = return - baseline (the critic), normalised
        values = self.critic(s).data.ravel()
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logp = F.log_softmax(self.actor(s), axis=1).data[idx, a]
        for _ in range(self.epochs):
            logp = F.log_softmax(self.actor(s), axis=1)[idx, a]
            ratio = (logp - Tensor(old_logp)).exp()
            surr1 = ratio * Tensor(adv)
            surr2 = ratio.clip(1 - self.clip_eps, 1 + self.clip_eps) * Tensor(adv)
            # elementwise min(surr1, surr2) = surr1 - relu(surr1 - surr2)
            clipped = surr1 - (surr1 - surr2).relu()
            actor_loss = -clipped.mean()
            value = self.critic(s).reshape(-1)
            critic_loss = ((value - Tensor(returns)) ** 2).mean()
            loss = actor_loss + 0.5 * critic_loss
            self._opt.zero_grad(); loss.backward(); self._opt.step()
        return float(loss.item())


__all__ = ["DQN", "PPO"]
