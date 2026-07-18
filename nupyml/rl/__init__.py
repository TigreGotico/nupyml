"""Reinforcement learning and bandits: learning from consequences.

WHAT MAKES THIS DIFFERENT FROM EVERYTHING ELSE
----------------------------------------------
Supervised learning is told the right answer for each input. Reinforcement
learning is told only a REWARD -- a scalar, often delayed, that grades the whole
sequence of choices rather than any single one. Three difficulties follow, and
they are what the field is about:

* **No labels, only rewards.** You are never told the best action, only how good
  the one you took turned out to be. You must try actions to find out.
* **Credit assignment.** A reward arrives after a long chain of decisions. Which
  of them earned it? A move early in a chess game may decide the result twenty
  moves later.
* **Exploration vs exploitation.** To learn you must try unknown actions; to
  score you must take the best known one. Every method here is, at bottom, a
  policy for trading those off.

TWO SETTINGS, ONE FIELD
-----------------------
* **Bandits** (``bandits.py``) -- one state, repeated. Pull an arm, get a reward,
  repeat. No credit-assignment problem, so exploration-vs-exploitation stands
  ALONE and can be studied cleanly. This is where UCB and Thompson sampling
  come from, and it is the right place to start.
* **Full RL** (``tabular.py``, ``policy_gradient.py``) -- many states, actions
  that change the state, rewards that arrive later. Now all three difficulties
  are present at once.

THE TWO WAYS TO SOLVE FULL RL
-----------------------------
* **Value-based** (``tabular.py``: Q-learning, SARSA). Learn how good each
  (state, action) is; act greedily on those values. The Bellman equation --
  value now equals reward plus discounted value next -- is the whole engine, and
  it is bootstrapping: an estimate updated toward another estimate.
* **Policy-based** (``policy_gradient.py``: REINFORCE, actor-critic). Skip the
  values and adjust the POLICY directly by gradient ascent on expected reward.
  Necessary when the action space is continuous, and the natural home for the
  neural-network policies the autograd engine makes possible.

The on-policy/off-policy distinction (SARSA vs Q-learning) and the bias-variance
trade in the policy-gradient baseline are the two ideas worth carrying out of
this package.
"""
from .bandits import (EpsilonGreedy, UCB1, ThompsonSampling, LinUCB, EXP3)
from .tabular import QLearning, SARSA, ExpectedSARSA, value_iteration, policy_iteration
from .policy_gradient import REINFORCE, ActorCritic
from ._deep import DQN, PPO
from ._continuous import (DDPG, TD3, SAC, generalized_advantage_estimation, MCTS)
from ._bandits2 import (LinearThompsonSampling, CombinatorialBandit, SleepingBandit, DynaQ)

__all__ = [
    "EpsilonGreedy", "UCB1", "ThompsonSampling", "LinUCB", "EXP3",
    "QLearning", "SARSA", "ExpectedSARSA", "value_iteration", "policy_iteration",
    "REINFORCE", "ActorCritic", "DQN", "PPO",
    "DDPG", "TD3", "SAC", "generalized_advantage_estimation", "MCTS",
    "LinearThompsonSampling", "CombinatorialBandit", "SleepingBandit", "DynaQ",
]
