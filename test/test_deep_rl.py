"""G9: deep RL with function approximation -- DQN and PPO.

Both are trained on a small LineWorld (walk right to a goal for +1) and must
learn a near-optimal policy: greedy evaluation reaches the goal (return ~1),
clearly beating a random policy. DQN's learned Q must prefer moving toward the
goal at every state.
"""
import numpy as np
import pytest

from nupyml.rl import DQN, PPO


class LineWorld:
    """positions 0..n-1, start 0, goal n-1; action 0=left, 1=right; +1 at goal."""

    def __init__(self, n=6, max_steps=20):
        self.n = n
        self.max_steps = max_steps

    def reset(self):
        self.pos = 0
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(self.n)
        o[self.pos] = 1.0
        return o

    def step(self, a):
        self.pos = max(0, min(self.n - 1, self.pos + (1 if a == 1 else -1)))
        self.t += 1
        done = self.pos == self.n - 1 or self.t >= self.max_steps
        return self._obs(), (1.0 if self.pos == self.n - 1 else 0.0), done


def _greedy_return(policy, n_eval=30):
    total = 0.0
    for _ in range(n_eval):
        env = LineWorld(); s = env.reset(); done = False
        while not done:
            s, r, done = env.step(policy(s))
            total += r
    return total / n_eval


def _random_return(seed=0, n_eval=50):
    rng = np.random.RandomState(seed)
    total = 0.0
    for _ in range(n_eval):
        env = LineWorld(); env.reset(); done = False
        while not done:
            _, r, done = env.step(rng.randint(2))
            total += r
    return total / n_eval


def test_dqn_learns_optimal_policy():
    dqn = DQN(6, 2, hidden=(32,), lr=2e-3, random_state=0)
    for ep in range(500):
        eps = max(0.05, 1.0 - ep / 300)
        env = LineWorld(); s = env.reset(); done = False
        while not done:
            a = dqn.act(s, epsilon=eps)
            s2, r, done = env.step(a)
            dqn.remember(s, a, r, s2, done)
            dqn.update()
            s = s2
    ret = _greedy_return(lambda s: dqn.act(s, epsilon=0.0))
    assert ret > 0.9                                  # reaches the goal reliably
    assert ret > _random_return()
    # Q prefers moving RIGHT (toward the goal) at every non-terminal state
    Q = dqn.q(np.eye(6)).data
    assert np.all(Q[:5, 1] > Q[:5, 0])


def test_ppo_learns_to_reach_the_goal():
    ppo = PPO(6, 2, hidden=(32,), lr=3e-3, random_state=0)
    for ep in range(150):
        env = LineWorld(); s = env.reset(); done = False
        S, A, R, D = [], [], [], []
        while not done:
            a = ppo.act(s)
            s2, r, done = env.step(a)
            S.append(s); A.append(a); R.append(r); D.append(done)
            s = s2
        ppo.update(S, A, R, D)
    ret = _greedy_return(ppo.act)
    assert ret > 0.9
    assert ret > _random_return()


def test_dqn_target_network_syncs():
    dqn = DQN(6, 2, hidden=(16,), batch_size=8, target_sync=5, random_state=0)
    env = LineWorld(); s = env.reset(); done = False
    # fill the buffer
    for _ in range(50):
        a = dqn.act(s, epsilon=1.0)
        s2, r, done = env.step(a)
        dqn.remember(s, a, r, s2, done)
        s = env.reset() if done else s2
        done = False
    before = list(dqn.target.parameters())[0].data.copy()
    for _ in range(20):
        dqn.update()
    # after >target_sync updates the target has been resynced to the online net
    assert np.allclose(list(dqn.target.parameters())[0].data,
                       list(dqn.q.parameters())[0].data)
