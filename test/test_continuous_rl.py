"""H6: continuous-control RL (DDPG, TD3, SAC), GAE, and MCTS.

The three actor-critic agents are trained on a 1-D reacher (move a point to a
goal, continuous action) and must beat a random policy; GAE must match its
recursive definition; MCTS must pick the winning arm of a two-armed tree.
"""
import numpy as np
import pytest

from nupyml.rl import DDPG, TD3, SAC, generalized_advantage_estimation, MCTS


class Reacher:
    """Continuous control: move a point to a goal; reward = -distance."""

    def __init__(self, goal=0.7, rng=None):
        self.goal = goal
        self.rng = rng or np.random.RandomState(0)

    def reset(self):
        self.s = np.array([self.rng.uniform(-1, 1)])
        self.t = 0
        return self.s.copy()

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.s = np.clip(self.s + 0.15 * a, -1.5, 1.5)
        self.t += 1
        return self.s.copy(), float(-abs(self.s[0] - self.goal)), self.t >= 15


def _train_eval(agent, is_off_policy, episodes=60, seed=0):
    rng = np.random.RandomState(seed)
    for _ in range(episodes):
        env = Reacher(rng=rng); s = env.reset(); done = False
        while not done:
            a = agent.act(s, 0.2) if is_off_policy else agent.act(s)
            s2, r, done = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.update(32)
            s = s2
    rets = []
    for _ in range(15):
        env = Reacher(rng=rng); s = env.reset(); done = False; R = 0.0
        while not done:
            a = agent.act(s, 0.0) if is_off_policy else agent.act(s, deterministic=True)
            s, r, done = env.step(a); R += r
        rets.append(R)
    return np.mean(rets)


def _random_return(seed=0):
    rng = np.random.RandomState(seed + 99)
    rets = []
    for _ in range(15):
        env = Reacher(rng=rng); env.reset(); done = False; R = 0.0
        while not done:
            _, r, done = env.step(rng.uniform(-1, 1, 1)); R += r
        rets.append(R)
    return np.mean(rets)


@pytest.mark.parametrize("make,off", [
    (lambda: DDPG(1, 1, hidden=16, random_state=0), True),
    (lambda: TD3(1, 1, hidden=16, random_state=0), True),
    (lambda: SAC(1, 1, hidden=16, random_state=0), False),
], ids=["ddpg", "td3", "sac"])
def test_continuous_agent_beats_random(make, off):
    learned = _train_eval(make(), off)
    assert learned > _random_return()


def test_td3_has_twin_critics():
    td3 = TD3(1, 1, hidden=8, random_state=0)
    assert hasattr(td3, "critic") and hasattr(td3, "critic2")   # the twin critics


def test_gae_matches_recursive_definition():
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5, 0.0]
    dones = [0, 0, 1]
    adv = generalized_advantage_estimation(rewards, values, dones,
                                           gamma=0.99, lam=0.95)
    # last step: delta = 1 + 0 - 0.5 = 0.5 (terminal) -> adv[-1] = 0.5
    assert adv[-1] == pytest.approx(0.5)
    assert len(adv) == 3 and adv[0] > adv[1] > adv[2]   # advantage accumulates back


def test_mcts_picks_the_winning_arm():
    class TwoArm:
        def legal_actions(self, s):
            return [0, 1] if s == 0 else []
        def step(self, s, a):
            return (1 if a == 1 else 2), (1.0 if a == 1 else 0.0), True
    m = MCTS(TwoArm(), n_simulations=80, random_state=0)
    assert m.search(0) == 1                            # the arm that returns reward 1
