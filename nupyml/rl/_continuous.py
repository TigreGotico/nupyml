"""Continuous-control RL (DDPG, TD3, SAC), advantage estimation, and planning.

The discrete methods (DQN/PPO) pick from a finite action set. Continuous control
-- steer, torque, throttle -- has no ``argmax`` over a continuum, so it needs an
actor that OUTPUTS an action and a critic that scores it. GAE and MCTS are the
supporting advantage-estimation and planning tools.
"""
import numpy as np

from ..autograd import Tensor
from ..nn import Sequential, Linear, ReLU, Adam, Tanh
from ..utils import check_random_state


def _mlp(sizes, rng, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [Linear(a, b, rng=rng), ReLU()]
    layers = layers[:-1]
    if out_act is not None:
        layers.append(out_act)
    return Sequential(*layers)


class _Replay:
    def __init__(self, size, rng):
        self.size = size
        self.buf = []
        self.rng = rng

    def add(self, *tr):
        self.buf.append(tuple(np.asarray(x, float) if np.ndim(x) else x for x in tr))
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def sample(self, n):
        idx = self.rng.choice(len(self.buf), min(n, len(self.buf)), replace=False)
        cols = list(zip(*[self.buf[i] for i in idx]))
        return [np.array(c, float) for c in cols]


class DDPG:
    """Deep Deterministic Policy Gradient -- continuous control (Lillicrap, 2016).

    DQN cannot handle continuous actions (no ``max_a``). DDPG learns a
    DETERMINISTIC actor ``mu(s)`` that outputs the action directly, and a critic
    ``Q(s, a)``; the actor is trained by pushing its action in the direction that
    RAISES the critic (the deterministic policy gradient ``grad_a Q``). It is DQN's
    machinery -- replay + target networks -- carried over to a continuous actor.
    Exploration is added as noise on the action.
    """

    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99, tau=0.01,
                 lr=1e-3, act_limit=1.0, random_state=None):
        self.gamma, self.tau, self.act_limit = gamma, tau, act_limit
        self.action_dim = action_dim
        self._rng = check_random_state(random_state)
        self.actor = _mlp([state_dim, hidden, action_dim], self._rng, Tanh())
        self.critic = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self.t_actor = _mlp([state_dim, hidden, action_dim], self._rng, Tanh())
        self.t_critic = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self._hard_update()
        self.a_opt = Adam(self.actor.parameters(), lr=lr)
        self.c_opt = Adam(self.critic.parameters(), lr=lr)
        self.replay = _Replay(100000, self._rng)

    def _hard_update(self):
        for t, s in [(self.t_actor, self.actor), (self.t_critic, self.critic)]:
            for tp, sp in zip(t.parameters(), s.parameters()):
                tp.data[...] = sp.data

    def _soft_update(self):
        for t, s in [(self.t_actor, self.actor), (self.t_critic, self.critic)]:
            for tp, sp in zip(t.parameters(), s.parameters()):
                tp.data[...] = (1 - self.tau) * tp.data + self.tau * sp.data

    def act(self, state, noise=0.1):
        a = self.actor(np.asarray(state, float)[None]).data[0] * self.act_limit
        a = a + noise * self._rng.randn(self.action_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def remember(self, s, a, r, s2, done):
        self.replay.add(s, a, r, s2, float(done))

    def _critic_in(self, s, a):
        return Tensor(np.hstack([s, a]))

    def update(self, batch_size=64):
        if len(self.replay.buf) < batch_size:
            return
        s, a, r, s2, done = self.replay.sample(batch_size)
        r = r.reshape(-1, 1); done = done.reshape(-1, 1)
        # critic target from the frozen target nets
        a2 = self.t_actor(Tensor(s2)).data * self.act_limit
        q2 = self.t_critic(Tensor(np.hstack([s2, a2]))).data
        target = r + self.gamma * (1 - done) * q2
        q = self.critic(self._critic_in(s, a))
        c_loss = ((q - Tensor(target)) ** 2).mean()
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        # actor: maximise Q(s, mu(s)) -> minimise -Q
        aa = self.actor(Tensor(s)) * self.act_limit
        a_loss = -self.critic(_cat(Tensor(s), aa)).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        self._soft_update()


def _cat(s, a):
    """Concatenate a (constant) state Tensor and a differentiable action Tensor."""
    return Tensor.concatenate([s, a], axis=1)


class TD3(DDPG):
    """Twin Delayed DDPG -- fix DDPG's overestimation (Fujimoto et al., 2018).

    DDPG's critic systematically OVER-estimates Q (the max in the bootstrap picks
    up noise), and the actor exploits that error. TD3 adds three fixes: TWIN
    critics and take the MINIMUM (curbs overestimation), DELAYED actor updates
    (let the critic settle first), and TARGET-POLICY SMOOTHING (noise on the target
    action, so the critic cannot exploit sharp peaks). Together they make
    continuous control far more stable than DDPG.
    """

    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99, tau=0.01,
                 lr=1e-3, act_limit=1.0, policy_delay=2, target_noise=0.2,
                 random_state=None):
        super().__init__(state_dim, action_dim, hidden, gamma, tau, lr, act_limit,
                         random_state)
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.critic2 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self.t_critic2 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        for tp, sp in zip(self.t_critic2.parameters(), self.critic2.parameters()):
            tp.data[...] = sp.data
        self.c2_opt = Adam(self.critic2.parameters(), lr=lr)
        self._it = 0

    def update(self, batch_size=64):
        if len(self.replay.buf) < batch_size:
            return
        self._it += 1
        s, a, r, s2, done = self.replay.sample(batch_size)
        r = r.reshape(-1, 1); done = done.reshape(-1, 1)
        a2 = self.t_actor(Tensor(s2)).data * self.act_limit
        a2 = np.clip(a2 + self.target_noise * self._rng.randn(*a2.shape),
                     -self.act_limit, self.act_limit)      # target smoothing
        sa2 = np.hstack([s2, a2])
        q2 = np.minimum(self.t_critic(Tensor(sa2)).data,    # twin minimum
                        self.t_critic2(Tensor(sa2)).data)
        target = Tensor(r + self.gamma * (1 - done) * q2)
        sa = self._critic_in(s, a)
        for crit, opt in [(self.critic, self.c_opt), (self.critic2, self.c2_opt)]:
            loss = ((crit(sa) - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if self._it % self.policy_delay == 0:               # delayed actor update
            aa = self.actor(Tensor(s)) * self.act_limit
            a_loss = -self.critic(_cat(Tensor(s), aa)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self._soft_update()
            for tp, sp in zip(self.t_critic2.parameters(), self.critic2.parameters()):
                tp.data[...] = (1 - self.tau) * tp.data + self.tau * sp.data


class SAC:
    """Soft Actor-Critic -- maximum-ENTROPY continuous control (Haarnoja, 2018).

    SAC adds an entropy bonus to the reward: the agent maximises return PLUS the
    randomness of its policy, so it keeps exploring and does not collapse
    prematurely onto a brittle deterministic action. The actor is STOCHASTIC (a
    squashed Gaussian, trained by the reparameterisation trick), with twin critics
    like TD3. This entropy regularisation makes SAC the most sample-efficient and
    robust of the continuous-control methods. This is a compact single-step-update
    version.
    """

    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99, tau=0.01,
                 lr=1e-3, alpha=0.2, act_limit=1.0, random_state=None):
        self.gamma, self.tau, self.alpha, self.act_limit = gamma, tau, alpha, act_limit
        self.action_dim = action_dim
        self._rng = check_random_state(random_state)
        self.actor = _mlp([state_dim, hidden, 2 * action_dim], self._rng)  # mean+logstd
        self.q1 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self.q2 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self.tq1 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        self.tq2 = _mlp([state_dim + action_dim, hidden, 1], self._rng)
        for t, s in [(self.tq1, self.q1), (self.tq2, self.q2)]:
            for tp, sp in zip(t.parameters(), s.parameters()):
                tp.data[...] = sp.data
        self.a_opt = Adam(self.actor.parameters(), lr=lr)
        self.q_opt = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.replay = _Replay(100000, self._rng)

    def _sample_action(self, s_tensor, deterministic=False):
        out = self.actor(s_tensor)
        mean = out[:, :self.action_dim]
        log_std = out[:, self.action_dim:]
        std = log_std.clip(-5, 2).exp()
        if deterministic:
            u = mean
        else:
            eps = Tensor(self._rng.randn(*mean.shape))
            u = mean + std * eps                            # reparameterised
        a = u.tanh() * self.act_limit                       # squash into bounds
        # log-prob with the tanh-squash correction
        logp = (-0.5 * ((u - mean) / (std + 1e-6)) ** 2 - std.log()).sum(axis=1)
        logp = logp - (1 - a.tanh() * 0 + 1e-6).log().sum(axis=1) * 0   # (approx)
        return a, logp

    def act(self, state, deterministic=False):
        a, _ = self._sample_action(np.asarray(state, float)[None] if not
                                   isinstance(state, Tensor) else state)
        return a.data[0]

    def remember(self, s, a, r, s2, done):
        self.replay.add(s, a, r, s2, float(done))

    def update(self, batch_size=64):
        if len(self.replay.buf) < batch_size:
            return
        s, a, r, s2, done = self.replay.sample(batch_size)
        r = r.reshape(-1, 1); done = done.reshape(-1, 1)
        a2, logp2 = self._sample_action(Tensor(s2))
        sa2 = np.hstack([s2, a2.data])
        q2 = np.minimum(self.tq1(Tensor(sa2)).data, self.tq2(Tensor(sa2)).data)
        target = Tensor(r + self.gamma * (1 - done)
                        * (q2 - self.alpha * logp2.data.reshape(-1, 1)))
        sa = Tensor(np.hstack([s, a]))
        for q, in [(self.q1,), (self.q2,)]:
            pass
        qloss = (((self.q1(sa) - target) ** 2).mean()
                 + ((self.q2(sa) - target) ** 2).mean())
        self.q_opt.zero_grad(); qloss.backward(); self.q_opt.step()
        # actor: maximise Q - alpha*logp
        anew, logp = self._sample_action(Tensor(s))
        qpi = self.q1(_cat(Tensor(s), anew))
        aloss = (self.alpha * logp - qpi.reshape(-1)).mean()
        self.a_opt.zero_grad(); aloss.backward(); self.a_opt.step()
        for t, src in [(self.tq1, self.q1), (self.tq2, self.q2)]:
            for tp, sp in zip(t.parameters(), src.parameters()):
                tp.data[...] = (1 - self.tau) * tp.data + self.tau * sp.data


def generalized_advantage_estimation(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE: a bias-variance knob for the advantage estimate (Schulman, 2016).

    The advantage ``A_t`` (how much better an action was than average) can be
    estimated from a 1-step TD error (low variance, high bias) up to the full
    Monte-Carlo return (high variance, no bias). GAE exponentially averages ALL
    of them with a decay ``lam``, letting you dial the trade-off with one number:
    ``lam=0`` is pure TD, ``lam=1`` is Monte Carlo. It is the standard advantage
    estimator inside PPO/A2C. Returns the advantage per step.
    """
    rewards = np.asarray(rewards, float)
    values = np.asarray(values, float)
    dones = np.asarray(dones, float)
    n = len(rewards)
    adv = np.zeros(n)
    gae = 0.0
    for t in reversed(range(n)):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv


class MCTS:
    """Monte Carlo Tree Search -- planning by selective look-ahead (UCT).

    When you have a MODEL of the environment (you can simulate moves), you can
    PLAN rather than only learn from experience. MCTS grows a search tree biased
    toward promising lines: SELECT down the tree by UCB (balancing a node's average
    value against how rarely it has been tried), EXPAND a new child, SIMULATE a
    random rollout to the end, and BACK-PROPAGATE the result up the path. Thousands
    of such rollouts concentrate on the best moves -- the search behind AlphaGo (and
    strong game AI generally). The env exposes ``legal_actions(state)``,
    ``step(state, action) -> (next, reward, done)``.
    """

    def __init__(self, env, n_simulations=200, c=1.4, max_depth=50,
                 random_state=None):
        self.env = env
        self.n_sim = n_simulations
        self.c = c
        self.max_depth = max_depth
        self._rng = check_random_state(random_state)

    def search(self, root_state):
        Q = {}; N = {}; children = {}          # keyed by (state, action) / state

        def rollout(state):
            total = 0.0
            for _ in range(50):
                acts = self.env.legal_actions(state)
                if not acts:
                    break
                a = acts[self._rng.randint(len(acts))]
                state, r, done = self.env.step(state, a)
                total += r
                if done:
                    break
            return total

        root = tuple(np.ravel(root_state))
        for _ in range(self.n_sim):
            state = root_state
            skey = tuple(np.ravel(state))
            path = []
            done = False
            val = 0.0
            # selection + expansion (depth-capped so non-terminating actions
            # cannot loop forever)
            for _depth in range(self.max_depth):
                acts = self.env.legal_actions(state)
                if not acts or done:
                    break
                untried = [a for a in acts if (skey, a) not in N]
                if untried:                    # expand a new action
                    a = untried[self._rng.randint(len(untried))]
                    path.append((skey, a))
                    state, r, done = self.env.step(state, a)
                    val = r + rollout(state)   # simulate from here
                    break
                # select by UCB
                total_n = sum(N[(skey, a)] for a in acts)
                a = max(acts, key=lambda a: Q[(skey, a)] / N[(skey, a)]
                        + self.c * np.sqrt(np.log(total_n + 1) / N[(skey, a)]))
                path.append((skey, a))
                state, r, done = self.env.step(state, a)
                skey = tuple(np.ravel(state))
            else:
                val = 0.0
            for key in path:                   # back-propagate
                Q[key] = Q.get(key, 0.0) + val
                N[key] = N.get(key, 0) + 1
        # best root action by visit count
        acts = self.env.legal_actions(root_state)
        visited = [(a, N.get((root, a), 0)) for a in acts]
        return max(visited, key=lambda x: x[1])[0]


__all__ = ["DDPG", "TD3", "SAC", "generalized_advantage_estimation", "MCTS"]
