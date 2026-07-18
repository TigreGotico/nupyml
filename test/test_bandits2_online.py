"""H12: bandit/planning v2 (linear Thompson, combinatorial, sleeping, Dyna-Q) and
online statistics (covariance, quantile).

Each bandit must converge to the right choice; Dyna-Q must learn the optimal chain
policy from few real steps; the streaming statistics must match their batch
counterparts in one pass.
"""
import numpy as np
import pytest

from nupyml.rl import (LinearThompsonSampling, CombinatorialBandit,
                      SleepingBandit, DynaQ)
from nupyml.streaming import OnlineCovariance, OnlineQuantile


def test_linear_thompson_learns_to_pick_the_best_arm():
    rng = np.random.RandomState(0)
    d = 4
    theta = rng.randn(d)
    lts = LinearThompsonSampling(d, random_state=0)
    for _ in range(300):
        ctx = rng.randn(3, d)
        a = lts.select(ctx)
        lts.update(ctx[a], ctx[a] @ theta + 0.1 * rng.randn())
    hits = 0
    for _ in range(200):
        ctx = rng.randn(3, d)
        hits += lts.select(ctx) == np.argmax(ctx @ theta)
    assert hits / 200 > 0.85                           # usually picks the best arm


def test_combinatorial_bandit_finds_the_best_subset():
    rng = np.random.RandomState(0)
    means = np.array([0.9, 0.8, 0.3, 0.2, 0.1])         # arms 0,1 are best
    cb = CombinatorialBandit(5, 2, random_state=0)
    for _ in range(300):
        arms = cb.select()
        cb.update(arms, means[arms] + 0.05 * rng.randn(len(arms)))
    assert sorted(cb.select().tolist()) == [0, 1]


def test_sleeping_bandit_picks_best_available():
    rng = np.random.RandomState(0)
    m = np.array([0.9, 0.5, 0.4, 0.3])
    sb = SleepingBandit(4, random_state=0)
    for _ in range(400):
        avail = [a for a in range(4) if rng.rand() > 0.3] or [0]
        a = sb.select(avail)
        sb.update(a, m[a] + 0.05 * rng.randn())
    assert sb.select([0, 1, 2, 3]) == 0                 # arm 0 is best
    assert sb.select([1, 2, 3]) == 1                    # best AVAILABLE when 0 sleeps


def test_dyna_q_learns_chain_policy():
    rng = np.random.RandomState(0)
    dq = DynaQ(3, 2, n_planning=20, random_state=0)     # 0 -> 1 -> 2(goal)
    for _ in range(50):
        s = 0
        for _ in range(10):
            a = dq.act(s)
            s2 = min(2, s + 1) if a == 1 else max(0, s - 1)
            r = 1.0 if s2 == 2 else 0.0
            dq.update(s, a, r, s2)
            s = s2
            if s == 2:
                break
    assert dq.Q[0].argmax() == 1 and dq.Q[1].argmax() == 1   # move right toward goal


def test_online_covariance_matches_batch():
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal([5, 5], [[2, 1], [1, 3]], 2000)
    oc = OnlineCovariance(2).update_batch(X)
    assert np.allclose(oc.covariance(), np.cov(X, rowvar=False), atol=1e-6)
    assert np.allclose(oc.mean, X.mean(axis=0), atol=1e-6)


def test_online_quantile_matches_true_quantile():
    rng = np.random.RandomState(0)
    vals = rng.randn(5000)
    oq = OnlineQuantile(0.9).update_batch(vals)
    assert abs(oq.quantile() - np.quantile(vals, 0.9)) < 0.1
    med = OnlineQuantile(0.5).update_batch(vals)
    assert abs(med.quantile() - np.median(vals)) < 0.1
