"""M7: time series v6 -- k-Shape, convergent cross mapping, SPOT, Bayesian VAR,
block bootstrap.

k-Shape recovers shift/scale-invariant shape clusters; convergent cross mapping
detects the causal direction and shows the tell-tale convergence with library
size; SPOT flags injected extremes from the fitted tail with no false alarms;
Bayesian VAR forecasts a coupled system; the block bootstrap resamples a series
without destroying its autocorrelation.
"""
import numpy as np

from nupyml.timeseries import (KShape, convergent_cross_mapping, SPOT,
                               BayesianVAR, block_bootstrap)
from nupyml.metrics import adjusted_rand_score


def test_kshape_clusters_by_shape():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 8 * np.pi, 80)
    slow = [np.sin(t * 1.0 + rng.uniform(0, 2)) * rng.uniform(0.5, 2)
            + rng.randn(80) * 0.1 for _ in range(15)]
    fast = [np.sin(t * 2.5 + rng.uniform(0, 2)) * rng.uniform(0.5, 2)
            + rng.randn(80) * 0.1 for _ in range(15)]
    X = np.array(slow + fast)
    y = np.array([0] * 15 + [1] * 15)
    # invariant to the per-series shift and amplitude scaling above
    best = max(adjusted_rand_score(y, KShape(2, random_state=s).fit(X).labels_)
               for s in range(5))
    assert best > 0.9


def test_ccm_detects_causal_direction_and_converges():
    n = 400
    x = np.zeros(n); y = np.zeros(n); x[0], y[0] = 0.4, 0.2
    for i in range(1, n):
        x[i] = x[i - 1] * (3.8 - 3.8 * x[i - 1])          # X is autonomous
        y[i] = y[i - 1] * (3.5 - 3.5 * y[i - 1] - 0.8 * x[i - 1])   # X drives Y
    # X drives Y => X is recoverable from Y's manifold, and skill converges up
    libs, skill = convergent_cross_mapping(x, y, embed_dim=3, tau=1)
    assert skill[-1] > 0.8
    assert skill[-1] > skill[0]


def test_spot_flags_extremes_from_the_tail():
    rng = np.random.RandomState(0)
    y = rng.randn(2000)
    anom = rng.choice(2000, 10, replace=False)
    y[anom] += rng.uniform(5, 9, 10)
    spot = SPOT(q=5e-3, init_quantile=0.9).fit(y)
    pred = spot.predict(y)
    assert pred[anom].mean() >= 0.9                        # recalls the extremes
    assert pred.sum() - pred[anom].sum() <= 2              # ~no false alarms


def test_bayesian_var_forecasts_coupled_series():
    rng = np.random.RandomState(0)
    T = 60
    A = np.array([[0.5, 0.2], [-0.1, 0.6]])
    Y = np.zeros((T, 2)); Y[0] = [1, 0]
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + 0.2 * rng.randn(2)
    bvar = BayesianVAR(p=1, lam=0.2).fit(Y)
    fc = bvar.forecast(5)
    assert fc.shape == (5, 2)
    assert np.all(np.isfinite(fc))
    # a stable system decays toward zero, not explode
    assert np.abs(fc[-1]).max() < np.abs(Y).max() + 1


def test_block_bootstrap_preserves_autocorrelation():
    rng = np.random.RandomState(0)
    ar = np.zeros(300)
    for i in range(1, 300):
        ar[i] = 0.7 * ar[i - 1] + rng.randn()

    def acf1(s):
        return np.corrcoef(s[:-1], s[1:])[0, 1]

    boots = block_bootstrap(ar, block_size=20, n_boot=200, statistic=acf1,
                            random_state=0)
    assert boots.shape == (200,)
    # blocks keep the lag-1 correlation; independent resampling would kill it
    assert abs(boots.mean() - acf1(ar)) < 0.15
    assert boots.mean() > 0.4
