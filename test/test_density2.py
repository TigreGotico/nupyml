"""H9: masked autoregressive flow, k-NN density, conditional KDE.

MAF must assign higher likelihood to in-distribution than out-of-distribution
data; k-NN density must be higher in dense regions; conditional KDE must recover
a bimodal p(y|x) that a mean-regressor would miss.
"""
import numpy as np
import pytest

from nupyml.nn import MAF
from nupyml.neighbors import KNNDensity
from nupyml.nonparametric import ConditionalKDE


def test_maf_scores_in_distribution_higher():
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal([1, -1], [[1, 0.7], [0.7, 1]], 400)
    maf = MAF(2, n_transforms=4, hidden=(32, 32), n_epochs=150,
              random_state=0).fit(X)
    ll_in = maf.score_samples(X).mean()
    ll_out = maf.score_samples(rng.uniform(-6, 6, (200, 2))).mean()
    assert np.isfinite(ll_in)
    assert ll_in > ll_out                              # exact-likelihood density


def test_maf_training_reduces_nll():
    rng = np.random.RandomState(1)
    X = rng.randn(300, 2) @ np.array([[1.0, 0.5], [0.0, 1.0]])
    maf = MAF(2, n_transforms=3, n_epochs=5, random_state=0)
    before = -maf.log_prob(X).data.mean()
    maf.fit(X)
    after = -maf.log_prob(X).data.mean()
    assert after < before


def test_knn_density_is_higher_in_dense_regions():
    rng = np.random.RandomState(0)
    data = rng.randn(600, 2)                            # dense at the origin
    kd = KNNDensity(k=10).fit(data)
    center = kd.score_samples(np.array([[0.0, 0.0]]))[0]
    edge = kd.score_samples(np.array([[4.0, 4.0]]))[0]
    assert center > edge                               # denser -> higher log-density


def test_conditional_kde_recovers_bimodal_distribution():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (500, 1))
    # given x, y is +2 OR -2 -> the conditional MEAN (0) never occurs
    y = np.where(rng.rand(500) < 0.5, 2.0, -2.0) + 0.2 * rng.randn(500)
    ckde = ConditionalKDE(bandwidth_x=0.3, bandwidth_y=0.4).fit(X, y)
    grid = np.linspace(-4, 4, 200)
    dens = ckde.pdf([0.0], grid)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(dens)
    modes = np.sort(grid[peaks])
    assert len(modes) >= 2                             # bimodal, not unimodal
    assert modes[0] < -1 and modes[-1] > 1             # near -2 and +2
    # the conditional mean sits between the modes (where no data is)
    assert abs(ckde.conditional_mean(np.array([[0.0]]))[0]) < 0.5
