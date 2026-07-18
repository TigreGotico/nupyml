"""H11: neural recommenders (NeuralCF, DeepFM) and deep feature synthesis.

The neural recommenders must beat the global-mean baseline on a low-rank rating
matrix; deep feature synthesis must compute the correct per-group aggregates and
fall back gracefully for unseen entities.
"""
import numpy as np
import pytest

from nupyml.recommend import NeuralCF, DeepFM
from nupyml.feature_extraction import DeepFeatureSynthesis


@pytest.fixture
def ratings():
    rng = np.random.RandomState(0)
    nu, ni, k = 50, 40, 3
    full = (rng.rand(nu, k) @ rng.rand(k, ni)) * 4 + 1
    tr, te = [], []
    for u in range(nu):
        for i in range(ni):
            if rng.rand() < 0.6:
                tr.append((u, i, full[u, i]))
            elif rng.rand() < 0.3:
                te.append((u, i, full[u, i]))
    return tr, te


def _rmse(model, te):
    return np.sqrt(np.mean([(model.predict(u, i) - r) ** 2 for u, i, r in te]))


def _baseline(tr, te):
    gm = np.mean([r for _, _, r in tr])
    return np.sqrt(np.mean([(gm - r) ** 2 for _, _, r in te]))


@pytest.mark.parametrize("make", [
    lambda: NeuralCF(n_factors=16, epochs=50, random_state=0),
    lambda: DeepFM(n_factors=16, epochs=60, random_state=0),
], ids=["neuralcf", "deepfm"])
def test_neural_recommender_beats_global_mean(ratings, make):
    tr, te = ratings
    model = make().fit(tr)
    assert _rmse(model, te) < _baseline(tr, te)
    # unseen indices fall back to the global mean, no crash
    assert model.predict(9999, 9999) == pytest.approx(model.global_mean_)


def test_deep_feature_synthesis_computes_group_aggregates():
    # group column 0, value column 1
    X = np.array([[0, 10], [0, 20], [1, 5], [1, 7], [1, 9], [2, 100]], float)
    dfs = DeepFeatureSynthesis(group_col=0,
                               primitives=("mean", "max", "count")).fit(X)
    T = dfs.transform(X)
    # row 0: [value=10, group-mean=15, group-max=20, group-count=2]
    assert np.allclose(T[0], [10, 15, 20, 2])
    # group 1 rows share their aggregates (mean 7, max 9, count 3)
    assert np.allclose(T[2, 1:], [7, 9, 3])


def test_deep_feature_synthesis_handles_unseen_group():
    X = np.array([[0, 10.0], [0, 20.0], [1, 5.0]])
    dfs = DeepFeatureSynthesis(group_col=0, primitives=("mean", "count")).fit(X)
    # an unseen group id falls back to the global aggregate
    out = dfs.transform(np.array([[99, 3.0]]))
    assert out.shape[1] == T_width(dfs)
    assert np.all(np.isfinite(out))


def T_width(dfs):
    return 1 + 2   # value col + (mean, count)
