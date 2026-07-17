"""G6: Barlow Twins and BYOL self-supervised learning (no labels).

Both learn from augmented view pairs. Each must (a) drive its loss down and (b)
produce representations that a linear probe can classify -- WITHOUT ever seeing
the labels during training. BYOL must additionally move its EMA target and not
collapse to a constant.
"""
import numpy as np
import pytest

from nupyml import nn
from nupyml.linear_model import LogisticRegression


def _clustered(seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(3, 10) * 3
    X = np.repeat(centers, 60, axis=0) + rng.randn(180, 10)
    y = np.repeat([0, 1, 2], 60)
    return X, y, rng


def _views(X, rng):
    return X + 0.3 * rng.randn(*X.shape), X + 0.3 * rng.randn(*X.shape)


def _probe(H, y):
    return (LogisticRegression(max_iter=300).fit(H, y).predict(H) == y).mean()


def test_barlow_twins_learns_useful_representations():
    X, y, rng = _clustered()
    bt = nn.BarlowTwins(10, hidden=(32,), embed_dim=16, rng=0)
    opt = nn.Adam(bt.parameters(), lr=0.01)
    first = float(bt.loss(*_views(X, rng)).data)
    for _ in range(150):
        opt.zero_grad(); bt.loss(*_views(X, rng)).backward(); opt.step()
    assert float(bt.loss(*_views(X, rng)).data) < first        # loss decreases
    assert _probe(bt.encode(X).data, y) > 0.9                  # representation useful


def test_byol_learns_and_moves_target_without_collapse():
    X, y, rng = _clustered(seed=1)
    by = nn.BYOL(10, hidden=(32,), embed_dim=16, proj_dim=8, tau=0.9, rng=0)
    opt = nn.Adam(by.parameters(), lr=0.01)
    target0 = by._target_params()[0].data.copy()
    first = float(by.loss(*_views(X, rng)).data)
    for _ in range(150):
        opt.zero_grad(); by.loss(*_views(X, rng)).backward()
        opt.step(); by.update_target()
    assert float(by.loss(*_views(X, rng)).data) < first        # loss decreases
    assert not np.allclose(target0, by._target_params()[0].data)   # EMA moved
    H = by.encode(X).data
    assert H.std() > 1e-3                                       # not collapsed
    assert _probe(H, y) > 0.9


def test_byol_only_online_path_has_trainable_params():
    by = nn.BYOL(10, embed_dim=8, proj_dim=4, rng=0)
    # trainable = online encoder + projector + predictor; target is EMA-only
    n_online = (len(by.online_encoder.parameters())
                + len(by.online_proj.parameters())
                + len(by.predictor.parameters()))
    assert len(by.parameters()) == n_online
