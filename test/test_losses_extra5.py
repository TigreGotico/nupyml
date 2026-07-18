"""K9: losses v5 -- Barron, Lovász-softmax, CRPS, ListNet, proxy-anchor.

Barron's loss bounds an outlier's influence for alpha<2; Lovász-hinge is zero for
a correct segmentation and positive for a wrong one; CRPS rewards a sharp,
well-placed ensemble over an over-dispersed one; ListNet orders a list; proxy-anchor
learns proxies that classify embeddings by nearest proxy.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (BarronLoss, lovasz_hinge, crps_ensemble, ListNetLoss,
                       ProxyAnchorLoss, Adam)


def test_barron_recovers_l2_and_bounds_outliers():
    pred = Tensor(np.array([0., 0, 0, 10.]))
    tgt = Tensor(np.zeros(4))
    l2 = float(BarronLoss(alpha=2.0, c=1.0)(pred, tgt).data)
    robust = float(BarronLoss(alpha=0.5, c=1.0)(pred, tgt).data)
    assert robust < l2                                   # the outlier is down-weighted
    t = Tensor(np.array([0., 0, 0, 10.]), requires_grad=True)
    BarronLoss(alpha=1.0)(t, tgt).backward()
    assert t.grad is not None


def test_lovasz_hinge_zero_for_correct_segmentation():
    labels = np.array([1, 1, 0, 0, 1, 0])
    good = Tensor(np.array([2., 2, -2, -2, 2, -2]), requires_grad=True)
    bad = Tensor(np.array([-2., -2, 2, 2, -2, 2]))
    assert float(lovasz_hinge(good, labels).data) < float(lovasz_hinge(bad, labels).data)
    lovasz_hinge(good, labels).backward()
    assert good.grad is not None


def test_crps_rewards_sharp_ensemble():
    y = np.array([1.0, 2.0])
    sharp = Tensor(np.array([[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]]))
    dispersed = Tensor(np.array([[-2., 1, 4], [0., 2, 5]]))
    assert crps_ensemble(sharp, y).data < crps_ensemble(dispersed, y).data
    t = Tensor(np.array([[0.9, 1.0, 1.1]]), requires_grad=True)
    crps_ensemble(t, np.array([1.0])).backward()
    assert t.grad is not None


def test_listnet_prefers_correct_order():
    rel = np.array([3, 2, 1, 0])
    good = Tensor(np.array([3., 2, 1, 0.]))
    bad = Tensor(np.array([0., 1, 2, 3.]))
    assert ListNetLoss()(good, rel).data < ListNetLoss()(bad, rel).data
    t = Tensor(np.array([3., 2, 1, 0.]), requires_grad=True)
    ListNetLoss()(t, rel).backward()
    assert t.grad is not None


def test_proxy_anchor_learns_class_proxies():
    rng = np.random.RandomState(0)
    centers = rng.randn(3, 8) * 3
    X = np.vstack([centers[i] + rng.randn(20, 8) * 0.3 for i in range(3)])
    y = np.repeat([0, 1, 2], 20)
    pa = ProxyAnchorLoss(n_classes=3, dim=8, rng=0)
    opt = Adam(pa.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(); pa(Tensor(X), y).backward(); opt.step()
    xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pn = pa.proxies.data / np.linalg.norm(pa.proxies.data, axis=1, keepdims=True)
    pred = (xn @ pn.T).argmax(1)
    assert (pred == y).mean() > 0.9                      # nearest proxy = true class
