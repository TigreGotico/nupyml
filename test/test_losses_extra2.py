"""H3: loss zoo v2 -- gradient checks plus the property each loss advertises."""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml import nn


def _gc(fn, x, *a):
    t = Tensor(x.copy(), requires_grad=True)
    fn(t, *a).backward()
    ana = t.grad.copy()
    num = np.zeros_like(x)
    e = 1e-5
    for idx in np.ndindex(*x.shape):
        xp = x.copy(); xp[idx] += e
        xm = x.copy(); xm[idx] -= e
        num[idx] = (fn(Tensor(xp), *a).data - fn(Tensor(xm), *a).data) / (2 * e)
    return np.max(np.abs(ana - num))


def test_gradients_of_all_v2_losses():
    rng = np.random.RandomState(0)
    logits, y = rng.randn(6, 4), rng.randint(0, 4, 6)
    assert _gc(nn.SymmetricCrossEntropy(), logits, y) < 1e-4
    assert _gc(nn.GeneralizedCrossEntropy(0.7), logits, y) < 1e-4
    zb, yb = rng.randn(20), rng.randint(0, 2, 20).astype(float)
    assert _gc(nn.AsymmetricLoss(), zb, yb) < 1e-4
    assert _gc(nn.CharbonnierLoss(), rng.randn(15), rng.randn(15)) < 1e-4
    pos = np.abs(rng.randn(15)) * 5
    assert _gc(nn.MSLELoss(), np.abs(rng.randn(15)) * 5, pos) < 1e-4
    assert _gc(nn.SMAPELoss(), np.abs(rng.randn(15)) + 1, np.abs(rng.randn(15)) + 1) < 1e-4
    assert _gc(nn.ListMLELoss(), rng.randn(8), rng.randint(0, 3, 8)) < 1e-4


def test_charbonnier_between_l1_and_l2():
    ch = nn.CharbonnierLoss(eps=1e-3)
    large = ch(Tensor(np.array([100.0])), np.array([0.0])).data
    assert abs(large - 100.0) < 0.1                    # ~ L1 for large errors


def test_msle_penalises_relative_error():
    msle = nn.MSLELoss()
    # 10 vs 11 (10% off) costs about the same as 100 vs 110 (10% off)
    small = msle(Tensor(np.array([10.0])), np.array([11.0])).data
    big = msle(Tensor(np.array([100.0])), np.array([110.0])).data
    assert abs(small - big) < 0.2 * small + 1e-6


def test_smape_is_bounded():
    smape = nn.SMAPELoss()
    val = smape(Tensor(np.array([1.0, 100.0, 0.01])), np.array([50.0, 1.0, 5.0])).data
    assert 0 <= val <= 2


def test_listmle_minimised_by_correct_order():
    rel = np.array([3, 2, 1, 0], float)
    loss = nn.ListMLELoss()
    good = loss(Tensor(np.array([4.0, 3.0, 2.0, 1.0])), rel).data   # matches order
    bad = loss(Tensor(np.array([1.0, 2.0, 3.0, 4.0])), rel).data    # reversed
    assert good < bad


def test_vicreg_lower_when_views_agree_and_prevents_collapse():
    rng = np.random.RandomState(0)
    za = rng.randn(16, 8)
    same = nn.VICRegLoss()(Tensor(za), Tensor(za + 0.01)).data
    diff = nn.VICRegLoss()(Tensor(za), Tensor(rng.randn(16, 8))).data
    assert same < diff
    # a COLLAPSED (constant) representation incurs a large variance penalty
    collapsed = np.ones((16, 8)) * 0.5
    coll = nn.VICRegLoss()(Tensor(collapsed), Tensor(collapsed)).data
    assert coll > same


def test_symmetric_ce_lower_for_correct_predictions():
    y = np.array([0, 1, 2])
    correct = np.array([[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]])
    wrong = np.array([[0, 9.0, 0], [0, 0, 9.0], [9.0, 0, 0]])
    sce = nn.SymmetricCrossEntropy()
    assert sce(Tensor(correct), y).data < sce(Tensor(wrong), y).data
