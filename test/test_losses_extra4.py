"""J10: losses v4 -- soft-DTW, evidential, seesaw, RankNet, boundary.

Soft-DTW ranks a warped copy below an unrelated series and is differentiable;
the evidential loss is lower for a correct high-evidence prediction and reports
high uncertainty when evidence is scarce; RankNet scores a correctly-ordered batch
below a mis-ordered one; the boundary loss rewards predicting inside the object.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (soft_dtw, EvidentialLoss, SeesawLoss, RankNetLoss,
                       BoundaryLoss)


def test_soft_dtw_ranks_warp_below_unrelated_and_backprops():
    a = np.sin(np.linspace(0, 3, 15))
    warp = np.sin(np.linspace(0.3, 3.3, 15))          # same shape, shifted
    other = np.cos(np.linspace(0, 3, 15)) * 3
    d_warp = float(soft_dtw(a, warp, gamma=0.5).data)
    d_other = float(soft_dtw(a, other, gamma=0.5).data)
    assert d_warp < d_other                            # warping tolerated
    t = Tensor(a, requires_grad=True)
    soft_dtw(t, Tensor(other), gamma=0.5).backward()
    assert t.grad is not None and np.all(np.isfinite(t.grad))


def test_evidential_loss_and_uncertainty():
    ev = EvidentialLoss(3)
    correct = np.array([[10., 0, 0]])
    wrong = np.array([[0., 0, 10]])
    assert ev(Tensor(correct), [0]).data < ev(Tensor(wrong), [0]).data
    # scarce evidence -> high vacuity; abundant evidence -> low
    assert ev.uncertainty(np.array([[0.1, 0, 0.1]]))[0] > 0.8
    assert ev.uncertainty(np.array([[20., 0, 0]]))[0] < 0.2
    t = Tensor(correct, requires_grad=True)
    ev(t, [0]).backward()
    assert t.grad is not None


def test_seesaw_loss_runs_and_backprops():
    ss = SeesawLoss(class_counts=[1000, 10])
    logits = Tensor(np.array([[2.0, 1.0]]), requires_grad=True)
    loss = ss(logits, [1])                             # a rare-class sample
    assert loss.data > 0
    loss.backward()
    assert logits.grad is not None


def test_ranknet_orders_pairs():
    rn = RankNetLoss()
    good = rn(Tensor(np.array([3., 4., 5.])),
              Tensor(np.array([1., 2., 3.])), [1, 1, 1])
    bad = rn(Tensor(np.array([1., 2., 3.])),
             Tensor(np.array([3., 4., 5.])), [1, 1, 1])
    assert good.data < bad.data                        # correct order -> lower loss
    t = Tensor(np.array([1., 2., 3.]), requires_grad=True)
    rn(t, Tensor(np.array([3., 4., 5.])), [1, 1, 1]).backward()
    assert t.grad is not None


def test_boundary_loss_rewards_predicting_inside():
    mask = np.zeros((20, 20)); mask[5:15, 5:15] = 1
    bl = BoundaryLoss()
    inside = float(bl(Tensor(mask), mask).data)        # predict the object
    outside = float(bl(Tensor(1 - mask), mask).data)   # predict the background
    assert inside < outside
    t = Tensor(mask.astype(float), requires_grad=True)
    bl(t, mask).backward()
    assert t.grad is not None
