"""M9: losses v7 -- SoftTriple, active-contour, ApproxNDCG, logit adjustment, MMCE.

SoftTriple computes a positive metric loss with gradients flowing to its centres;
the active-contour loss is lower for a matching mask than a mismatched one; ApproxNDCG
scores a correct ranking better than a reversed one; logit adjustment shifts logits by
the class prior; MMCE is ~0 for a calibrated model and large for a miscalibrated one.
"""
import numpy as np

from nupyml.autograd import Tensor
from nupyml.nn import (SoftTripleLoss, ActiveContourLoss, ApproxNDCGLoss,
                      LogitAdjustmentLoss, MMCELoss)


def test_soft_triple_backprops_to_centres():
    rng = np.random.RandomState(0)
    emb = Tensor(rng.randn(12, 8))
    emb = emb / (emb * emb).sum(axis=1, keepdims=True).sqrt()
    y = np.array([0, 1, 2] * 4)
    loss = SoftTripleLoss(3, 8, n_centers=2, random_state=0)
    out = loss(emb, y)
    out.backward()
    assert float(out.data) > 0
    assert loss.centers.grad is not None and np.any(loss.centers.grad != 0)


def test_approx_ndcg_prefers_correct_ranking():
    rel = np.array([3.0, 2.0, 1.0, 0.0])
    good = float(ApproxNDCGLoss(0.1)(Tensor(np.array([3.0, 2.0, 1.0, 0.0])), rel).data)
    bad = float(ApproxNDCGLoss(0.1)(Tensor(np.array([0.0, 1.0, 2.0, 3.0])), rel).data)
    assert good < bad                       # lower (more negative -NDCG) is better
    assert good <= -0.99                     # perfect order recovers NDCG = 1


def test_logit_adjustment_shifts_by_prior():
    rng = np.random.RandomState(0)
    logits = Tensor(rng.randn(10, 3))
    tgt = rng.randint(0, 3, 10)
    la = LogitAdjustmentLoss(priors=[0.8, 0.15, 0.05], tau=1.0)
    plain = LogitAdjustmentLoss(priors=[1 / 3, 1 / 3, 1 / 3], tau=1.0)
    # a skewed prior changes the loss relative to a uniform one
    assert abs(float(la(logits, tgt).data) - float(plain(logits, tgt).data)) > 1e-3


def test_active_contour_lower_for_matching_mask():
    mask = np.zeros((1, 6, 6)); mask[0, 2:4, 2:4] = 1.0
    ac = ActiveContourLoss(weight=0.5)
    match = float(ac(Tensor(mask.copy()), mask).data)
    mismatch = float(ac(Tensor(1 - mask), mask).data)
    assert match < mismatch


def _binary_logits(confs, corrects):
    rows, tgt = [], []
    for c, ok in zip(confs, corrects):
        g = np.log(c / (1 - c))                 # class-0 prob = c
        rows.append([g, 0.0]); tgt.append(0 if ok else 1)
    return Tensor(np.array(rows)), np.array(tgt)


def test_mmce_detects_miscalibration():
    # calibrated: 0.9-confident right 90%, 0.6-confident right 60%
    Lc, tc = _binary_logits([0.9] * 10 + [0.6] * 10,
                            [True] * 9 + [False] + [True] * 6 + [False] * 4)
    # miscalibrated: 0.95-confident everywhere, right only half the time
    Lm, tm = _binary_logits([0.95] * 20, [True] * 10 + [False] * 10)
    calibrated = float(MMCELoss(0.2)(Lc, tc).data)
    miscalibrated = float(MMCELoss(0.2)(Lm, tm).data)
    assert calibrated < 0.05
    assert miscalibrated > calibrated + 0.1
