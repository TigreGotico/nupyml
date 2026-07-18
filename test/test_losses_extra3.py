"""I7: losses v3 -- Circle, MultiSimilarity, Angular, box regression, SSIM, Tukey.

Each is held to the property that motivates it: the metric losses rank a
well-separated batch below a badly-separated one; the box losses reward overlap
and (for GIoU/DIoU/CIoU) keep pulling disjoint boxes together; SSIM is zero for
identical images; Tukey's biweight caps an outlier's contribution.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (CircleLoss, MultiSimilarityLoss, AngularLoss,
                       BoxRegressionLoss, SSIMLoss, TukeyBiweightLoss)


def test_circle_loss_prefers_separated_pairs_and_backprops():
    loss = CircleLoss()
    good = loss(Tensor(np.array([0.9, 0.95])), Tensor(np.array([0.1, 0.05]))).data
    bad = loss(Tensor(np.array([0.2, 0.3])), Tensor(np.array([0.8, 0.7]))).data
    assert good < bad
    sp = Tensor(np.array([0.5, 0.6]), requires_grad=True)
    sn = Tensor(np.array([0.4, 0.3]), requires_grad=True)
    loss(sp, sn).backward()
    assert sp.grad is not None and np.all(np.isfinite(sp.grad))


def test_multisimilarity_positive_and_backprops():
    loss = MultiSimilarityLoss()
    s = Tensor(np.array([0.9, 0.8, 0.2, 0.1]), requires_grad=True)
    out = loss(s, np.array([1, 1, 0, 0]))
    assert out.data > 0
    out.backward()
    assert s.grad is not None


def test_angular_loss_ranks_easy_below_hard_triplet():
    rng = np.random.RandomState(0)
    a = rng.randn(8, 4)
    p = a + 0.01 * rng.randn(8, 4)                     # positive ~ anchor
    n = rng.randn(8, 4)                                # negative far
    loss = AngularLoss()
    easy = loss(Tensor(a), Tensor(p), Tensor(n)).data  # correct roles
    hard = loss(Tensor(a), Tensor(n), Tensor(p)).data  # positive/negative swapped
    assert easy < hard


@pytest.mark.parametrize("mode", ["iou", "giou", "diou", "ciou"])
def test_box_loss_zero_for_identical_and_grows_with_separation(mode):
    loss = BoxRegressionLoss(mode)
    box = np.array([[0., 0, 10, 10]])
    close = np.array([[1., 1, 11, 11]])
    far = np.array([[20., 20, 30, 30]])
    same = float(loss(Tensor(box), Tensor(box)).data)
    near = float(loss(Tensor(box), Tensor(close)).data)
    disjoint = float(loss(Tensor(box), Tensor(far)).data)
    assert same == pytest.approx(0.0, abs=1e-6)
    assert 0 < near < disjoint


def test_giou_diou_penalise_disjoint_beyond_plain_iou():
    box = np.array([[0., 0, 10, 10]])
    far = np.array([[20., 20, 30, 30]])
    iou_l = float(BoxRegressionLoss("iou")(Tensor(box), Tensor(far)).data)
    giou_l = float(BoxRegressionLoss("giou")(Tensor(box), Tensor(far)).data)
    diou_l = float(BoxRegressionLoss("diou")(Tensor(box), Tensor(far)).data)
    # plain IoU saturates at 1 for any disjoint pair; the others keep a gradient
    assert iou_l == pytest.approx(1.0, abs=1e-6)
    assert giou_l > 1.0 and diou_l > 1.0


def test_ssim_zero_for_identical_and_grows_with_noise():
    rng = np.random.RandomState(0)
    img = rng.rand(16, 16)
    loss = SSIMLoss()
    assert float(loss(Tensor(img), Tensor(img)).data) == pytest.approx(0.0, abs=1e-6)
    noisy = img + 0.3 * rng.rand(16, 16)
    assert loss(Tensor(img), Tensor(noisy)).data > 0.0


def test_tukey_biweight_caps_outlier_influence():
    rng = np.random.RandomState(0)
    clean = np.zeros(100)
    pred = 0.5 * np.ones(100)
    loss = TukeyBiweightLoss(c=4.685)
    base = float(loss(Tensor(pred), Tensor(clean)).data)
    with_outlier = pred.copy(); with_outlier[0] = 100.0
    outlier = float(loss(Tensor(with_outlier), Tensor(clean)).data)
    # one gross outlier adds at most a single capped bin (c^2/6 / N), not infinity
    cap = (4.685 ** 2 / 6.0) / 100
    assert outlier - base < cap + 1e-6
    # gradient flows for the inliers
    p = Tensor(pred, requires_grad=True)
    loss(p, Tensor(clean)).backward()
    assert p.grad is not None
