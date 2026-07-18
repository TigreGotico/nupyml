"""L9: losses v6 -- energy score, DINO, distribution focal, region-MI, EIoU/SIoU.

The energy score rewards a sharp vector-ensemble forecast; DINO is minimal when the
student matches the teacher; distribution focal loss is low when probability
concentrates on the target's bins; region MI is lower (better) for a prediction
correlated with the mask; EIoU/SIoU reward box overlap.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (energy_score, DINOLoss, distribution_focal_loss,
                       region_mutual_information, BoxRegressionLoss2)


def test_energy_score_rewards_sharp_ensemble():
    y = np.array([[1.0, 2.0]])
    sharp = Tensor(np.array([[[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]]))
    dispersed = Tensor(np.array([[[-2., 1], [4., 3], [0., 5]]]))
    assert energy_score(sharp, y).data < energy_score(dispersed, y).data
    t = Tensor(np.array([[[1., 2], [1.1, 2.1], [0.9, 1.9]]]), requires_grad=True)
    energy_score(t, y).backward()
    assert t.grad is not None


def test_dino_loss_minimal_when_student_matches_teacher():
    teacher = np.array([[5., 0, 0, 0, 0]])
    match = Tensor(np.array([[5., 0, 0, 0, 0]]))
    mismatch = Tensor(np.array([[0., 0, 0, 0, 5]]))
    assert DINOLoss(5)(match, teacher).data < DINOLoss(5)(mismatch, teacher).data
    t = Tensor(np.array([[5., 0, 0, 0, 0]]), requires_grad=True)
    DINOLoss(5)(t, teacher).backward()
    assert t.grad is not None


def test_distribution_focal_loss_concentrates_on_target_bins():
    edges = np.array([0., 1, 2, 3, 4])
    target = np.array([1.5])                              # between bins 1 and 2
    good = Tensor(np.array([[0., 5, 5, 0, 0]]))
    bad = Tensor(np.array([[5., 0, 0, 0, 5]]))
    assert distribution_focal_loss(good, target, edges).data < \
        distribution_focal_loss(bad, target, edges).data
    t = Tensor(np.array([[0., 5, 5, 0, 0]]), requires_grad=True)
    distribution_focal_loss(t, target, edges).backward()
    assert t.grad is not None


def test_region_mi_prefers_correlated_prediction():
    mask = np.array([1, 1, 0, 0, 1, 0, 1, 0.])
    correlated = Tensor(np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1]))
    uninformative = Tensor(np.full(8, 0.5))
    # lower (more negative) region-MI loss = higher mutual information
    assert region_mutual_information(correlated, mask).data < \
        region_mutual_information(uninformative, mask).data
    t = Tensor(np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1]),
               requires_grad=True)
    region_mutual_information(t, mask).backward()
    assert t.grad is not None


@pytest.mark.parametrize("mode", ["eiou", "siou"])
def test_box_regression2_rewards_overlap(mode):
    box = np.array([[0., 0, 10, 10]])
    close = np.array([[1., 1, 11, 11]])
    far = np.array([[20., 20, 30, 30]])
    bl = BoxRegressionLoss2(mode)
    same = float(bl(Tensor(box), Tensor(box)).data)
    near = float(bl(Tensor(box), Tensor(close)).data)
    disjoint = float(bl(Tensor(box), Tensor(far)).data)
    assert same == pytest.approx(0.0, abs=1e-6)
    assert 0 < near < disjoint
