"""Losses v6: multivariate proper scores, self-distillation, distributional box
regression, region mutual information, and refined IoU box losses.

Five more losses. The energy score is the multivariate CRPS for probabilistic vector
forecasts. The DINO loss trains a student to match a sharpened teacher. Distribution
focal loss learns a DISTRIBUTION over box offsets instead of a point. Region mutual
information ties a segmentation to its target through mutual information. EIoU/SIoU
refine the IoU box-regression family.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _norm(v, axis):
    return ((v * v).sum(axis=axis) + 1e-12) ** 0.5


def energy_score(forecast_samples, y):
    """The multivariate CRPS -- a proper score for VECTOR forecasts (Gneiting, 2008).

    CRPS scores a scalar predictive distribution; the energy score generalises it to
    vectors, so it can judge a probabilistic forecast of several correlated quantities
    at once (a wind field, a portfolio of returns). In its ensemble form it is
    ``mean||X_i - y|| - 0.5 mean||X_i - X_j||`` -- reward for the ensemble being close
    to the outcome, penalty for being spread out -- and it is PROPER, minimised in
    expectation only by the true joint distribution, so a model cannot cheat by
    misstating correlations. Fully differentiable. ``forecast_samples`` is
    ``(batch, n_samples, dim)``; ``y`` is ``(batch, dim)``.
    """
    X = Tensor._wrap(forecast_samples)
    B, n, d = X.shape
    y = Tensor(np.asarray(y, float)).reshape(B, 1, d)
    term1 = _norm(X - y, axis=2).mean(axis=1)             # E||X - y||
    xi = X.reshape(B, n, 1, d)
    xj = X.reshape(B, 1, n, d)
    term2 = _norm(xi - xj, axis=3).mean(axis=2).mean(axis=1)   # E||X - X'||
    return (term1 - 0.5 * term2).mean()


class DINOLoss(Module):
    """Self-distillation: match a SHARPENED teacher (Caron et al., 2021).

    DINO trains a network with no labels by distillation from itself: a "student" is
    pushed to match a "teacher" (an exponential moving average of the student) on
    different augmented views. Two tricks stop it collapsing to a constant -- the
    teacher output is SHARPENED (a low softmax temperature makes it confident) and
    CENTERED (a running mean is subtracted so no dimension dominates) -- and the
    student, at a higher temperature, is trained by cross-entropy to that target. The
    result is a powerful self-supervised representation. ``center`` is the running
    teacher mean.
    """

    def __init__(self, n_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.center = np.zeros((1, n_dim))

    def forward(self, student_out, teacher_out):
        s = Tensor._wrap(student_out)
        t = np.asarray(Tensor._wrap(teacher_out).data, float)
        # teacher: centre then sharpen (low temperature), as a fixed target
        teacher_soft = _softmax((t - self.center) / self.teacher_temp)
        logp = F.log_softmax(s * (1.0 / self.student_temp), axis=1)
        loss = -(Tensor(teacher_soft) * logp).sum(axis=1).mean()
        # update the centre with the batch mean (prevents collapse)
        self.center = (self.center_momentum * self.center
                       + (1 - self.center_momentum) * t.mean(axis=0, keepdims=True))
        return loss


def distribution_focal_loss(logits, target, bin_edges):
    """Learn a DISTRIBUTION over box offsets, not a point (Li et al., 2020).

    A detector usually regresses a box edge to a single number, which cannot express
    that a blurry or occluded boundary is UNCERTAIN. Distribution focal loss discretises
    the offset into bins and predicts a probability over them, so the model can be
    sharp on clear edges and spread on ambiguous ones (and the expectation is the
    prediction). For a continuous target between two adjacent bins, it is the cross-
    entropy that pushes probability onto exactly those two bins in proportion to the
    target's position between them. ``logits`` is ``(batch, n_bins)``.
    """
    logits = Tensor._wrap(logits)
    edges = np.asarray(bin_edges, float)
    target = np.asarray(target, float).ravel()
    logp = F.log_softmax(logits, axis=1)
    n = len(target)
    left = np.clip(np.searchsorted(edges, target, "right") - 1, 0, len(edges) - 2)
    right = left + 1
    wl = (edges[right] - target) / (edges[right] - edges[left])
    wr = 1.0 - wl
    onehot_l = np.zeros((n, logits.shape[1])); onehot_l[np.arange(n), left] = 1
    onehot_r = np.zeros((n, logits.shape[1])); onehot_r[np.arange(n), right] = 1
    target_dist = Tensor(onehot_l) * Tensor(wl.reshape(-1, 1)) \
        + Tensor(onehot_r) * Tensor(wr.reshape(-1, 1))
    return -(target_dist * logp).sum(axis=1).mean()


def region_mutual_information(pred, target):
    """Tie a segmentation to its target through MUTUAL INFORMATION (Zhao et al., 2019).

    Per-pixel losses treat pixels independently and ignore that neighbouring pixels are
    correlated -- the STRUCTURE a good segmentation must capture. Region mutual
    information maximises the mutual information between the predicted and the true
    label fields, which rewards getting the joint pattern right, not just each pixel.
    This soft version reads the 2x2 joint distribution of (predicted probability, true
    label) over the image and returns the negative mutual information, so minimising it
    raises the dependence between prediction and target. ``pred`` in [0,1], ``target``
    binary.
    """
    p = Tensor._wrap(pred).reshape(-1)
    y = np.asarray(target, float).ravel()
    yt = Tensor(y)
    n = len(y)
    # soft 2x2 joint distribution of (pred, target)
    p11 = (p * yt).sum() * (1.0 / n)
    p10 = (p * (1.0 - yt)).sum() * (1.0 / n)
    p01 = ((1.0 - p) * yt).sum() * (1.0 / n)
    p00 = ((1.0 - p) * (1.0 - yt)).sum() * (1.0 / n)
    pp1 = p.sum() * (1.0 / n); pp0 = 1.0 - pp1            # marginals
    py1 = float(y.mean()); py0 = 1.0 - py1
    mi = (_mi_term(p11, pp1, py1) + _mi_term(p10, pp1, py0)
          + _mi_term(p01, pp0, py1) + _mi_term(p00, pp0, py0))
    return -mi                                            # minimise -> maximise MI


def _mi_term(pjoint, pa, pb):
    return pjoint * (pjoint / (pa * pb + 1e-9) + 1e-9).log()


class BoxRegressionLoss2(Module):
    """Refined IoU box-regression losses: EIoU and SIoU (Zhang 2021; Gevorgyan 2022).

    CIoU couples the width/height into a single aspect-ratio term, whose gradient can
    push both the wrong way. ``eiou`` (Efficient IoU) instead penalises the width and
    height errors SEPARATELY, normalised by the enclosing box -- cleaner gradients and
    faster convergence. ``siou`` (SCYLLA IoU) adds an ANGLE cost that first aligns the
    boxes along the axis of their centre offset before closing the distance, which
    empirically trains detectors faster still. Boxes are ``(x1, y1, x2, y2)``; returns
    the mean loss.
    """

    def __init__(self, mode="eiou"):
        super().__init__()
        assert mode in ("eiou", "siou")
        self.mode = mode

    def forward(self, pred, target):
        pred = np.asarray(Tensor._wrap(pred).data, float)
        target = np.asarray(Tensor._wrap(target).data, float)
        px1, py1, px2, py2 = pred.T
        tx1, ty1, tx2, ty2 = target.T
        iw = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
        ih = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
        inter = iw * ih
        ap = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
        at = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
        iou = inter / (ap + at - inter + 1e-9)
        cx1 = np.minimum(px1, tx1); cy1 = np.minimum(py1, ty1)
        cx2 = np.maximum(px2, tx2); cy2 = np.maximum(py2, ty2)
        cw, ch = cx2 - cx1, cy2 - cy1
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        cdx, cdy = pcx - tcx, pcy - tcy
        diag = cw ** 2 + ch ** 2 + 1e-9
        if self.mode == "eiou":
            # centre distance + separate width and height distances (all normalised)
            dist = (cdx ** 2 + cdy ** 2) / diag
            wdist = ((px2 - px1) - (tx2 - tx1)) ** 2 / (cw ** 2 + 1e-9)
            hdist = ((py2 - py1) - (ty2 - ty1)) ** 2 / (ch ** 2 + 1e-9)
            loss = 1 - iou + dist + wdist + hdist
        else:                                             # siou
            sigma = np.sqrt(cdx ** 2 + cdy ** 2) + 1e-9
            sin_alpha = np.abs(cdy) / sigma
            angle_cost = 1 - 2 * np.sin(np.arcsin(np.clip(sin_alpha, 0, 1))
                                        - np.pi / 4) ** 2
            rho_x = (cdx / (cw + 1e-9)) ** 2; rho_y = (cdy / (ch + 1e-9)) ** 2
            gamma = 2 - angle_cost
            dist_cost = 2 - np.exp(-gamma * rho_x) - np.exp(-gamma * rho_y)
            loss = 1 - iou + 0.5 * (dist_cost + angle_cost * 0.1)
        return Tensor(float(loss.mean()))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


__all__ = ["energy_score", "DINOLoss", "distribution_focal_loss",
           "region_mutual_information", "BoxRegressionLoss2"]
