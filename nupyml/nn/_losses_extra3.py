"""Losses v3: pair-weighting metric losses, box-regression losses, structural
similarity, and robust regression.

Two threads here. The metric-learning losses (Circle, MultiSimilarity, Angular)
all improve on triplet loss by WEIGHTING pairs instead of using a hard margin --
hard pairs pull harder, easy pairs are ignored. The regression losses cover
targets the squared error handles badly: overlapping BOXES (GIoU/DIoU/CIoU),
image STRUCTURE (SSIM), and OUTLIERS (Tukey's biweight, which caps their
influence entirely).
"""
import numpy as np

from ..autograd import Tensor
from .module import Module
from .metric_losses import _l2_normalize


def _logsumexp(x, axis=None):
    m = Tensor(np.max(x.data, axis=axis, keepdims=True))
    return (x - m).exp().sum(axis=axis, keepdims=True).log() + m


class CircleLoss(Module):
    """Weight each pair by HOW FAR it still has to move (Sun et al., 2020).

    Triplet loss treats every violating pair the same and stops the moment a fixed
    margin is met. Circle loss instead gives each similarity score its own weight:
    a within-class pair that is already close gets a small gradient, one that is
    still far gets a large one -- and symmetrically for between-class pairs. The
    name is geometric: the optimal decision region is a CIRCLE in
    (s_positive, s_negative) space, not the straight line a margin gives, which is
    why convergence is more definite. Operates on cosine similarities in [-1, 1].
    """

    def __init__(self, margin=0.25, gamma=64.0):
        super().__init__()
        self.m = margin
        self.gamma = gamma

    def forward(self, sp, sn):
        # sp: positive-pair similarities, sn: negative-pair similarities (1-D)
        sp = Tensor._wrap(sp); sn = Tensor._wrap(sn)
        ap = (1 + self.m - sp).relu()                 # self-paced weights
        an = (sn + self.m).relu()
        dp, dn = 1 - self.m, self.m
        logit_p = -self.gamma * ap * (sp - dp)
        logit_n = self.gamma * an * (sn - dn)
        # softplus(logsumexp(neg) + logsumexp(pos)) -- the circle-loss objective
        z = _logsumexp(logit_p) + _logsumexp(logit_n)
        return (1.0 + z.exp()).log().sum()


class MultiSimilarityLoss(Module):
    """Weight pairs by their similarity RELATIVE to the other pairs (Wang, 2019).

    A pair is informative not in absolute terms but compared to its neighbours: a
    negative that is more similar than most other negatives is the one worth
    pushing. Multi-similarity mines pairs on that relative criterion and then
    soft-weights them, capturing self-similarity, positive-relative and
    negative-relative signals in one loss -- which is why it tops many retrieval
    benchmarks. Takes a matrix of cosine similarities and the binary label of each
    pair.
    """

    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha, self.beta, self.base = alpha, beta, base

    def forward(self, sims, is_positive):
        sims = Tensor._wrap(sims)
        pos = np.asarray(is_positive, dtype=bool)
        sp = sims[np.where(pos)[0]]
        sn = sims[np.where(~pos)[0]]
        pos_term = (1.0 / self.alpha) * (
            1.0 + (-self.alpha * (sp - self.base)).exp().sum()).log()
        neg_term = (1.0 / self.beta) * (
            1.0 + (self.beta * (sn - self.base)).exp().sum()).log()
        return pos_term + neg_term


class AngularLoss(Module):
    """Constrain the ANGLE at the negative, not a distance (Wang et al., 2017).

    Distance-based triplet losses are sensitive to scale: the right margin differs
    across the space. The angular loss instead bounds the angle at the negative
    vertex of the anchor-positive-negative triangle, a quantity that is
    rotation- and scale-invariant and has a clear geometric meaning (a tighter
    angle => a better-separated triplet). It also brings in third-order
    relationships the pairwise losses miss. Inputs are (anchor, positive, negative)
    embedding batches.
    """

    def __init__(self, alpha_deg=45.0):
        super().__init__()
        self.tan_sq = np.tan(np.radians(alpha_deg)) ** 2

    def forward(self, anchor, positive, negative):
        a = _l2_normalize(Tensor._wrap(anchor))
        p = _l2_normalize(Tensor._wrap(positive))
        n = _l2_normalize(Tensor._wrap(negative))
        center = (a + p) * 0.5
        # f = 4 tan^2(alpha) (a+p).n  -  2(1+tan^2(alpha)) a.p
        term1 = 4.0 * self.tan_sq * ((a + p) * n).sum(axis=-1)
        term2 = 2.0 * (1.0 + self.tan_sq) * (a * p).sum(axis=-1)
        f = term1 - term2
        return (1.0 + f.exp()).log().mean()           # softplus of the angular margin


def _box_ious(pred, target):
    """Return (iou, enclosing corners, centers, wh) for two batches of xyxy boxes."""
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
    inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
    inter = inter_w * inter_h
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_t = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
    union = area_p + area_t - inter + 1e-9
    iou = inter / union
    return iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2), union


class BoxRegressionLoss(Module):
    """IoU-family losses for bounding boxes: the metric IS the objective
    (Rezatofighi 2019; Zheng 2020).

    Regressing box corners with squared error optimises the wrong thing -- two
    boxes can have equal corner error yet wildly different overlap, and non-
    overlapping boxes give no gradient at all. The IoU family fixes this:

    * ``iou``  -- 1 - overlap. Zero gradient when boxes are disjoint.
    * ``giou`` -- adds the area of the smallest ENCLOSING box, so disjoint boxes
      still get pulled together.
    * ``diou`` -- penalises the distance between box CENTRES: faster, and it
      converges even when one box contains the other.
    * ``ciou`` -- diou plus an ASPECT-RATIO term, the most complete of the four.

    Boxes are ``(x1, y1, x2, y2)``. Returns the mean loss.
    """

    def __init__(self, mode="ciou"):
        super().__init__()
        assert mode in ("iou", "giou", "diou", "ciou")
        self.mode = mode

    def forward(self, pred, target):
        pred = np.asarray(Tensor._wrap(pred).data, dtype=float)
        target = np.asarray(Tensor._wrap(target).data, dtype=float)
        iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2), _ = _box_ious(pred, target)
        loss = 1.0 - iou
        if self.mode == "iou":
            return Tensor(float(loss.mean()))
        # smallest enclosing box
        cx1 = np.minimum(px1, tx1); cy1 = np.minimum(py1, ty1)
        cx2 = np.maximum(px2, tx2); cy2 = np.maximum(py2, ty2)
        if self.mode == "giou":
            enc = (cx2 - cx1) * (cy2 - cy1) + 1e-9
            area_p = (px2 - px1) * (py2 - py1)
            area_t = (tx2 - tx1) * (ty2 - ty1)
            inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
            inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
            union = area_p + area_t - inter_w * inter_h + 1e-9
            giou = iou - (enc - union) / enc          # penalise the empty enclosure
            return Tensor(float((1.0 - giou).mean()))
        # center distance (DIoU / CIoU)
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        center_d = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        diag = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-9
        diou = iou - center_d / diag
        if self.mode == "diou":
            return Tensor(float((1.0 - diou).mean()))
        pw, ph = np.clip(px2 - px1, 1e-9, None), np.clip(py2 - py1, 1e-9, None)
        tw, th = np.clip(tx2 - tx1, 1e-9, None), np.clip(ty2 - ty1, 1e-9, None)
        v = (4 / np.pi ** 2) * (np.arctan(tw / th) - np.arctan(pw / ph)) ** 2
        alpha = v / (1 - iou + v + 1e-9)
        ciou = diou - alpha * v
        return Tensor(float((1.0 - ciou).mean()))


class SSIMLoss(Module):
    """Optimise for perceived STRUCTURE, not per-pixel error (Wang et al., 2004).

    Two images can have identical MSE while one looks fine and the other is
    visibly wrong, because MSE is blind to structure. The structural similarity
    index compares LUMINANCE, CONTRAST and STRUCTURE (via local means, variances
    and covariance), matching human judgement far better -- which is why it is the
    standard image-quality metric and a common training loss for super-resolution
    and denoising. ``1 - SSIM`` so that identical images give zero loss. Uses
    global per-image statistics.
    """

    def __init__(self, c1=0.01 ** 2, c2=0.03 ** 2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def forward(self, pred, target):
        x = Tensor._wrap(pred); y = Tensor._wrap(target)
        mx = x.mean(); my = y.mean()
        vx = ((x - mx) * (x - mx)).mean()
        vy = ((y - my) * (y - my)).mean()
        cov = ((x - mx) * (y - my)).mean()
        ssim = ((2 * mx * my + self.c1) * (2 * cov + self.c2)) / \
               ((mx * mx + my * my + self.c1) * (vx + vy + self.c2))
        return 1.0 - ssim


class TukeyBiweightLoss(Module):
    """Cap an outlier's influence ENTIRELY once it is far enough (Beaton, 1974).

    Squared error lets one outlier dominate; Huber caps its GRADIENT to a constant
    but still lets it pull forever. Tukey's biweight goes further: past a threshold
    ``c`` the residual contributes a CONSTANT loss and ZERO gradient -- a gross
    outlier is not down-weighted, it is switched off. This "redescending" behaviour
    is the most robust of the M-estimators, at the cost of being non-convex (so it
    needs a decent initialisation, e.g. from a Huber fit).
    """

    def __init__(self, c=4.685):
        super().__init__()
        self.c = c

    def forward(self, pred, target):
        r = Tensor._wrap(pred) - Tensor._wrap(target)
        c2 = self.c ** 2
        # rho(r) = (c^2/6)[1 - (1-(r/c)^2)^3] for |r|<=c, else c^2/6
        inside = (1.0 - (r / self.c) * (r / self.c)).relu()   # 0 outside |r|<=c
        rho = (c2 / 6.0) * (1.0 - inside * inside * inside)
        return rho.mean()


__all__ = ["CircleLoss", "MultiSimilarityLoss", "AngularLoss",
           "BoxRegressionLoss", "SSIMLoss", "TukeyBiweightLoss"]
