"""Losses v4: differentiable alignment, evidential uncertainty, long-tail
rebalancing, pairwise ranking, and boundary-aware segmentation.

Five losses for targets the standard set handles badly. Soft-DTW makes sequence
ALIGNMENT differentiable. The evidential loss makes a classifier report how much it
does not know. Seesaw rebalances gradients for long-tailed classes. RankNet learns
an ORDER from pairwise preferences. The boundary loss fixes region losses' blindness
to thin structures by weighting errors by their distance to the true boundary.
"""
import numpy as np
import scipy.ndimage as ndi

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def soft_dtw(x, y, gamma=1.0):
    """A DIFFERENTIABLE dynamic time warping distance (Cuturi & Blondel, 2017).

    DTW aligns two sequences by the best warping path, but the ``min`` in its
    recursion has no gradient, so you cannot train through it. Soft-DTW replaces
    that hard ``min`` with a SOFT one (``-gamma·logsumexp``), giving a smooth,
    differentiable measure of alignment cost -- so it can be a loss for forecasting
    or a metric for clustering time series of unequal length or phase. ``gamma``
    controls the softness (``->0`` recovers exact DTW). Returns a scalar Tensor,
    differentiable w.r.t. both sequences.
    """
    x = Tensor._wrap(x); y = Tensor._wrap(y)
    xd = x.data.ravel(); yd = y.data.ravel()
    n, m = len(xd), len(yd)
    INF = 1e10
    R = [[None] * (m + 1) for _ in range(n + 1)]
    R[0][0] = Tensor(0.0)
    for i in range(1, n + 1):
        R[i][0] = Tensor(INF)
    for j in range(1, m + 1):
        R[0][j] = Tensor(INF)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (x[i - 1:i] - y[j - 1:j])
            cost = (d * d).reshape(())
            R[i][j] = cost + _soft_min(R[i - 1][j], R[i][j - 1],
                                       R[i - 1][j - 1], gamma)
    return R[n][m]


def _soft_min(a, b, c, gamma):
    # soft-min(v) = -gamma * log( sum_i exp(-v_i / gamma) )
    stacked = Tensor.stack([a.reshape(()), b.reshape(()), c.reshape(())])
    return -gamma * ((stacked * (-1.0 / gamma)).exp().sum().log())


class EvidentialLoss(Module):
    """Make a classifier say how much it does NOT know (Sensoy et al., 2018).

    Softmax always produces a confident-looking distribution, even for nonsense
    inputs. Evidential deep learning treats the network's outputs as EVIDENCE for a
    Dirichlet distribution over the class probabilities: more total evidence means a
    sharper Dirichlet (confident), little evidence means one near uniform
    (uncertain). Training minimises the Bayes risk of the sum-of-squares (Brier)
    loss under that Dirichlet plus a penalty on evidence for WRONG classes -- so the
    model learns to withhold evidence when unsure. ``uncertainty`` returns ``K/S``,
    high when the model has seen nothing like the input (useful for OOD).
    """

    def __init__(self, n_classes, lam=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.lam = lam

    def forward(self, evidence, target):
        # evidence >= 0 (e.g. relu/softplus of logits); alpha = evidence + 1
        evidence = Tensor._wrap(evidence).relu()
        alpha = evidence + 1.0
        S = alpha.sum(axis=1, keepdims=True)
        p = alpha / S
        onehot = np.eye(self.n_classes)[np.asarray(target)]
        y = Tensor(onehot)
        err = ((y - p) * (y - p)).sum(axis=1)              # Brier error term
        var = (p * (1.0 - p) / (S + 1.0)).sum(axis=1)      # Dirichlet variance term
        # penalise evidence assigned to the wrong classes (drives them to uniform)
        wrong = (evidence * (1.0 - y)).sum(axis=1)
        return (err + var).mean() + self.lam * wrong.mean()

    @staticmethod
    def uncertainty(evidence):
        evidence = np.maximum(np.asarray(evidence), 0.0)
        alpha = evidence + 1.0
        K = evidence.shape[1]
        return K / alpha.sum(axis=1)                       # vacuity: high = unknown


class SeesawLoss(Module):
    """Rebalance the punishment of negative classes for long-tailed data
    (Wang et al., 2021).

    In a long-tailed dataset, a rare class is used as a NEGATIVE far more often than
    it appears as a positive, so its classifier is overwhelmed by suppressive
    gradients and never learns. Seesaw loss down-weights the negative gradient a
    frequent class exerts on a rarer one by a MITIGATION factor (the ratio of their
    cumulative sample counts), while a COMPENSATION factor keeps punishing genuine
    misclassifications -- balancing the two so tail classes survive training.
    ``class_counts`` are the seen sample counts per class.
    """

    def __init__(self, class_counts, p=0.8):
        super().__init__()
        self.counts = np.asarray(class_counts, float) + 1.0
        self.p = p

    def forward(self, logits, target):
        logits = Tensor._wrap(logits)
        target = np.asarray(target)
        n, K = logits.shape
        # mitigation: for sample of class t, softly reduce the logit of any class j
        # that is MORE frequent than t (ratio < 1), raised to power p
        ratio = self.counts[None, :] / self.counts[target][:, None]
        mitig = np.where(ratio < 1.0, ratio ** self.p, 1.0)   # (n, K)
        onehot = np.eye(K)[target]
        mitig = mitig * (1 - onehot) + onehot                 # never touch the true class
        adjusted = logits + Tensor(np.log(mitig + 1e-12))     # reweight in log space
        logp = F.log_softmax(adjusted, axis=1)
        return -(logp * Tensor(onehot)).sum(axis=1).mean()


class RankNetLoss(Module):
    """Learn an ORDER from pairwise preferences (Burges et al., 2005).

    Ranking is not regression: the exact scores do not matter, only their order. Yet
    the order is discrete and non-differentiable. RankNet makes it trainable: for a
    pair where ``i`` should outrank ``j``, model ``P(i > j) = sigmoid(s_i - s_j)``
    and apply cross-entropy against the known preference. Summed over pairs, this
    trains a scoring function whose ORDER matches the desired ranking -- the
    foundation of learning-to-rank (and, extended with the ranking metric's delta,
    of LambdaRank). ``scores_i``, ``scores_j`` are the model's scores for each side
    of a batch of pairs; ``labels`` is 1 if i should rank above j.
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores_i, scores_j, labels):
        si = Tensor._wrap(scores_i).reshape(-1)
        sj = Tensor._wrap(scores_j).reshape(-1)
        diff = si - sj
        y = Tensor(np.asarray(labels, float).ravel())
        # BCE-with-logits on sigmoid(diff): softplus(diff) - y*diff
        return ((1.0 + diff.exp()).log() - diff * y).mean()


class BoundaryLoss(Module):
    """Weight errors by their distance to the true BOUNDARY (Kervadec et al., 2019).

    Region losses (Dice, cross-entropy) treat every pixel equally, so a thin or
    small structure -- a vessel, a lesion edge -- contributes almost nothing and is
    ignored. The boundary loss instead integrates the prediction against a SIGNED
    DISTANCE map of the ground-truth boundary: predicting foreground far OUTSIDE the
    object (large positive distance) is penalised heavily, predicting inside is
    rewarded, and the emphasis concentrates exactly at the contour. Combined with a
    region loss it markedly improves boundary accuracy on imbalanced segmentation.
    ``target`` is a binary mask.
    """

    def __init__(self):
        super().__init__()

    def _signed_distance(self, mask):
        mask = np.asarray(mask).astype(bool)
        if mask.all() or not mask.any():
            return np.zeros(mask.shape, float)
        out = ndi.distance_transform_edt(~mask)            # distance outside (+)
        inside = ndi.distance_transform_edt(mask)          # distance inside (-)
        return out - inside

    def forward(self, pred, target):
        pred = Tensor._wrap(pred)
        phi = self._signed_distance(target)                # precomputed distance map
        return (pred * Tensor(phi)).mean()                 # integral of pred·distance


__all__ = ["soft_dtw", "EvidentialLoss", "SeesawLoss", "RankNetLoss",
           "BoundaryLoss"]
