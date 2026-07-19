"""Tversky index, focused on hard regions (Abraham & Khan, 2019)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _as_float(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.float64)


class FocalTverskyLoss(Module):
    """Tversky index, focused on hard regions (Abraham & Khan, 2019).

    THE SEGMENTATION IMBALANCE
    --------------------------
    In segmentation the object is a few pixels among a sea of background, so
    Dice/IoU (which weight false positives and false negatives equally) still let
    the easy background dominate. The Tversky index weights them SEPARATELY::

        TI = TP / (TP + alpha*FP + beta*FN)

    -- raise ``beta`` to punish missed object pixels (recall) over false alarms.
    Focal-Tversky then raises the shortfall to a power, ``(1 - TI) ** gamma``, to
    concentrate training on the images the model is getting wrong. Operates on
    per-sample sigmoid probabilities of the positive class.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        y = Tensor(_as_float(target))
        p = logits.sigmoid()
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - ti) ** self.gamma


__all__ = ["FocalTverskyLoss"]
