"""Asymmetric loss for long-tailed MULTI-LABEL classification (Ben-Baruch, 2020)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class AsymmetricLoss(Module):
    """Asymmetric loss for long-tailed MULTI-LABEL classification (Ben-Baruch, 2020).

    In multi-label problems each image has a few positive labels and hundreds of
    negatives, so the easy negatives dominate. Asymmetric loss focuses on positives
    and negatives DIFFERENTLY: a larger focusing ``gamma_neg`` for negatives (down-
    weight the easy ones hard) than ``gamma_pos`` for positives, plus a probability
    SHIFT that fully discards very-easy negatives. This asymmetry is what makes it
    the strong default for multi-label with extreme negative imbalance. Operates on
    per-label sigmoid probabilities.
    """

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, target):
        y = Tensor(_float(target))
        p = logits.sigmoid()
        pm = p if self.clip <= 0 else (p - self.clip)   # shift down easy negatives
        pm = pm.relu() * (1 - y) + p * y                # only shift the negatives
        loss_pos = y * ((1 - p) ** self.gamma_pos) * (p + 1e-8).log()
        loss_neg = (1 - y) * (pm ** self.gamma_neg) * ((1 - p) + 1e-8).log()
        return -(loss_pos + loss_neg).mean()


__all__ = ["AsymmetricLoss"]
