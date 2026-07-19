"""Constrain the ANGLE at the negative, not a distance (Wang et al., 2017)."""
import numpy as np
from ..autograd import Tensor
from .module import Module
from .metric_losses import _l2_normalize


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


__all__ = ["AngularLoss"]
