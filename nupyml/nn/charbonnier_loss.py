"""A smooth, robust L1 (Charbonnier / pseudo-Huber): ``sqrt((y-p)^2 + eps^2)``."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class CharbonnierLoss(Module):
    """A smooth, robust L1 (Charbonnier / pseudo-Huber): ``sqrt((y-p)^2 + eps^2)``.

    L2 is smooth but outlier-sensitive; L1 is robust but has a kink at zero (bad
    gradient there). Charbonnier is L1's smooth cousin -- quadratic near zero (clean
    gradient) and linear far away (outlier-robust) -- controlled by one ``eps``. The
    standard reconstruction loss in image super-resolution and optical flow.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - Tensor(_float(target))
        return ((diff * diff + self.eps ** 2).sqrt()).mean()


__all__ = ["CharbonnierLoss"]
