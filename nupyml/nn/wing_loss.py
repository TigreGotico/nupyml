"""Wing loss for regression of small targets (Feng et al., 2018)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _as_float(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.float64)


class WingLoss(Module):
    """Wing loss for regression of small targets (Feng et al., 2018).

    Built for facial-landmark regression, where most errors are TINY and a few are
    large. L2 ignores the small errors (their gradient vanishes near zero) yet is
    dominated by the large ones; L1 treats all equally. Wing loss is LOGARITHMIC
    for small errors -- so it keeps pushing on the many near-misses that matter --
    and LINEAR for large ones, so outliers do not explode it::

        wing(x) = w * ln(1 + |x|/eps)      if |x| < w
                = |x| - C                   otherwise

    with ``C`` chosen to make the two pieces meet continuously. ``w`` sets the
    switch point, ``eps`` the curvature of the log region.
    """

    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * np.log(1 + w / eps)

    def forward(self, pred, target):
        diff = (pred - Tensor(_as_float(target))).abs()
        small = self.w * (1.0 + diff / self.eps).log()
        large = diff - self.C
        mask = (diff.data < self.w).astype(np.float64)   # constant selector
        return (small * Tensor(mask) + large * Tensor(1.0 - mask)).mean()


__all__ = ["WingLoss"]
