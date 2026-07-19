"""Mean squared error in LOG space -- penalise relative, not absolute, error."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class MSLELoss(Module):
    """Mean squared error in LOG space -- penalise relative, not absolute, error.

    For targets spanning orders of magnitude (counts, prices, populations) squared
    error is dominated by the large values. MSLE takes the error of ``log(1+y)``,
    so being off by 10% costs the same whether the true value is 10 or 10000 -- it
    measures RELATIVE error, and it penalises UNDER-prediction more than over.
    Requires non-negative targets.
    """

    def forward(self, pred, target):
        y = Tensor(_float(target))
        lp = (pred.relu() + 1.0).log()
        ly = (y + 1.0).log()
        return ((lp - ly) ** 2).mean()


__all__ = ["MSLELoss"]
