"""Symmetric mean absolute percentage error -- a bounded relative loss."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class SMAPELoss(Module):
    """Symmetric mean absolute percentage error -- a bounded relative loss.

    Plain MAPE blows up when the target is near zero and is asymmetric (penalises
    over- and under-prediction unequally). SMAPE divides the absolute error by the
    AVERAGE of the magnitudes, bounding it in [0, 2] and treating over/under
    symmetrically -- the standard forecasting-competition metric, here as a
    differentiable loss.
    """

    def forward(self, pred, target):
        y = Tensor(_float(target))
        num = (pred - y).abs()
        denom = pred.abs() + y.abs() + 1e-8
        return (num / denom).mean()


__all__ = ["SMAPELoss"]
