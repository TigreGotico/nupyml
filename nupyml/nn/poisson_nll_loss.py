"""Negative log-likelihood for COUNT targets under a Poisson model."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _as_float(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.float64)


class PoissonNLLLoss(Module):
    """Negative log-likelihood for COUNT targets under a Poisson model.

    When the target is a count (events per interval, clicks, defects), squared
    error is wrong -- the variance grows with the mean, and predictions can go
    negative. The Poisson NLL respects that::

        log_input=True :  loss = exp(input) - target * input
        log_input=False:  loss = input - target * log(input)

    With ``log_input=True`` (the safe default) the network outputs ``log(rate)``,
    so the rate ``exp(input)`` is automatically positive and the loss is stable.
    This is the count-data analogue of using cross-entropy for classes.
    """

    def __init__(self, log_input=True, eps=1e-8):
        super().__init__()
        self.log_input = log_input
        self.eps = eps

    def forward(self, pred, target):
        y = Tensor(_as_float(target))
        if self.log_input:
            return (pred.exp() - y * pred).mean()
        return (pred - y * (pred + self.eps).log()).mean()


__all__ = ["PoissonNLLLoss"]
