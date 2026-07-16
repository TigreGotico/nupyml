"""Loss functions (modules returning scalar Tensors)."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


class MSELoss(Module):
    def forward(self, pred, target):
        target = Tensor._wrap(target)
        return ((pred - target.detach()) ** 2).mean()


class MAELoss(Module):
    def forward(self, pred, target):
        target = Tensor._wrap(target)
        return (pred - target.detach()).abs().mean()


class HuberLoss(Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        target = Tensor._wrap(target).detach()
        diff = pred - target
        absdiff = diff.abs()
        quad = diff * diff * 0.5
        lin = absdiff * self.delta - 0.5 * self.delta ** 2
        return Tensor.where(absdiff.data <= self.delta, quad, lin).mean()


class CrossEntropyLoss(Module):
    """Fused log-softmax + NLL over integer class targets (logits input)."""

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        picked = logp[np.arange(n), target]
        return -picked.mean()


class NLLLoss(Module):
    """Negative log likelihood over log-probabilities."""

    def forward(self, logp, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        n = logp.shape[0]
        return -logp[np.arange(n), target].mean()


class BCELoss(Module):
    """Binary cross-entropy over probabilities in (0, 1)."""

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = Tensor._wrap(target).detach()
        p = pred.clip(self.eps, 1 - self.eps)
        return -(target * p.log() + (1.0 - target) * (1.0 - p).log()).mean()


class BCEWithLogitsLoss(Module):
    """Numerically stable BCE over logits: max(x,0) - x*y + log(1+exp(-|x|))."""

    def forward(self, logits, target):
        target = Tensor._wrap(target).detach()
        relu_x = logits.relu()
        softplus = ((-logits.abs()).exp() + 1.0).log()
        return (relu_x - logits * target + softplus).mean()


__all__ = ["MSELoss", "MAELoss", "HuberLoss", "CrossEntropyLoss", "NLLLoss",
           "BCELoss", "BCEWithLogitsLoss"]
