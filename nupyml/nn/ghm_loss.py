"""Gradient Harmonizing Mechanism for classification (Li et al., 2019)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _as_float(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.float64)


class GHMLoss(Module):
    """Gradient Harmonizing Mechanism for classification (Li et al., 2019).

    THE INSIGHT
    -----------
    Focal loss down-weights easy examples by a fixed formula; GHM asks the data
    instead. It bins examples by their GRADIENT NORM ``g = |p - y|`` and
    down-weights each example by how CROWDED its bin is -- because a bin holding
    thousands of examples (the trivial easy ones, and separately the noisy
    outliers with huge gradients) is over-represented in the total gradient.
    Dividing each example's loss by its bin's gradient DENSITY equalises the
    contribution across the gradient spectrum, taming BOTH the easy majority and
    the harmful outliers at once -- something focal loss does not do.

    The bin weights are computed from the current predictions and treated as
    constants (they modulate the loss, they are not differentiated through).
    Binary classification on logits.
    """

    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, logits, target):
        y = _as_float(target)
        p = logits.sigmoid()
        g = np.abs(p.data - y)                        # gradient norm per example
        n = len(g)
        edges = np.linspace(0, 1.0 + 1e-6, self.bins + 1)
        weights = np.ones(n)
        for b in range(self.bins):
            mask = (g >= edges[b]) & (g < edges[b + 1])
            count = mask.sum()
            if count > 0:
                weights[mask] = n / (count * self.bins)   # inverse bin density
        w = Tensor(weights)
        # weighted binary cross-entropy
        eps = 1e-7
        bce = -(Tensor(y) * (p + eps).log()
                + Tensor(1.0 - y) * ((1.0 - p) + eps).log())
        return (w * bce).mean()


__all__ = ["GHMLoss"]
