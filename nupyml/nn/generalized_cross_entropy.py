"""A noise-robust interpolation between CE and MAE (Zhang & Sabuncu, 2018)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _int(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.int64)


class GeneralizedCrossEntropy(Module):
    """A noise-robust interpolation between CE and MAE (Zhang & Sabuncu, 2018).

    ``L_q = (1 - p_true^q) / q`` interpolates between cross-entropy (``q -> 0``,
    accurate but noise-sensitive) and mean-absolute-error-on-probabilities
    (``q = 1``, robust but slow). A middle ``q`` (e.g. 0.7) keeps CE's fast early
    learning while bounding the gradient a wrong label can produce -- the tunable
    robustness knob CE lacks.
    """

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, target):
        target = _int(target)
        p = F.softmax(logits, axis=-1)
        n = p.shape[0]
        p_true = p[np.arange(n), target]
        return ((1.0 - p_true ** self.q) / self.q).mean()


__all__ = ["GeneralizedCrossEntropy"]
