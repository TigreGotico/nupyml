"""Reweight classes by their EFFECTIVE number of samples (Cui et al., 2019)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _as_int(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.int64)


class ClassBalancedLoss(Module):
    """Reweight classes by their EFFECTIVE number of samples (Cui et al., 2019).

    THE PROBLEM WITH 1/frequency
    ----------------------------
    Long-tailed data tempts you to weight each class by ``1/count``. But samples
    within a class OVERLAP -- the 1000th cat photo adds far less new information
    than the 2nd -- so raw frequency over-weights rare classes. The "effective
    number" ``(1 - beta^n) / (1 - beta)`` models this saturation: it grows with
    ``n`` but flattens out, so a class with 10 samples and one with 10000 are not
    treated as 1000x apart. The per-class weight is ``(1 - beta) / (1 - beta^n)``.

    ``beta`` near 1 (e.g. 0.999) makes the effect strong; ``beta=0`` recovers no
    reweighting. Wraps cross-entropy or focal loss.
    """

    def __init__(self, samples_per_class, beta=0.999, gamma=0.0):
        super().__init__()
        n = np.asarray(samples_per_class, dtype=np.float64)
        eff = 1.0 - np.power(beta, n)
        w = (1.0 - beta) / np.maximum(eff, 1e-12)
        self.weights = w / w.sum() * len(w)          # normalise to mean 1
        self.gamma = gamma

    def forward(self, logits, target):
        target = _as_int(target)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        logp_t = logp[np.arange(n), target]
        modulator = (1.0 - logp_t.exp()) ** self.gamma if self.gamma else 1.0
        w = Tensor(self.weights[target])
        return (-(w * modulator * logp_t)).mean()


__all__ = ["ClassBalancedLoss"]
