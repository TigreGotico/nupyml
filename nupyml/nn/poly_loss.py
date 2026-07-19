"""Cross-entropy plus a first-order polynomial correction (Leng et al., 2022)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _as_int(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.int64)


class PolyLoss(Module):
    """Cross-entropy plus a first-order polynomial correction (Leng et al., 2022).

    Poly-1 adds a single term to cross-entropy::

        loss = CE + epsilon * (1 - p_true)

    Expanding CE as a Taylor series in ``(1 - p_true)`` shows its leading term is
    exactly ``(1 - p_true)``; PolyLoss lets you tune that leading coefficient
    directly with one hyperparameter, which reliably beats plain CE by a small
    margin across tasks. Positive ``epsilon`` up-weights not-yet-confident
    examples (like a gentle focal loss); negative ``epsilon`` down-weights them.
    """

    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):
        target = _as_int(target)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        logp_t = logp[np.arange(n), target]
        p_t = logp_t.exp()
        ce = -logp_t
        return (ce + self.epsilon * (1.0 - p_t)).mean()


__all__ = ["PolyLoss"]
