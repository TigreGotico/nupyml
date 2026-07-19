"""CE + reverse-CE, robust to NOISY labels (Wang et al., 2019)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _int(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.int64)


class SymmetricCrossEntropy(Module):
    """CE + reverse-CE, robust to NOISY labels (Wang et al., 2019).

    Cross-entropy over-trusts labels: a mislabelled example produces a huge
    gradient that the network dutifully fits, memorising the noise. Symmetric CE
    adds a REVERSE term -- the cross-entropy of the label given the PREDICTION --
    which saturates for confident-but-wrong cases and stops them dominating::

        SCE = alpha * CE(p, y) + beta * CE(y, p)

    The reverse term is bounded, so noisy labels can no longer drag training. A
    one-line robustness fix over plain CE.
    """

    def __init__(self, alpha=1.0, beta=1.0, clip=-4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.clip = clip

    def forward(self, logits, target):
        target = _int(target)
        logp = F.log_softmax(logits, axis=-1)
        n, k = logp.shape
        ce = -logp[np.arange(n), target].mean()
        p = logp.exp()
        # reverse CE: treat the (one-hot) label's log as clipped constant
        onehot = np.eye(k)[target]
        log_y = np.clip(np.log(onehot + 1e-12), self.clip, 0.0)
        rce = -(p * Tensor(log_y)).sum(axis=1).mean()
        return self.alpha * ce + self.beta * rce


__all__ = ["SymmetricCrossEntropy"]
