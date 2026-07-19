"""MMCE: a differentiable kernel measure of miscalibration."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


class MMCELoss(Module):
    """Make CONFIDENCE match ACCURACY, differentiably (Kumar et al., 2018).

    A network can be accurate yet badly CALIBRATED -- 90%-confident on predictions that
    are right only 70% of the time. Expected Calibration Error measures this but is
    piecewise-constant (it bins predictions), so it cannot be trained on. MMCE reframes
    calibration as a kernel two-sample statistic: in a good model the confidences of
    CORRECT and INCORRECT predictions are distributed so that, weighted by a Laplacian
    kernel over confidence, the gap ``(correct - confidence)`` has zero mean embedding.
    The resulting quantity is smooth in the logits, so it can be added to cross-entropy
    as a trainable calibration regulariser. ``kernel_width`` sets the kernel scale.
    """

    def __init__(self, kernel_width=0.2):
        super().__init__()
        self.h = kernel_width

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        probs = F.softmax(Tensor._wrap(logits), axis=-1)
        n = probs.shape[0]
        pred = np.argmax(probs.data, axis=1)
        conf = probs[np.arange(n), pred]                    # top-class confidence (diff'able)
        correct = (pred == target).astype(float)
        r = Tensor(correct) - conf                          # per-sample calibration gap
        # Laplacian kernel over confidences (constant weights, not differentiated)
        cd = np.abs(conf.data[:, None] - conf.data[None, :])
        k = Tensor(np.exp(-cd / self.h))
        ri = r.reshape((n, 1)); rj = r.reshape((1, n))
        mmce_sq = (ri * rj * k).sum() * (1.0 / (n * n))     # PSD kernel -> non-negative
        return (mmce_sq + 1e-12).sqrt()


__all__ = ["MMCELoss"]
