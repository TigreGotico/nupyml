"""Logit-adjustment loss: correct for long-tailed class priors."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


class LogitAdjustmentLoss(Module):
    """Bake the class PRIOR into the loss for long-tailed data (Menon et al., 2020).

    On imbalanced data a plain softmax learns to favour frequent classes, because
    predicting the majority is a cheap way to cut the loss. Logit adjustment fixes this
    at the source: it ADDS ``tau * log(prior)`` to each logit before the softmax, so a
    frequent class must clear a higher bar to be chosen. This is exactly the Bayes-
    optimal correction for the shifted label distribution -- it targets the balanced
    error rate without resampling or reweighting, and costs one vector add. ``tau``
    scales the adjustment; ``priors`` are the training class frequencies.
    """

    def __init__(self, priors, tau=1.0):
        super().__init__()
        p = np.asarray(priors, dtype=float)
        self.log_prior = np.log(p / p.sum() + 1e-12)
        self.tau = tau

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        adjusted = Tensor._wrap(logits) + Tensor(self.tau * self.log_prior)
        logp = F.log_softmax(adjusted, axis=-1)
        n = logp.shape[0]
        return -logp[np.arange(n), target].mean()


__all__ = ["LogitAdjustmentLoss"]
