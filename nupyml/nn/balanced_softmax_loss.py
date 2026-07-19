"""Correct the softmax for the TRAINING class prior (Ren et al., 2020)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _as_int(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.int64)


class BalancedSoftmaxLoss(Module):
    """Correct the softmax for the TRAINING class prior (Ren et al., 2020).

    Under class imbalance the softmax bakes in the training frequencies, so at
    test time (where classes are balanced) it over-predicts the head classes.
    Balanced softmax simply ADDS ``log(prior_c)`` to each class logit before the
    ordinary cross-entropy::

        loss = CE( logits + log(prior),  target )

    This is the Bayes-consistent adjustment: it cancels the training prior so the
    learned scores reflect the likelihood, not the frequency. One line, no tuning,
    and it often beats elaborate reweighting on long-tailed benchmarks.
    """

    def __init__(self, samples_per_class):
        super().__init__()
        prior = np.asarray(samples_per_class, float)
        self.log_prior = np.log(prior / prior.sum() + 1e-12)

    def forward(self, logits, target):
        target = _as_int(target)
        adjusted = logits + Tensor(self.log_prior)   # broadcast add to every row
        logp = F.log_softmax(adjusted, axis=-1)
        n = logp.shape[0]
        return (-logp[np.arange(n), target]).mean()


__all__ = ["BalancedSoftmaxLoss"]
