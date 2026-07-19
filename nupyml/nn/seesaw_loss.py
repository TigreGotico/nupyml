"""Rebalance the punishment of negative classes for long-tailed data"""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


class SeesawLoss(Module):
    """Rebalance the punishment of negative classes for long-tailed data
    (Wang et al., 2021).

    In a long-tailed dataset, a rare class is used as a NEGATIVE far more often than
    it appears as a positive, so its classifier is overwhelmed by suppressive
    gradients and never learns. Seesaw loss down-weights the negative gradient a
    frequent class exerts on a rarer one by a MITIGATION factor (the ratio of their
    cumulative sample counts), while a COMPENSATION factor keeps punishing genuine
    misclassifications -- balancing the two so tail classes survive training.
    ``class_counts`` are the seen sample counts per class.
    """

    def __init__(self, class_counts, p=0.8):
        super().__init__()
        self.counts = np.asarray(class_counts, float) + 1.0
        self.p = p

    def forward(self, logits, target):
        logits = Tensor._wrap(logits)
        target = np.asarray(target)
        n, K = logits.shape
        # mitigation: for sample of class t, softly reduce the logit of any class j
        # that is MORE frequent than t (ratio < 1), raised to power p
        ratio = self.counts[None, :] / self.counts[target][:, None]
        mitig = np.where(ratio < 1.0, ratio ** self.p, 1.0)   # (n, K)
        onehot = np.eye(K)[target]
        mitig = mitig * (1 - onehot) + onehot                 # never touch the true class
        adjusted = logits + Tensor(np.log(mitig + 1e-12))     # reweight in log space
        logp = F.log_softmax(adjusted, axis=1)
        return -(logp * Tensor(onehot)).sum(axis=1).mean()


__all__ = ["SeesawLoss"]
