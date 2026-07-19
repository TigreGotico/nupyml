"""Weight each pair by HOW FAR it still has to move (Sun et al., 2020)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _logsumexp(x, axis=None):
    m = Tensor(np.max(x.data, axis=axis, keepdims=True))
    return (x - m).exp().sum(axis=axis, keepdims=True).log() + m


class CircleLoss(Module):
    """Weight each pair by HOW FAR it still has to move (Sun et al., 2020).

    Triplet loss treats every violating pair the same and stops the moment a fixed
    margin is met. Circle loss instead gives each similarity score its own weight:
    a within-class pair that is already close gets a small gradient, one that is
    still far gets a large one -- and symmetrically for between-class pairs. The
    name is geometric: the optimal decision region is a CIRCLE in
    (s_positive, s_negative) space, not the straight line a margin gives, which is
    why convergence is more definite. Operates on cosine similarities in [-1, 1].
    """

    def __init__(self, margin=0.25, gamma=64.0):
        super().__init__()
        self.m = margin
        self.gamma = gamma

    def forward(self, sp, sn):
        # sp: positive-pair similarities, sn: negative-pair similarities (1-D)
        sp = Tensor._wrap(sp); sn = Tensor._wrap(sn)
        ap = (1 + self.m - sp).relu()                 # self-paced weights
        an = (sn + self.m).relu()
        dp, dn = 1 - self.m, self.m
        logit_p = -self.gamma * ap * (sp - dp)
        logit_n = self.gamma * an * (sn - dn)
        # softplus(logsumexp(neg) + logsumexp(pos)) -- the circle-loss objective
        z = _logsumexp(logit_p) + _logsumexp(logit_n)
        return (1.0 + z.exp()).log().sum()


__all__ = ["CircleLoss"]
