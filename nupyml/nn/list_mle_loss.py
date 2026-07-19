"""Listwise ranking loss: the Plackett-Luce likelihood of the correct order"""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class ListMLELoss(Module):
    """Listwise ranking loss: the Plackett-Luce likelihood of the correct order
    (Xia et al., 2008).

    Pairwise ranking losses (RankNet) look at pairs; listwise losses model the
    WHOLE permutation. ListMLE maximises the probability, under the Plackett-Luce
    model, of drawing items in their true relevance order: at each position, the
    next item should have the highest score among those not yet placed::

        L = -sum_i [ s_{(i)} - logsumexp(s_{(i)}, s_{(i+1)}, ...) ]

    where ``(i)`` is the item at rank ``i`` in the ideal order. Optimising the
    entire ordering at once often beats pairwise losses on ranking metrics.
    ``scores`` and ``relevance`` are per-query 1-D arrays.
    """

    def forward(self, scores, relevance):
        rel = _float(relevance)
        order = np.argsort(-rel)                       # ideal descending order
        s = scores[order] if isinstance(scores, Tensor) else Tensor(scores)[order]
        # log P(order) = sum_i [ s_i - logsumexp(s_i..s_n) ]  (reverse cumulative)
        n = s.shape[0]
        loss = Tensor(0.0)
        # logsumexp of the tail via a numerically stable manual pass
        for i in range(n):
            tail = s[i:]
            m = tail.data.max()
            lse = (tail - Tensor(m)).exp().sum().log() + m
            loss = loss + (lse - s[i])
        return loss / n


__all__ = ["ListMLELoss"]
