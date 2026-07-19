"""Rank a whole LIST at once, not pair by pair (Cao et al., 2007)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class ListNetLoss(Module):
    """Rank a whole LIST at once, not pair by pair (Cao et al., 2007).

    Pointwise ranking losses ignore that ranking is about ORDER; pairwise losses
    (RankNet) consider two items at a time and can conflict. ListNet takes the whole
    list: it turns both the model's scores and the true relevance labels into a
    "top-one probability" distribution (a softmax over items -- the probability each
    item is ranked first) and minimises the cross-entropy between them. Optimising
    the list distribution directly tends to order the whole list better than
    stitching pairwise decisions together. ``scores`` and ``relevance`` are each one
    query's item vector.
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores, relevance):
        scores = Tensor._wrap(scores).reshape(1, -1)
        rel = np.asarray(relevance, float).reshape(1, -1)
        target = _softmax_np(rel)                         # true top-1 distribution
        logp = F.log_softmax(scores, axis=1)
        return -(Tensor(target) * logp).sum()


__all__ = ["ListNetLoss"]
