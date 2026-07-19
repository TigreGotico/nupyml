"""Learn an ORDER from pairwise preferences (Burges et al., 2005)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class RankNetLoss(Module):
    """Learn an ORDER from pairwise preferences (Burges et al., 2005).

    Ranking is not regression: the exact scores do not matter, only their order. Yet
    the order is discrete and non-differentiable. RankNet makes it trainable: for a
    pair where ``i`` should outrank ``j``, model ``P(i > j) = sigmoid(s_i - s_j)``
    and apply cross-entropy against the known preference. Summed over pairs, this
    trains a scoring function whose ORDER matches the desired ranking -- the
    foundation of learning-to-rank (and, extended with the ranking metric's delta,
    of LambdaRank). ``scores_i``, ``scores_j`` are the model's scores for each side
    of a batch of pairs; ``labels`` is 1 if i should rank above j.
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores_i, scores_j, labels):
        si = Tensor._wrap(scores_i).reshape(-1)
        sj = Tensor._wrap(scores_j).reshape(-1)
        diff = si - sj
        y = Tensor(np.asarray(labels, float).ravel())
        # BCE-with-logits on sigmoid(diff): softplus(diff) - y*diff
        return ((1.0 + diff.exp()).log() - diff * y).mean()


__all__ = ["RankNetLoss"]
