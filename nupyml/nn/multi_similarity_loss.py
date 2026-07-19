"""Weight pairs by their similarity RELATIVE to the other pairs (Wang, 2019)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class MultiSimilarityLoss(Module):
    """Weight pairs by their similarity RELATIVE to the other pairs (Wang, 2019).

    A pair is informative not in absolute terms but compared to its neighbours: a
    negative that is more similar than most other negatives is the one worth
    pushing. Multi-similarity mines pairs on that relative criterion and then
    soft-weights them, capturing self-similarity, positive-relative and
    negative-relative signals in one loss -- which is why it tops many retrieval
    benchmarks. Takes a matrix of cosine similarities and the binary label of each
    pair.
    """

    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha, self.beta, self.base = alpha, beta, base

    def forward(self, sims, is_positive):
        sims = Tensor._wrap(sims)
        pos = np.asarray(is_positive, dtype=bool)
        sp = sims[np.where(pos)[0]]
        sn = sims[np.where(~pos)[0]]
        pos_term = (1.0 / self.alpha) * (
            1.0 + (-self.alpha * (sp - self.base)).exp().sum()).log()
        neg_term = (1.0 / self.beta) * (
            1.0 + (self.beta * (sn - self.base)).exp().sum()).log()
        return pos_term + neg_term


__all__ = ["MultiSimilarityLoss"]
