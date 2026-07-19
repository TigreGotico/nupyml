"""ApproxNDCG loss: a differentiable surrogate for NDCG in learning-to-rank."""
import numpy as np

from ..autograd import Tensor
from .module import Module


class ApproxNDCGLoss(Module):
    """Optimise the RANKING metric directly, smoothly (Qin et al., 2010).

    NDCG is what ranking systems are judged on, but it depends only on the sorted ORDER
    of scores, so its gradient is zero almost everywhere -- useless for training.
    ApproxNDCG replaces the hard rank of each item with a SMOOTH one: item i's rank is
    ``1 + sum_j sigmoid((s_j - s_i)/T)``, which counts how many items outscore it but
    differentiably. Plugging that soft rank into NDCG's discount gives a loss whose
    gradient actually points toward better orderings, so the model optimises the metric
    it is measured by rather than a convenient proxy. ``temperature`` controls the
    approximation sharpness. Scores and relevances are per-query 1-D.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, relevance):
        s = Tensor._wrap(scores)
        rel = np.asarray(relevance if not isinstance(relevance, Tensor)
                         else relevance.data, dtype=float)
        n = s.shape[0]
        gain = 2.0 ** rel - 1.0                             # standard NDCG gains
        # soft rank_i = 1 + sum_{j != i} sigmoid((s_j - s_i)/T); diagonal adds 0.5
        si = s.reshape((n, 1))
        sj = s.reshape((1, n))
        pairs = ((sj - si) * (1.0 / self.temperature)).sigmoid()
        approx_rank = pairs.sum(axis=1) + Tensor(0.5 * np.ones(n))
        discount = (approx_rank + 1.0).log() * (1.0 / np.log(2.0))   # log2(1 + rank)
        dcg = (Tensor(gain) / discount).sum()
        ideal_rank = np.argsort(-rel).argsort() + 1.0       # true sorted positions
        idcg = float((gain / np.log2(ideal_rank + 1.0)).sum()) + 1e-9
        return -(dcg / idcg)                                # maximise NDCG


__all__ = ["ApproxNDCGLoss"]
