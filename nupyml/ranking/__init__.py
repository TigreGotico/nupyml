"""Learning to rank and ranking metrics.

Ranking is its own problem: what matters is the ORDER of items within a query,
not their exact predicted values, and evaluation rewards putting relevant items
near the top.

* ``metrics`` -- DCG/NDCG, (mean) average precision, MRR, hit-rate: all reward a
  good order, most with a rank discount.
* ``learning_to_rank`` -- ``RankNet`` (pairwise cross-entropy on a linear scorer)
  and ``LambdaMART`` (gradient-boosted trees driven by NDCG-scaled pairwise lambda
  gradients), trained on query-grouped data.
"""
from .metrics import (dcg_at_k, ndcg_at_k, average_precision,
                      mean_average_precision, mean_reciprocal_rank,
                      hit_rate_at_k, ndcg_from_scores)
from .learning_to_rank import RankNet, LambdaMART

__all__ = ["dcg_at_k", "ndcg_at_k", "average_precision",
           "mean_average_precision", "mean_reciprocal_rank", "hit_rate_at_k",
           "ndcg_from_scores", "RankNet", "LambdaMART"]
