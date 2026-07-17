"""Ranking metrics: judge an ORDER, not a value.

Ranking evaluation asks "did the relevant items end up near the TOP?", which is
what matters for search and recommendation -- the 1000th result's exact score is
irrelevant. Every metric here takes items sorted by predicted score (or scores +
true relevances) and rewards putting relevant items high, most with a discount
that falls off with rank.
"""
import numpy as np


def dcg_at_k(relevances, k=None):
    """Discounted Cumulative Gain: summed relevance, discounted by log(rank).

    ``relevances`` are the true relevance grades in PREDICTED order. Each is
    discounted by ``1/log2(rank+1)`` so a relevant item found at rank 1 counts
    far more than the same item at rank 10 -- encoding that users scan from the
    top and rarely go deep.
    """
    rel = np.asarray(relevances, float)
    if k is not None:
        rel = rel[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(relevances, k=None):
    """NDCG: DCG divided by the DCG of the IDEAL ordering -> in [0, 1].

    Normalising by the best possible DCG makes the metric comparable across
    queries with different numbers of relevant items. 1.0 means the ranking is
    perfect down to rank ``k``.
    """
    ideal = np.sort(np.asarray(relevances, float))[::-1]
    best = dcg_at_k(ideal, k)
    return dcg_at_k(relevances, k) / best if best > 0 else 0.0


def average_precision(relevances):
    """Average of the precision each time a relevant (binary) item is hit.

    Precision-at-k averaged over the positions of the relevant items -- it rewards
    packing relevant items early AND retrieving all of them. The per-query basis
    of mean average precision (MAP).
    """
    rel = (np.asarray(relevances) > 0).astype(float)
    if rel.sum() == 0:
        return 0.0
    cum = np.cumsum(rel)
    precisions = cum / np.arange(1, len(rel) + 1)
    return float(np.sum(precisions * rel) / rel.sum())


def mean_average_precision(list_of_relevances):
    """MAP: average precision averaged over queries."""
    return float(np.mean([average_precision(r) for r in list_of_relevances]))


def mean_reciprocal_rank(list_of_relevances):
    """MRR: average of 1/(rank of the FIRST relevant item) over queries.

    The metric when only the first correct answer matters (a factoid question, "I
    feel lucky" search). A first relevant item at rank 3 scores 1/3.
    """
    rr = []
    for rel in list_of_relevances:
        rel = np.asarray(rel) > 0
        hit = np.where(rel)[0]
        rr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return float(np.mean(rr))


def hit_rate_at_k(list_of_relevances, k):
    """Fraction of queries with at least one relevant item in the top ``k``.

    The blunt "did we get anything useful on the first page?" measure, ubiquitous
    in recommender evaluation."""
    hits = [np.any(np.asarray(rel)[:k] > 0) for rel in list_of_relevances]
    return float(np.mean(hits))


def _order_by_score(scores, relevances):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(relevances)[order]


def ndcg_from_scores(scores, relevances, k=None):
    """Convenience: sort relevances by predicted score, then NDCG@k."""
    return ndcg_at_k(_order_by_score(scores, relevances), k)


__all__ = ["dcg_at_k", "ndcg_at_k", "average_precision",
           "mean_average_precision", "mean_reciprocal_rank", "hit_rate_at_k",
           "ndcg_from_scores"]
