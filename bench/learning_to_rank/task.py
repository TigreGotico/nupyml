"""Learning to rank: order the items within each query by relevance.

Query-grouped data: the submission sees training feature vectors with graded
relevances and the query-group sizes, and must score the TEST items so that
relevant ones rank high WITHIN each query. Scored by mean NDCG@10 over the test
queries -- so only the order inside a query matters, not absolute scores.
"""
import numpy as np

from nupyml.ranking import ndcg_from_scores

KIND = "ranking"
GOAL = ("Rank items within each query by relevance; scored by mean NDCG@10 over "
        "the test queries.")
METRIC = "ndcg@10"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80


def _make(rng, w, n_queries, items=12, d=6):
    X, y, groups = [], [], []
    for _ in range(n_queries):
        Xg = rng.randn(items, d)
        # NON-LINEAR relevance (an interaction + a threshold), so a pointwise
        # linear model cannot capture it and tree/pairwise rankers earn their keep
        rel = (Xg @ w + 1.2 * Xg[:, 0] * Xg[:, 1]
               + 0.8 * (Xg[:, 2] > 0.5) + 0.3 * rng.randn(items))
        rel = rel - rel.min()
        rel = np.round(3 * rel / (rel.max() + 1e-9)).astype(int)   # grades 0..3
        X.append(Xg); y.append(rel); groups.append(items)
    return np.vstack(X), np.concatenate(y), groups


def load():
    rng = np.random.RandomState(0)
    w = rng.randn(6)                                  # shared relevance direction
    X_train, y_train, g_train = _make(rng, w, n_queries=80)
    X_test, y_test, g_test = _make(rng, w, n_queries=40)
    return X_train, y_train, g_train, X_test, y_test, g_test


def metric(y_true, scores, groups):
    ndcgs, start = [], 0
    for g in groups:
        sl = slice(start, start + g); start += g
        ndcgs.append(ndcg_from_scores(scores[sl], y_true[sl], k=10))
    return float(np.mean(ndcgs))
