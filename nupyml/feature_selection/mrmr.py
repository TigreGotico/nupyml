"""Minimum Redundancy Maximum Relevance: pick informative, NON-redundant ones."""
import numpy as np
from ..utils import check_X_y, check_array, check_random_state


def mrmr(X, y, n_features, discrete=False):
    """Minimum Redundancy Maximum Relevance: pick informative, NON-redundant ones.

    THE INSIGHT
    -----------
    Ranking features by relevance alone (as univariate filters do) picks the top
    ``k`` most-correlated-with-target -- but if those top features are correlated
    with EACH OTHER, they carry the same information and the set is redundant. mRMR
    greedily builds the set to maximise relevance to the target MINUS the average
    redundancy with features already chosen::

        pick argmax_f [ relevance(f, y) - mean redundancy(f, selected) ]

    So the second feature chosen is not the second-most-relevant, but the one
    that adds the most NEW information given the first. This is why mRMR beats
    top-k filtering when features are correlated -- which they usually are.

    Peng, Long & Ding (2005). Returns the selected feature indices, in order.
    """
    from ..feature_selection import (mutual_info_classif, mutual_info_regression)
    X = check_array(X)
    y = np.asarray(y)
    d = X.shape[1]
    # relevance of every feature to the target (mutual information)
    if discrete:
        relevance = mutual_info_classif(X, y)
    else:
        relevance = mutual_info_regression(X, y)

    selected = [int(np.argmax(relevance))]         # start with the most relevant
    candidates = set(range(d)) - set(selected)
    while len(selected) < min(n_features, d) and candidates:
        best_score, best_f = -np.inf, None
        for f in candidates:
            # redundancy = mean absolute correlation with already-selected features
            redundancy = np.mean([abs(np.corrcoef(X[:, f], X[:, s])[0, 1])
                                  for s in selected])
            score = relevance[f] - redundancy
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        candidates.remove(best_f)
    return np.array(selected)


__all__ = ["mrmr"]
