"""ReliefF: score features by how well they separate NEAR MISSES from NEAR HITS."""
import numpy as np
from ..utils import check_X_y, check_array, check_random_state


def relieff(X, y, n_neighbors=10, random_state=None):
    """ReliefF: score features by how well they separate NEAR MISSES from NEAR HITS.

    THE INSIGHT
    -----------
    For a sampled point, find its nearest same-class neighbours (near HITS) and
    nearest different-class neighbours (near MISSES). A good feature has SMALL
    differences to hits (same class => should look alike on it) and LARGE
    differences to misses (different class => should differ). ReliefF accumulates
    exactly that: reward for distinguishing misses, penalty for varying within a
    class.

    WHY IT BEATS UNIVARIATE FILTERS
    -------------------------------
    Because it works LOCALLY, in the context of each point's neighbourhood, it
    detects features that matter only in INTERACTION -- a feature useless on
    average but decisive near the boundary. A univariate correlation filter,
    judging each feature globally and alone, is blind to exactly those. That local,
    interaction-aware view is ReliefF's whole reason to exist.

    Kononenko (1994). Returns a relevance weight per feature.
    """
    from scipy.spatial.distance import cdist
    X, y = check_X_y(X, y)
    rng = check_random_state(random_state)
    n, d = X.shape
    # normalise features so the distance is not dominated by scale
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges == 0] = 1.0
    Xn = X / ranges
    D = cdist(Xn, Xn)
    np.fill_diagonal(D, np.inf)

    weights = np.zeros(d)
    classes = np.unique(y)
    priors = {c: np.mean(y == c) for c in classes}
    for i in range(n):
        # nearest same-class hits, and nearest misses per other class
        same = np.where(y == y[i])[0]
        same = same[same != i]
        if len(same):
            hits = same[np.argsort(D[i, same])[:n_neighbors]]
            # within class: penalise features that VARY (should be alike)
            weights -= np.abs(Xn[i] - Xn[hits]).mean(axis=0)
        for c in classes:
            if c == y[i]:
                continue
            other = np.where(y == c)[0]
            miss = other[np.argsort(D[i, other])[:n_neighbors]]
            # across classes: reward features that DIFFER, weighted by prior
            w = priors[c] / (1 - priors[y[i]] + 1e-12)
            weights += w * np.abs(Xn[i] - Xn[miss]).mean(axis=0)
    return weights / n


__all__ = ["relieff"]
