"""Advanced feature selection: Boruta, mRMR, ReliefF, stability selection.

The module already has wrapper (RFE, SFS) and filter (univariate) selectors.
These four capture ideas those miss: all-relevant selection, redundancy-aware
selection, interaction-aware relevance, and selection stability.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted, clone
from ..utils import check_X_y, check_array, check_random_state


class Boruta(BaseEstimator):
    """All-RELEVANT selection: keep every feature that beats random noise.

    THE DIFFERENT GOAL
    ------------------
    Most selectors find a MINIMAL set that predicts well -- they will drop a
    relevant feature if a correlated one already covers it. Boruta instead finds
    ALL-relevant features: every feature carrying information about the target,
    even redundant ones. That is what you want for UNDERSTANDING (which factors
    matter?), as opposed to building the leanest model.

    THE SHADOW TRICK
    ----------------
    How do you know a feature's importance is real and not noise? Boruta creates
    SHADOW features -- shuffled copies of the real ones, which by construction
    carry no information -- and trains a random forest on the real + shadow set. A
    real feature is confirmed only if its importance beats the BEST shadow's,
    repeatedly (a binomial test over many runs). So the noise floor is estimated
    from the data itself rather than guessed, which is the elegant part.

    Kursa & Rudnicki (2010).
    """

    def __init__(self, estimator=None, n_iter=30, alpha=0.05, random_state=None):
        self.estimator = estimator
        self.n_iter = n_iter
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        from ..ensemble import RandomForestClassifier
        from scipy import stats
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        base = (RandomForestClassifier(n_estimators=50, random_state=0)
                if self.estimator is None else self.estimator)

        hits = np.zeros(d)          # how often each feature beat the best shadow
        for it in range(self.n_iter):
            # shadow features: shuffled copies with the target relationship broken
            shadow = np.column_stack([rng.permutation(X[:, j]) for j in range(d)])
            combined = np.hstack([X, shadow])
            est = clone(base)
            est.fit(combined, y)
            imp = _importance(est, combined.shape[1])
            shadow_max = imp[d:].max()             # the noise ceiling this round
            hits += (imp[:d] > shadow_max)

        # a feature is confirmed if it beat the shadow max more often than a fair
        # coin would (binomial test) -- the data-driven significance test
        self.support_ = np.array([
            stats.binomtest(int(h), self.n_iter, 0.5,
                            alternative="greater").pvalue < self.alpha
            for h in hits])
        self.hits_ = hits
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return check_array(X)[:, self.support_]


def _importance(est, n_features):
    if hasattr(est, "feature_importances_"):
        return est.feature_importances_
    if hasattr(est, "coef_"):
        c = np.abs(est.coef_)
        return c.ravel() if c.ndim == 1 else c.mean(axis=0)
    return np.ones(n_features)


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


class StabilitySelection(BaseEstimator):
    """Keep only features selected CONSISTENTLY across many resamples.

    THE PROBLEM WITH ONE SELECTION RUN
    ----------------------------------
    Run a lasso (or any selector) once and the chosen features can change if you
    perturb the data slightly -- especially with correlated features, where the
    selector arbitrarily picks one of a group. A single run's selection is not
    trustworthy. Stability selection runs the selector on many BOOTSTRAP
    subsamples and keeps only features chosen in a high FRACTION of them.

    Consistency across resamples is strong evidence a feature is genuinely
    relevant, not an artefact of one particular sample. It comes with false-
    discovery control and is the principled answer to "which of my lasso's
    selected features can I actually believe". ``threshold`` is the selection
    frequency required to keep a feature.

    Meinshausen & Buhlmann (2010).
    """

    def __init__(self, estimator=None, n_bootstrap=50, threshold=0.6,
                 sample_fraction=0.5, random_state=None):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def fit(self, X, y):
        from ..linear_model import Lasso
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        base = Lasso(alpha=0.1) if self.estimator is None else self.estimator

        # standardise features AND target so a single lasso alpha bites at a
        # comparable strength -- on raw scales the penalty is dwarfed by the fit
        # term and the lasso keeps everything, so selection never sparsifies
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        ys = (y - y.mean()) / (y.std() + 1e-12)

        counts = np.zeros(d)
        n_sub = max(2, int(self.sample_fraction * n))
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, n_sub, replace=False)
            est = clone(base).fit(Xs[idx], ys[idx])
            imp = _importance(est, d)
            counts += (np.abs(imp) > 1e-8)         # was this feature selected?
        self.selection_frequency_ = counts / self.n_bootstrap
        self.support_ = self.selection_frequency_ >= self.threshold
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return check_array(X)[:, self.support_]


__all__ = ["Boruta", "mrmr", "relieff", "StabilitySelection"]
