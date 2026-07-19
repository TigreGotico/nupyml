"""All-RELEVANT selection: keep every feature that beats random noise."""
import numpy as np
from ..base import BaseEstimator, check_is_fitted, clone
from ..utils import check_X_y, check_array, check_random_state


def _importance(est, n_features):
    if hasattr(est, "feature_importances_"):
        return est.feature_importances_
    if hasattr(est, "coef_"):
        c = np.abs(est.coef_)
        return c.ravel() if c.ndim == 1 else c.mean(axis=0)
    return np.ones(n_features)


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


__all__ = ["Boruta"]
