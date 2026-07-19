"""Keep only features selected CONSISTENTLY across many resamples."""
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


__all__ = ["StabilitySelection"]
