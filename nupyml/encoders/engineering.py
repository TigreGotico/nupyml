"""Feature-engineering transformers: taming outliers, rare levels, and cycles."""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip extreme values to percentiles instead of dropping the rows.

    An outlier can wreck a mean, a variance, or a linear fit -- but dropping the
    row throws away everything else it carries. Winsorizing instead CLIPS each
    feature to, say, its 5th and 95th percentiles: the outlier's other features
    survive, and its extreme value is pulled to the boundary rather than deleted.
    It is the gentler alternative to outlier removal, and it keeps the sample size
    intact -- worth it when the extreme is a genuine-but-noisy observation rather
    than an error. The bounds are learned on training data and applied unchanged
    to test data, so no leakage.
    """

    def __init__(self, lower=5.0, upper=95.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = check_array(X)
        self.lower_bounds_ = np.percentile(X, self.lower, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        return np.clip(check_array(X), self.lower_bounds_, self.upper_bounds_)


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Fold infrequent categories into a single 'Rare' bucket.

    High-cardinality categoricals have a long tail of levels seen only a handful
    of times -- too rare to estimate anything reliable from, and a source of
    overfitting (the model memorises them) and of unseen levels at test time.
    Grouping every category below a frequency threshold into one 'Rare' label
    keeps the frequent, learnable levels distinct while collapsing the noise. It
    is the standard first step before target-encoding a messy categorical, and it
    guarantees test-time levels the model never saw still map somewhere sensible.
    """

    def __init__(self, tol=0.01, rare_label="Rare"):
        self.tol = tol
        self.rare_label = rare_label

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.frequent_ = []
        for j in range(X.shape[1]):
            cats, counts = np.unique(X[:, j], return_counts=True)
            freq = counts / counts.sum()
            # keep only the levels above the frequency threshold
            self.frequent_.append(set(cats[freq >= self.tol].tolist()))
        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_")
        X = np.asarray(X, dtype=object)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        out = X.copy()
        for j in range(X.shape[1]):
            keep = self.frequent_[j]
            out[:, j] = [v if v in keep else self.rare_label for v in X[:, j]]
        return out


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encode a cyclical feature as (sin, cos) so its ends wrap around.

    THE PROBLEM
    -----------
    Hour-of-day, month, day-of-week, compass bearing -- these are CYCLICAL: hour
    23 is adjacent to hour 0, but as raw integers they look maximally far apart
    (23 vs 0), and December (12) looks distant from January (1). A model fed the
    raw integer learns a false discontinuity at the wrap-around.

    THE FIX
    -------
    Map the value onto a circle: ``(sin(2*pi*v/period), cos(2*pi*v/period))``.
    Now 23:00 and 00:00 sit right next to each other on the circle, and the
    distance between any two times reflects their true cyclical gap. Two columns
    per cyclical feature, and the discontinuity is gone. ``period`` is the length
    of the cycle (24 for hours, 12 for months, 7 for weekdays).
    """

    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        check_array(X)
        return self

    def transform(self, X):
        X = check_array(X)
        blocks = []
        for j in range(X.shape[1]):
            angle = 2 * np.pi * X[:, j] / self.period
            blocks.append(np.sin(angle))
            blocks.append(np.cos(angle))
        return np.column_stack(blocks)


__all__ = ["Winsorizer", "RareLabelEncoder", "CyclicalEncoder"]
