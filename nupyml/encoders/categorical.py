"""Target-based and target-free category encoders.

Each takes a column (or 2-D array) of categorical values. Target-based encoders
also take ``y`` and map each category to a number derived from the target,
shrinking toward a prior to bound the leak that raw target-encoding has.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array


def _as_2d(X):
    X = np.asarray(X, dtype=object)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence: encode a category by its log-odds against the base.

    THE IDEA
    --------
    For a binary target, each category is replaced by::

        WOE = log( P(category | y=1) / P(category | y=0) )

    -- how much more a category is associated with the positive class than the
    negative, in log-odds. A category that is neutral gets WOE 0; one that skews
    positive gets a large positive value.

    WHY CREDIT SCORING LOVES IT
    ---------------------------
    WOE is already on the LOG-ODDS scale, which is exactly what logistic
    regression models linearly -- so a WOE-encoded feature enters the model
    additively and its coefficient reads as a clean effect. That interpretability
    (and the monotone, regulator-friendly transform) is why it is the standard
    encoding in credit and fraud scorecards. Smoothing keeps a rare category from
    producing an infinite log-odds.
    """

    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y)
        self.maps_ = []
        for j in range(X.shape[1]):
            n_pos = max((y == 1).sum(), 1)
            n_neg = max((y == 0).sum(), 1)
            m = {}
            for cat in np.unique(X[:, j]):
                mask = X[:, j] == cat
                pos = (y[mask] == 1).sum() + self.smoothing
                neg = (y[mask] == 0).sum() + self.smoothing
                # log of the within-category odds relative to the base rate
                m[cat] = np.log((pos / n_pos) / (neg / n_neg))
            self.maps_.append(m)
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = _as_2d(X)
        out = np.zeros(X.shape)
        for j in range(X.shape[1]):
            out[:, j] = [self.maps_[j].get(v, 0.0) for v in X[:, j]]
        return out


class JamesSteinEncoder(BaseEstimator, TransformerMixin):
    """Shrink each category's target mean toward the global mean, by reliability.

    THE STEIN IDEA
    --------------
    A category seen 3 times gives a noisy estimate of its target mean; one seen
    3000 times gives a reliable one. James-Stein shrinks each category's mean
    toward the GLOBAL mean by an amount inversely proportional to how reliable
    that category's estimate is -- rare categories get pulled almost all the way
    to the global mean, frequent ones barely move. This is Stein's paradox again
    (the same shrinkage that beats the sample covariance in ``covariance``):
    trading a little bias for much less variance, provably better than the raw
    per-category means.

    The shrinkage weight is derived from the ratio of within-category to between-
    category variance, so it is DATA-DRIVEN, not a tuned knob -- the appeal over
    the simpler m-estimate.
    """

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, float)
        self.global_mean_ = y.mean()
        pooled_var = y.var() + 1e-12            # within-group noise scale
        self.maps_ = []
        for j in range(X.shape[1]):
            # between-category variance: how much the category means genuinely
            # differ (the signal shrinkage preserves)
            means = np.array([y[X[:, j] == c].mean() for c in np.unique(X[:, j])])
            tau2 = means.var() + 1e-12
            m = {}
            for cat in np.unique(X[:, j]):
                yc = y[X[:, j] == cat]
                n = len(yc)
                cat_mean = yc.mean()
                # the category mean's STANDARD ERROR is pooled_var / n, so a
                # small n makes the estimate unreliable EVEN IF its own values
                # happen to be identical. weight = signal / (signal + noise):
                # frequent categories (tiny se2) keep their mean, rare ones shrink
                se2 = pooled_var / n
                weight = tau2 / (tau2 + se2)
                m[cat] = weight * cat_mean + (1 - weight) * self.global_mean_
            self.maps_.append(m)
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = _as_2d(X)
        out = np.zeros(X.shape)
        for j in range(X.shape[1]):
            out[:, j] = [self.maps_[j].get(v, self.global_mean_) for v in X[:, j]]
        return out


class MEstimateEncoder(BaseEstimator, TransformerMixin):
    """Target mean shrunk toward the prior by a single smoothing count.

    The pragmatic cousin of James-Stein: instead of deriving the shrinkage from
    variances, add ``m`` pseudo-observations of the global mean to every
    category::

        encoding = (count * category_mean + m * global_mean) / (count + m)

    Larger ``m`` shrinks harder. One transparent knob, and it is the encoder the
    ``ensemble`` target-statistic used -- here as a standalone transformer.
    """

    def __init__(self, m=1.0):
        self.m = m

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, float)
        self.global_mean_ = y.mean()
        self.maps_ = []
        for j in range(X.shape[1]):
            mp = {}
            for cat in np.unique(X[:, j]):
                yc = y[X[:, j] == cat]
                mp[cat] = ((len(yc) * yc.mean() + self.m * self.global_mean_)
                           / (len(yc) + self.m))
            self.maps_.append(mp)
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = _as_2d(X)
        out = np.zeros(X.shape)
        for j in range(X.shape[1]):
            out[:, j] = [self.maps_[j].get(v, self.global_mean_) for v in X[:, j]]
        return out


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Encode each row with its category mean computed WITHOUT that row.

    THE LEAK, REMOVED DIRECTLY
    --------------------------
    Plain target encoding puts a row's own label into its category's mean, which
    hands the model the answer for singleton categories. Leave-one-out removes
    exactly that: at TRAINING time, row ``i``'s encoding is its category's mean
    over all OTHER rows, so its own label never appears. At TEST time (no labels
    to leak) the full category mean is used. It is the ordered-statistic idea from
    ``ensemble``, packaged as an encoder -- the cleanest fix when you can afford
    the row-specific computation.
    """

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, float)
        self.global_mean_ = y.mean()
        self.sums_, self.counts_ = [], []
        for j in range(X.shape[1]):
            s, c = {}, {}
            for cat in np.unique(X[:, j]):
                mask = X[:, j] == cat
                s[cat] = y[mask].sum()
                c[cat] = mask.sum()
            self.sums_.append(s)
            self.counts_.append(c)
        return self

    def fit_transform(self, X, y):
        """Training-time transform: each row's category mean EXCLUDING itself."""
        self.fit(X, y)
        X2 = _as_2d(X)
        y = np.asarray(y, float)
        out = np.zeros(X2.shape)
        for j in range(X2.shape[1]):
            for i in range(len(X2)):
                cat = X2[i, j]
                n = self.counts_[j][cat]
                if n > 1:
                    out[i, j] = (self.sums_[j][cat] - y[i]) / (n - 1)   # exclude self
                else:
                    out[i, j] = self.global_mean_
        return out

    def transform(self, X):
        """Test-time transform: the full category mean (no labels to leak here)."""
        check_is_fitted(self, "sums_")
        X = _as_2d(X)
        out = np.zeros(X.shape)
        for j in range(X.shape[1]):
            for i in range(len(X)):
                cat = X[i, j]
                if cat in self.counts_[j]:
                    out[i, j] = self.sums_[j][cat] / self.counts_[j][cat]
                else:
                    out[i, j] = self.global_mean_
        return out


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Represent a category by the BINARY DIGITS of its ordinal id.

    THE COMPRESSION
    ---------------
    One-hot needs one column per category. Binary encoding assigns each category
    an integer, then writes that integer in binary across ``ceil(log2(n))``
    columns -- so 256 categories need 8 columns, not 256. Target-free, so no leak,
    and far more compact than one-hot for high-cardinality features. The cost: the
    bit columns have no individual meaning (they are an arbitrary code), so it is
    less interpretable than one-hot and imposes a faint spurious structure two
    categories sharing bits are not actually related. A middle ground between
    one-hot's width and ordinal encoding's false ordering.
    """

    def fit(self, X, y=None):
        X = _as_2d(X)
        self.maps_ = []
        self.n_bits_ = []
        for j in range(X.shape[1]):
            cats = list(np.unique(X[:, j]))
            self.maps_.append({c: i + 1 for i, c in enumerate(cats)})  # ids from 1
            self.n_bits_.append(max(1, int(np.ceil(np.log2(len(cats) + 1)))))
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = _as_2d(X)
        blocks = []
        for j in range(X.shape[1]):
            ids = np.array([self.maps_[j].get(v, 0) for v in X[:, j]])
            bits = ((ids[:, None] >> np.arange(self.n_bits_[j])) & 1)
            blocks.append(bits)
        return np.hstack(blocks).astype(float)


class CountEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by how OFTEN it occurs. Target-free, one column.

    Sometimes a category's FREQUENCY is itself predictive (rare product codes
    behave differently from common ones), and count encoding captures exactly that
    in a single column with no target leak. Cheap and surprisingly effective as a
    feature alongside others; it discards the category's identity, keeping only
    its prevalence.
    """

    def __init__(self, normalize=False):
        self.normalize = normalize

    def fit(self, X, y=None):
        X = _as_2d(X)
        self.maps_ = []
        for j in range(X.shape[1]):
            cats, counts = np.unique(X[:, j], return_counts=True)
            if self.normalize:
                counts = counts / counts.sum()
            self.maps_.append(dict(zip(cats, counts)))
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = _as_2d(X)
        out = np.zeros(X.shape)
        for j in range(X.shape[1]):
            out[:, j] = [self.maps_[j].get(v, 0) for v in X[:, j]]
        return out


__all__ = ["WOEEncoder", "JamesSteinEncoder", "MEstimateEncoder",
           "LeaveOneOutEncoder", "BinaryEncoder", "CountEncoder"]
