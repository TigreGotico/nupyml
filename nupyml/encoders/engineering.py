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


class MDLPDiscretizer(BaseEstimator, TransformerMixin):
    """Let the TARGET choose the bin edges (Fayyad & Irani, 1993).

    ``KBinsDiscretizer`` cuts a feature into equal-width or equal-frequency bins
    without ever looking at ``y`` -- so it will happily split a region where the
    class never changes and merge one where it flips every few samples. MDLP does
    the opposite: it recursively places a cut where it most reduces the class
    ENTROPY (maximum information gain), and -- crucially -- STOPS when the minimum
    description length says another cut costs more bits to encode than it saves.
    The result is a small number of bins that each mean something for the label,
    and a feature that a decision stump could split cleanly. Supervised, so it
    needs ``y`` at fit time; unfitted features with no useful cut get a single bin.
    """

    def __init__(self, min_gain=0.0):
        self.min_gain = min_gain

    @staticmethod
    def _entropy(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    def _best_cut(self, x, y):
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        n = len(ys)
        base_H = self._entropy(ys)
        # candidate cuts sit between adjacent distinct x where the class changes
        best_gain, best_t = -1.0, None
        for i in range(1, n):
            if xs[i] == xs[i - 1]:
                continue
            left, right = ys[:i], ys[i:]
            H = (len(left) / n) * self._entropy(left) + \
                (len(right) / n) * self._entropy(right)
            gain = base_H - H
            if gain > best_gain:
                best_gain, best_t = gain, (xs[i] + xs[i - 1]) / 2.0
        return best_gain, best_t, base_H

    def _mdl_accept(self, gain, y, left, right, n):
        # Fayyad-Irani MDLP criterion: accept the cut only if the information gain
        # exceeds the cost of encoding it (a per-partition MDL threshold).
        k = len(np.unique(y))
        k1, k2 = len(np.unique(left)), len(np.unique(right))
        delta = (np.log2(3 ** k - 2) -
                 (k * self._entropy(y) - k1 * self._entropy(left)
                  - k2 * self._entropy(right)))
        threshold = (np.log2(n - 1) + delta) / n
        return gain > threshold and gain > self.min_gain

    def _partition(self, x, y):
        gain, t, _ = self._best_cut(x, y)
        if t is None:
            return []
        mask = x <= t
        left, right = y[mask], y[~mask]
        if len(left) == 0 or len(right) == 0:
            return []
        if not self._mdl_accept(gain, y, left, right, len(y)):
            return []
        edges = [t]
        edges += self._partition(x[mask], left)          # recurse each side
        edges += self._partition(x[~mask], right)
        return edges

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        self.edges_ = []
        for j in range(X.shape[1]):
            cuts = sorted(self._partition(X[:, j], y))
            self.edges_.append(np.array(cuts))
        return self

    def transform(self, X):
        check_is_fitted(self, "edges_")
        X = check_array(X)
        cols = [np.searchsorted(self.edges_[j], X[:, j], side="right")
                for j in range(X.shape[1])]
        return np.column_stack(cols).astype(float)


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Explode a timestamp into the calendar features a model can actually use.

    A raw ``datetime64`` (or unix second) is a single huge monotone number -- a
    model sees "later" and nothing else. Almost all the signal in a timestamp is
    CALENDAR structure: the day of week, the month, the hour, whether it is a
    weekend. This unpacks each timestamp into those fields, and optionally the
    cyclical ``(sin, cos)`` encoding of the periodic ones so their wrap-around is
    respected (see ``CyclicalEncoder``). ``feature_names_`` records the columns.
    Accepts a 1-D array of ``numpy.datetime64`` or integer unix seconds.
    """

    def __init__(self, cyclical=True):
        self.cyclical = cyclical

    def _to_datetime64(self, x):
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.datetime64):
            return x.astype("datetime64[s]")
        return x.astype("datetime64[s]")             # unix seconds -> datetime64

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = self._to_datetime64(np.ravel(X))
        days = d.astype("datetime64[D]")
        year = days.astype("datetime64[Y]").astype(int) + 1970
        month = days.astype("datetime64[M]").astype(int) % 12 + 1
        day = (days - days.astype("datetime64[M]")).astype(int) + 1
        dow = (days.astype(int) + 3) % 7             # Monday = 0 (epoch is Thursday)
        doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
        hour = (d.astype("datetime64[h]") - days).astype(int)
        quarter = (month - 1) // 3 + 1
        is_weekend = (dow >= 5).astype(int)
        cols = {"year": year, "month": month, "day": day, "dayofweek": dow,
                "dayofyear": doy, "hour": hour, "quarter": quarter,
                "is_weekend": is_weekend}
        names, blocks = [], []
        for name, val in cols.items():
            names.append(name); blocks.append(val.astype(float))
        if self.cyclical:
            for name, val, period in [("month", month, 12), ("dayofweek", dow, 7),
                                      ("hour", hour, 24)]:
                angle = 2 * np.pi * val / period
                names += [f"{name}_sin", f"{name}_cos"]
                blocks += [np.sin(angle), np.cos(angle)]
        self.feature_names_ = names
        return np.column_stack(blocks)


__all__ = ["Winsorizer", "RareLabelEncoder", "CyclicalEncoder",
           "MDLPDiscretizer", "DateTimeFeatures"]
