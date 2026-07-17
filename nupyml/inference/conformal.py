"""Conformal prediction: honest intervals from any model, almost assumption-free.

THE PROMISE
-----------
Wrap ANY model -- a forest, a neural net, a linear regression -- and get
prediction intervals with a GUARANTEED coverage rate. Ask for 90% and you get
intervals that contain the truth at least 90% of the time, and this holds for any
model, any data distribution, and finite samples. The only assumption is
EXCHANGEABILITY: the data can be shuffled without changing its distribution
(weaker than i.i.d., and usually true when i.i.d. is).

Set against the rest of this library, this is the important contrast. NGBoost and
the RVM report intervals from a MODEL of the noise, so their coverage is only as
good as that model -- and both are shown, in their own tests, to be
overconfident when the model is wrong. Conformal makes no such model. Its
guarantee is distributional, proven, and survives a badly misspecified predictor
entirely. When the interval MUST be trustworthy, this is the method.

HOW IT WORKS
------------
Split the data. Train on one part. On a held-out CALIBRATION set, measure the
NONCONFORMITY of each point -- for regression, simply ``|y - prediction|``, how
wrong the model was. Those calibration errors are a sample of how wrong the model
tends to be. To predict at a new point, take the model's output and pad it by the
appropriate QUANTILE of those calibration errors::

    interval = prediction +- quantile_{1-alpha}(calibration errors)

The logic is pure exchangeability: the new point's error is exchangeable with the
calibration errors, so it falls below their ``(1-alpha)`` quantile with
probability ``1-alpha``. No distributional assumption, no asymptotics -- just the
observation that one more draw from an exchangeable bag lands in a known fraction
of it.

THE HONEST LIMITATION
---------------------
Split conformal gives one interval WIDTH for every point. That is the guarantee's
price: it is marginally valid (right on average) but not adaptive -- it will be
too wide in easy regions and too narrow in hard ones, even while the overall
coverage is exactly right. ``MondrianConformalRegressor`` recovers some
adaptivity by calibrating per group, and normalised conformal (scaling the error
by a difficulty estimate) is the other standard fix.

Vovk, Gammerman & Shafer (2005); Lei et al. (2018).
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class ConformalRegressor(BaseEstimator, RegressorMixin):
    """Split-conformal prediction intervals around any regressor.

    ``estimator`` is any fitted-or-unfitted regressor. Fit trains it on a
    training split and calibrates on the rest; ``predict_interval`` then returns
    intervals with the guaranteed coverage.
    """

    def __init__(self, estimator, calibration_fraction=0.3, random_state=None):
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)

        # the split is not optional: calibrating on the training data would make
        # the errors look far smaller than they are on unseen points, and the
        # guarantee -- which rests on the calibration errors being exchangeable
        # with future errors -- would evaporate
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]

        self.estimator_ = clone(self.estimator).fit(X[train_idx], y[train_idx])
        # nonconformity = absolute residual on the calibration set
        self.calibration_scores_ = np.abs(
            y[cal_idx] - self.estimator_.predict(X[cal_idx]))
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))

    def predict_interval(self, X, coverage=0.9):
        """Intervals containing the truth with probability >= ``coverage``."""
        check_is_fitted(self, "estimator_")
        pred = self.predict(X)
        n = len(self.calibration_scores_)
        # the finite-sample correction: the ceil((n+1)(1-alpha))/n quantile,
        # NOT the plain empirical quantile. That +1 accounts for the new point
        # itself joining the calibration bag, and it is exactly what turns an
        # asymptotic statement into a finite-sample guarantee
        alpha = 1 - coverage
        rank = int(np.ceil((n + 1) * (1 - alpha)))
        rank = min(rank, n)                # if too few calibration points, widen
        q = np.sort(self.calibration_scores_)[rank - 1]
        return pred - q, pred + q


class MondrianConformalRegressor(BaseEstimator, RegressorMixin):
    """Conformal intervals calibrated separately per group -- adaptive width.

    Split conformal's one-size-fits-all width is its weakness: valid on average,
    but too wide where the model is accurate and too narrow where it struggles.
    The Mondrian variant partitions the data into groups (by a categorical
    feature, or by binning a difficulty score) and calibrates WITHIN each group.

    The result: each group gets its own interval width, so the intervals widen
    where prediction is genuinely hard and tighten where it is easy -- while the
    coverage guarantee still holds within every group, which is strictly stronger
    than holding only on average. The cost is that each group needs enough
    calibration points to estimate its own quantile, so very fine grouping
    starves the calibration and the guarantee degrades.
    """

    def __init__(self, estimator, group_fn, calibration_fraction=0.3,
                 random_state=None):
        self.estimator = estimator
        self.group_fn = group_fn            # X -> array of group labels
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]

        self.estimator_ = clone(self.estimator).fit(X[train_idx], y[train_idx])
        scores = np.abs(y[cal_idx] - self.estimator_.predict(X[cal_idx]))
        groups = self.group_fn(X[cal_idx])
        # a separate bag of calibration errors per group
        self.group_scores_ = {}
        for g in np.unique(groups):
            self.group_scores_[g] = np.sort(scores[groups == g])
        self._all_scores = np.sort(scores)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))

    def predict_interval(self, X, coverage=0.9):
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        pred = self.predict(X)
        groups = self.group_fn(X)
        alpha = 1 - coverage
        widths = np.empty(len(X))
        for i, g in enumerate(groups):
            # fall back to the pooled errors if a group was unseen or tiny
            scores = self.group_scores_.get(g, self._all_scores)
            if len(scores) < 1 / alpha:
                scores = self._all_scores
            nn = len(scores)
            rank = min(int(np.ceil((nn + 1) * (1 - alpha))), nn)
            widths[i] = scores[rank - 1]
        return pred - widths, pred + widths


class ConformalClassifier(BaseEstimator, ClassifierMixin):
    """Conformal prediction SETS for classification.

    The classification analogue returns not a single label but a SET of labels
    guaranteed to contain the true one with probability ``coverage``. That shift
    -- from a guess to a set -- is the honest output under uncertainty: on an easy
    example the set is a singleton, on a genuinely ambiguous one it holds several
    labels, and the SIZE of the set is itself a calibrated measure of difficulty.

    Nonconformity is ``1 - predicted_probability_of_the_true_class``: the model is
    "surprised" by a point exactly when it gave the right answer low probability.
    A new label is admitted to the set when its own nonconformity would not be
    unusually large against the calibration scores.
    """

    def __init__(self, estimator, calibration_fraction=0.3, random_state=None):
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]

        self.estimator_ = clone(self.estimator).fit(X[train_idx], y[train_idx])
        self.classes_ = self.estimator_.classes_
        proba = self.estimator_.predict_proba(X[cal_idx])
        class_to_col = {c: j for j, c in enumerate(self.classes_)}
        true_cols = np.array([class_to_col[c] for c in y[cal_idx]])
        # 1 - p(true class): large when the model was surprised by the truth
        self.calibration_scores_ = np.sort(
            1 - proba[np.arange(len(cal_idx)), true_cols])
        return self

    def predict_set(self, X, coverage=0.9):
        """A set of labels per sample, guaranteed to contain the truth."""
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        proba = self.estimator_.predict_proba(X)
        n = len(self.calibration_scores_)
        alpha = 1 - coverage
        rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        threshold = self.calibration_scores_[rank - 1]
        # a label is admitted when its nonconformity is not unusually large
        include = (1 - proba) <= threshold
        return [set(self.classes_[row]) for row in include]

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))


__all__ = ["ConformalRegressor", "ConformalClassifier",
           "MondrianConformalRegressor"]
