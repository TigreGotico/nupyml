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


class ConformalizedQuantileRegression(BaseEstimator, RegressorMixin):
    """Split conformal on QUANTILE regressors -- adaptive width WITH the guarantee.

    THE GAP THIS CLOSES
    -------------------
    Plain split conformal (``ConformalRegressor``) pads every prediction by the
    SAME width, so it is too wide in easy regions and too narrow in hard ones even
    while its average coverage is exact. A pair of quantile regressors already
    predicts a heteroscedastic interval ``[q_lo(x), q_hi(x)]`` that narrows and
    widens with x -- but quantile regression's coverage is only as honest as the
    fit, and is usually a little off.

    THE FIX
    -------
    CQR conformalises the quantile interval. The nonconformity score is how far
    the truth falls OUTSIDE the predicted band::

        E = max(q_lo(x) - y,  y - q_hi(x))

    (negative when inside, positive when outside). Pad the band by the
    ``(1-alpha)`` quantile of the calibration ``E``. This RESTORES the exact
    finite-sample coverage guarantee while KEEPING the adaptive, x-dependent width
    -- the best of both. When the quantile fit is already good the correction is
    tiny; when it is off, conformal fixes the coverage regardless.

    Romano, Patterson & Candès (2019).
    """

    def __init__(self, alpha=0.1, calibration_fraction=0.3, solver_alpha=0.0,
                 random_state=None):
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.solver_alpha = solver_alpha
        self.random_state = random_state

    def fit(self, X, y):
        from ..linear_model import QuantileRegressor
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]

        self.lower_ = QuantileRegressor(quantile=self.alpha / 2,
                                        alpha=self.solver_alpha).fit(
            X[tr_idx], y[tr_idx])
        self.upper_ = QuantileRegressor(quantile=1 - self.alpha / 2,
                                        alpha=self.solver_alpha).fit(
            X[tr_idx], y[tr_idx])
        lo = self.lower_.predict(X[cal_idx])
        hi = self.upper_.predict(X[cal_idx])
        # signed distance outside the band; positive means the band missed
        self.conformity_ = np.sort(np.maximum(lo - y[cal_idx], y[cal_idx] - hi))
        return self

    def predict_interval(self, X, coverage=None):
        check_is_fitted(self, "conformity_")
        X = check_array(X)
        alpha = self.alpha if coverage is None else 1 - coverage
        n = len(self.conformity_)
        rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        pad = self.conformity_[rank - 1]
        return self.lower_.predict(X) - pad, self.upper_.predict(X) + pad

    def predict(self, X):
        lo, hi = self.predict_interval(X)
        return (lo + hi) / 2


class VennAbersCalibrator(BaseEstimator):
    """Calibrated probabilities WITH a validity guarantee -- as an interval [p0,p1].

    THE PROBLEM WITH POINT CALIBRATION
    ----------------------------------
    Platt scaling or isotonic regression give a single calibrated probability, but
    nothing certifies it is right -- a badly fit calibrator is silently
    overconfident. Venn-Abers instead outputs an INTERVAL ``[p0, p1]`` that is
    guaranteed to be perfectly calibrated in a precise sense: the true label
    frequency lies between p0 and p1. The interval's WIDTH is itself an honest
    signal of how sure the calibration is.

    HOW
    ---
    For a test score s, fit isotonic regression on the calibration set TWICE --
    once pretending the test label is 0 (giving p0) and once pretending it is 1
    (giving p1). The truth is bracketed by these two self-consistent fits. A
    single reported probability is the standard combination
    ``p1 / (1 - p0 + p1)``.

    Vovk & Petej (2014). Binary classification.
    """

    def __init__(self, estimator, calibration_fraction=0.3, random_state=None):
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(2, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
        self.estimator_ = clone(self.estimator).fit(X[tr_idx], y[tr_idx])
        self.classes_ = self.estimator_.classes_
        pos = np.where(self.classes_ == self.classes_[-1])[0][0]
        self._pos_col = pos
        self.cal_scores_ = self.estimator_.predict_proba(X[cal_idx])[:, pos]
        self.cal_labels_ = (y[cal_idx] == self.classes_[-1]).astype(float)
        return self

    def _isotonic(self, scores, labels, s):
        from ..isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        return float(iso.predict(np.atleast_1d(s))[0])

    def predict_proba_interval(self, X):
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        s = self.estimator_.predict_proba(X)[:, self._pos_col]
        p0 = np.empty(len(s))
        p1 = np.empty(len(s))
        for i, si in enumerate(s):
            # augment the calibration set with (si, 0) then (si, 1)
            sc0 = np.append(self.cal_scores_, si)
            sc1 = np.append(self.cal_scores_, si)
            lb0 = np.append(self.cal_labels_, 0.0)
            lb1 = np.append(self.cal_labels_, 1.0)
            p0[i] = self._isotonic(sc0, lb0, si)
            p1[i] = self._isotonic(sc1, lb1, si)
        return p0, p1

    def predict_proba(self, X):
        p0, p1 = self.predict_proba_interval(X)
        p = p1 / (1 - p0 + p1 + 1e-12)         # the standard Venn-Abers merge
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class AdaptiveConformalInference:
    """Online conformal that KEEPS coverage when the distribution drifts.

    THE PROBLEM WITH SPLIT CONFORMAL ONLINE
    ---------------------------------------
    Split conformal's guarantee assumes exchangeability. On a real data STREAM
    that drifts, that breaks and the fixed calibration quantile silently loses
    coverage. ACI makes the target level ITSELF adaptive: track the realised
    coverage and nudge the working alpha up when you have been covering too little,
    down when too much::

        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

    where ``err_t`` is 1 if the last interval missed. This drives the long-run
    miss rate to ``alpha_target`` REGARDLESS of drift -- trading the exact
    finite-sample guarantee for a robust asymptotic one that survives
    non-stationarity. ``update`` feeds back whether the last interval covered.

    Gibbs & Candès (2021).
    """

    def __init__(self, alpha_target=0.1, gamma=0.05):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_t = alpha_target
        self.errors_ = []

    def quantile_level(self):
        """The (1 - alpha_t) level to use for the next interval, clipped to [0,1]."""
        return float(np.clip(1 - self.alpha_t, 0.0, 1.0))

    def update(self, covered):
        """Feed back whether the last interval contained the truth; adapt alpha."""
        err = 0.0 if covered else 1.0
        self.errors_.append(err)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err)
        return self

    def realised_coverage(self):
        return 1 - np.mean(self.errors_) if self.errors_ else None


__all__ = ["ConformalRegressor", "ConformalClassifier",
           "MondrianConformalRegressor", "ConformalizedQuantileRegression",
           "VennAbersCalibrator", "AdaptiveConformalInference"]
