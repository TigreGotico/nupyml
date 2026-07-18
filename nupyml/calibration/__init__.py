"""Calibration: making predicted probabilities mean what they say.

A model can rank perfectly and still lie about probabilities. If you take every
case it called "70% likely" and only 40% of them happen, the ranking may be
flawless (AUC 1.0) while the number is useless for any decision that depends on
the actual risk -- pricing, triage, expected cost.

WHO NEEDS IT
------------
* **Naive Bayes** -- badly overconfident by construction. Its independence
  assumption double-counts correlated evidence, driving probabilities toward 0
  and 1.
* **SVM** -- a margin is a distance, not a probability, and has no
  probabilistic meaning at all.
* **Boosted trees** -- pushed toward extremes by the loss.
* **Bagged trees / random forests** -- pulled toward the middle by averaging.
* Logistic regression is generally fine already: it optimises log-loss, which
  IS a proper scoring rule, so calibration is what it was fitted for.

THE TWO METHODS
---------------
* **sigmoid** (Platt scaling) fits a 1-D logistic to the scores. Two parameters,
  so it works on little data, but it can only apply an S-shaped correction -- if
  the miscalibration has another shape, it cannot help.
* **isotonic** fits any monotone function. Strictly more flexible, and it will
  happily overfit on small data; it needs a few thousand samples to behave.

THE ESSENTIAL POINT
-------------------
Calibration must be fitted on data the model did not train on. A model's scores
on its own training data are already overconfident, so a calibrator fitted there
learns to correct a distortion that does not exist on new data -- and makes
things worse. That is why ``CalibratedClassifierCV`` cross-validates, and why
``cv="prefit"`` requires you to hand it a genuinely held-out set.

Calibration never changes the ranking (both maps are monotone), so AUC is
unchanged. Log loss and Brier score improve.
"""
import numpy as np
import scipy.optimize

from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..isotonic import IsotonicRegression
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, sigmoid


def _sigmoid_calibration(scores, targets, sample_weight=None):
    """Platt scaling: fit p = sigmoid(-(a*s + b)) by regularized MLE."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    w = np.ones(len(scores)) if sample_weight is None \
        else np.asarray(sample_weight, dtype=np.float64)
    # Platt's prior-corrected targets guard against overfit on small samples
    n_pos = w[targets == 1].sum()
    n_neg = w[targets == 0].sum()
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    t = np.where(targets == 1, hi, lo)

    def objective(theta):
        a, b = theta
        z = a * scores + b
        p = sigmoid(-z)
        eps = 1e-12
        loss = -np.sum(w * (t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)))
        dz = w * (t - p) * 1.0     # d/dz of the NLL with p = sigmoid(-z)
        return loss, np.array([np.sum(dz * scores), np.sum(dz)])

    res = scipy.optimize.minimize(objective, np.array([0.0, np.log((n_neg + 1)
                                                                   / (n_pos + 1))]),
                                  jac=True, method="L-BFGS-B")
    return res.x


class _SigmoidCalibrator:
    def fit(self, scores, targets, sample_weight=None):
        self.a_, self.b_ = _sigmoid_calibration(scores, targets, sample_weight)
        return self

    def predict(self, scores):
        return sigmoid(-(self.a_ * np.asarray(scores) + self.b_))


class _IsotonicCalibrator:
    def fit(self, scores, targets, sample_weight=None):
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip").fit(
            scores, targets, sample_weight=sample_weight)
        return self

    def predict(self, scores):
        return self._iso.predict(scores)


def _confidence_scores(est, X):
    if hasattr(est, "decision_function"):
        d = est.decision_function(X)
        return d if d.ndim == 1 else d
    p = est.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else p


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Calibrate a classifier's scores with cross-validated Platt/isotonic maps.

    ``cv="prefit"`` calibrates an already-fitted estimator on the given data.
    """

    def __init__(self, estimator=None, method="sigmoid", cv=5):
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def _new_calibrator(self):
        if self.method == "sigmoid":
            return _SigmoidCalibrator()
        if self.method == "isotonic":
            return _IsotonicCalibrator()
        raise ValueError(f"Unknown method: {self.method!r}")

    def _fit_one(self, est, X, y_idx):
        """Fit one calibrator per class from a fitted estimator's scores."""
        scores = _confidence_scores(est, X)
        k = len(self.classes_)
        cals = []
        if k == 2:
            cals.append(self._new_calibrator().fit(scores, (y_idx == 1).astype(float)))
        else:
            for c in range(k):
                col = scores[:, c] if scores.ndim == 2 else scores
                cals.append(self._new_calibrator().fit(
                    col, (y_idx == c).astype(float)))
        return cals

    def fit(self, X, y, sample_weight=None):
        from ..model_selection import _check_cv
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)

        self.calibrated_classifiers_ = []
        if self.cv == "prefit":
            est = self.estimator
            self.calibrated_classifiers_.append((est, self._fit_one(est, X, y_idx)))
            return self

        cv = _check_cv(self.cv, y, classifier=True)
        for train, test in cv.split(X, y):
            est = clone(self.estimator).fit(X[train], y[train])
            self.calibrated_classifiers_.append(
                (est, self._fit_one(est, X[test], y_idx[test])))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "calibrated_classifiers_")
        X = check_array(X)
        k = len(self.classes_)
        total = np.zeros((len(X), k))
        for est, cals in self.calibrated_classifiers_:
            scores = _confidence_scores(est, X)
            if k == 2:
                p = cals[0].predict(scores)
                total += np.column_stack([1 - p, p])
            else:
                cols = []
                for c, cal in enumerate(cals):
                    col = scores[:, c] if scores.ndim == 2 else scores
                    cols.append(cal.predict(col))
                P = np.column_stack(cols)
                s = P.sum(axis=1, keepdims=True)
                total += P / np.where(s > 0, s, 1.0)
        return total / len(self.calibrated_classifiers_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


from ._advanced import (expected_calibration_error, TemperatureScaling,  # noqa: E402
                        HistogramBinning, BetaCalibration)

__all__ = ["CalibratedClassifierCV", "expected_calibration_error",
           "TemperatureScaling", "HistogramBinning", "BetaCalibration"]
