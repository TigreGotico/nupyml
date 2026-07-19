"""Adaptive Prediction Sets for classification (Romano et al., 2020)."""
import numpy as np
from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class APS(BaseEstimator):
    """Adaptive Prediction Sets for classification (Romano et al., 2020).

    Split-conformal classification (``ConformalClassifier``) thresholds
    ``1 - p_true`` and can produce badly-sized sets on hard examples. APS instead
    accumulates the SORTED class probabilities until the true class is included:
    the conformity score is the total probability mass of classes at least as
    likely as the truth. Calibrating that gives sets whose size ADAPTS to
    difficulty -- a singleton on easy inputs, several labels on ambiguous ones --
    while keeping the coverage guarantee. ``predict_set`` returns a set of labels
    per sample.
    """

    def __init__(self, estimator, alpha=0.1, calibration_fraction=0.3,
                 random_state=None):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def _score(self, proba, label):
        # total mass of classes at least as probable as the true class
        order = np.argsort(-proba)
        cum = 0.0
        for j in order:
            cum += proba[j]
            if j == label:
                break
        return cum

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal, tr = perm[:n_cal], perm[n_cal:]
        self.estimator_ = clone(self.estimator).fit(X[tr], y[tr])
        self.classes_ = self.estimator_.classes_
        c2i = {c: i for i, c in enumerate(self.classes_)}
        P = self.estimator_.predict_proba(X[cal])
        scores = np.array([self._score(P[i], c2i[y[cal][i]]) for i in range(len(cal))])
        self.tau_ = np.quantile(scores, 1 - self.alpha)
        return self

    def predict_set(self, X):
        check_is_fitted(self, "tau_")
        P = self.estimator_.predict_proba(check_array(X))
        sets = []
        for p in P:
            order = np.argsort(-p)
            cum, chosen = 0.0, []
            for j in order:
                chosen.append(self.classes_[j])
                cum += p[j]
                if cum >= self.tau_:
                    break
            sets.append(set(chosen))
        return sets

    def predict(self, X):
        return self.estimator_.predict(check_array(X))


__all__ = ["APS"]
