"""Positive-Unlabeled learning by the Elkan-Noto method (2008)."""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class PUClassifier(BaseEstimator, ClassifierMixin):
    """Positive-Unlabeled learning by the Elkan-Noto method (2008).

    THE SETTING
    -----------
    You have some POSITIVE examples and a pile of UNLABELED ones -- no confirmed
    negatives (fraud you caught vs everything else; genes known to interact vs the
    rest). Training positive-vs-unlabeled naively is biased, because the unlabeled
    set contains hidden positives.

    THE ELKAN-NOTO CORRECTION
    -------------------------
    Under the "selected completely at random" assumption, a classifier ``g`` trained
    to predict LABELED-vs-unlabeled estimates ``c * P(y=1|x)`` where ``c =
    P(labeled | positive)`` is a constant. Estimate ``c`` as the average ``g`` on
    held-out KNOWN positives, then divide it out: ``P(y=1|x) = g(x) / c``. So a
    single calibration constant turns a biased PU classifier into an unbiased
    posterior. ``fit(X, s)`` with ``s=1`` for labelled-positive, ``s=0`` for
    unlabeled.
    """

    def __init__(self, estimator=None, calibration_fraction=0.3, random_state=None):
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, s):
        from ..linear_model import LogisticRegression
        X = check_array(X)
        s = np.asarray(s)
        rng = check_random_state(self.random_state)
        base = self.estimator or LogisticRegression(max_iter=500)
        pos = np.where(s == 1)[0]
        n_hold = max(1, int(self.calibration_fraction * len(pos)))
        hold = rng.permutation(pos)[:n_hold]           # held-out known positives
        train_mask = np.ones(len(s), bool); train_mask[hold] = False
        self.g_ = clone(base).fit(X[train_mask], s[train_mask])
        # c = average g on held-out positives = P(labeled | positive)
        self.c_ = float(self.g_.predict_proba(X[hold])[:, 1].mean())
        self.c_ = max(self.c_, 1e-3)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "g_")
        p = self.g_.predict_proba(check_array(X))[:, 1] / self.c_   # divide out c
        p = np.clip(p, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


__all__ = ["PUClassifier"]
