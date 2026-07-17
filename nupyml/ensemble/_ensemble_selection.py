"""Caruana's ensemble selection: build a weighted blend by greedy addition.

Given a LIBRARY of trained models, which subset (and weights) blend best? Trying
all subsets is exponential. Ensemble selection greedily adds, one at a time, the
model that most improves the ensemble's validation score -- WITH REPLACEMENT, so
a strong model can be added several times, which is how integer counts become
real-valued weights.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array
from ..model_selection import train_test_split


class _BaseEnsembleSelection(BaseEstimator):
    """Caruana et al. (2004) greedy selection with replacement."""

    def __init__(self, estimators, n_rounds=50, val_fraction=0.3,
                 random_state=None):
        self.estimators = estimators
        self.n_rounds = n_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _fit_library(self, X, y):
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.random_state)
        self.fitted_ = [clone(e).fit(Xtr, ytr) for e in self.estimators]
        val_preds = [self._raw(e, Xval) for e in self.fitted_]
        return np.array(val_preds), yval               # (n_models, n_val[, ...])

    def _greedy(self, val_preds, yval):
        n_models = len(val_preds)
        counts = np.zeros(n_models)
        current = np.zeros_like(val_preds[0], dtype=float)
        n_added = 0
        for _ in range(self.n_rounds):
            best_score, best_m = -np.inf, None
            for m in range(n_models):
                blended = (current * n_added + val_preds[m]) / (n_added + 1)
                score = self._score(blended, yval)
                if score > best_score:
                    best_score, best_m = score, m
            counts[best_m] += 1
            current = (current * n_added + val_preds[best_m]) / (n_added + 1)
            n_added += 1
        self.weights_ = counts / counts.sum()
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._prepare(y)
        val_preds, yval = self._fit_library(X, y)
        return self._greedy(val_preds, yval)


class EnsembleSelectionClassifier(_BaseEnsembleSelection, ClassifierMixin):
    """Greedy ensemble selection for classification (blends probabilities).

    The blend averages predicted class PROBABILITIES; the greedy step maximises
    validation accuracy. Because selection is with replacement and driven purely
    by held-out score, it resists overfitting the ensemble far better than
    stacking a meta-learner on many correlated models, and it can only match or
    beat the single best library model on the validation set.
    """

    def _prepare(self, y):
        self.classes_ = np.unique(y)

    def _raw(self, est, X):
        return est.predict_proba(X)

    def _score(self, blended, yval):
        pred = self.classes_[blended.argmax(axis=1)]
        return np.mean(pred == yval)

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X)
        proba = sum(w * e.predict_proba(X)
                    for w, e in zip(self.weights_, self.fitted_) if w > 0)
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class EnsembleSelectionRegressor(_BaseEnsembleSelection, RegressorMixin):
    """Greedy ensemble selection for regression (blends predictions by MSE)."""

    def _prepare(self, y):
        pass

    def _raw(self, est, X):
        return est.predict(X)

    def _score(self, blended, yval):
        return -np.mean((blended - yval) ** 2)          # higher = better

    def predict(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X)
        return sum(w * e.predict(X)
                   for w, e in zip(self.weights_, self.fitted_) if w > 0)


__all__ = ["EnsembleSelectionClassifier", "EnsembleSelectionRegressor"]
