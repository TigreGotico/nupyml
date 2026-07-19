"""Feature-weighted linear stacking: blend base models with input-dependent weights."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, clone
from ..utils import check_X_y, check_array, check_random_state


class FeatureWeightedLinearStacking(BaseEstimator, RegressorMixin):
    """Blend base models with weights that DEPEND on the input (Sill et al., 2009).

    Ordinary stacking learns one fixed weight per base model, but the best model often
    varies across the feature space -- model A wins for large x, model B for small.
    Feature-weighted linear stacking makes each blend weight a LINEAR function of
    (meta-)features, so the combination adapts per region while staying a single, cheap
    linear layer over the base predictions times the features. It was a winning trick
    in the Netflix Prize. Base models are fit on a hold-out split; the meta-layer is
    ridge-regressed on their out-of-fold predictions crossed with the features.
    """

    def __init__(self, estimators, alpha=1.0, val_fraction=0.5, random_state=None):
        self.estimators = estimators
        self.alpha = alpha
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        perm = rng.permutation(len(X))
        cut = int(len(X) * (1 - self.val_fraction))
        tr, va = perm[:cut], perm[cut:]
        self.experts_ = [clone(e).fit(X[tr], y[tr]) for e in self.estimators]
        P = np.column_stack([e.predict(X[va]) for e in self.experts_])
        # cross each base prediction with [1, features] -> per-region weights
        F = np.column_stack([np.ones(len(va)), X[va]])
        Z = np.einsum("nm,nf->nmf", P, F).reshape(len(va), -1)
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.w_ = np.linalg.solve(A, Z.T @ y[va])
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.column_stack([e.predict(X) for e in self.experts_])
        F = np.column_stack([np.ones(len(X)), X])
        Z = np.einsum("nm,nf->nmf", P, F).reshape(len(X), -1)
        return Z @ self.w_


__all__ = ["ProbabilisticRandomForest", "BayesianCommitteeMachine",
           "RegressionViaClassification", "FeatureWeightedLinearStacking"]


__all__ = ["FeatureWeightedLinearStacking"]
