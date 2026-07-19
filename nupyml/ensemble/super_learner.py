"""The cross-validated OPTIMAL combination of base learners (van der Laan, 2007)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


class SuperLearner(BaseEstimator, RegressorMixin):
    """The cross-validated OPTIMAL combination of base learners (van der Laan, 2007).

    Stacking learns to combine base models, but a naive meta-learner can overfit
    their in-sample predictions. The super learner fixes the combination to the
    CROSS-VALIDATED out-of-fold predictions and finds the convex weights that
    minimise held-out error (non-negative, summing to one). Under mild conditions it
    is asymptotically as good as the best possible combination of the library -- the
    "oracle" property -- so adding weak or redundant learners cannot hurt. Base
    models are refit on all data; predictions use the learned weights.
    """

    def __init__(self, estimators, n_folds=5, random_state=None):
        self.estimators = estimators
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        from ..model_selection import KFold
        X = check_array(X); y = np.asarray(y, float)
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=check_random_state(self.random_state))
        Z = np.zeros((len(y), len(self.estimators)))     # out-of-fold predictions
        for tr, te in kf.split(X):
            for j, e in enumerate(self.estimators):
                Z[te, j] = clone(e).fit(X[tr], y[tr]).predict(X[te])
        # non-negative least squares for convex weights (projected gradient)
        w = np.ones(len(self.estimators)) / len(self.estimators)
        for _ in range(500):
            grad = Z.T @ (Z @ w - y) / len(y)
            w = w - 0.1 * grad
            w = np.maximum(w, 0)
            w = w / (w.sum() + 1e-12)
        self.weights_ = w
        self.models_ = [clone(e).fit(X, y) for e in self.estimators]
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.column_stack([m.predict(X) for m in self.models_])
        return P @ self.weights_


__all__ = ["SuperLearner"]
