"""Ensembles v5: uncertainty-aware forests, precision-weighted expert fusion, and
two ways to build a stronger predictor out of weaker ones.

Four more ensemble ideas. The probabilistic random forest treats every feature and
label as a DISTRIBUTION and propagates that uncertainty through the trees. The
Bayesian committee machine fuses several regressors by their PRECISION, so confident
experts count more. Regression-via-classification turns a hard regression into an
easier classification over bins. Feature-weighted linear stacking lets each base
model's weight DEPEND on the input, so the blend adapts per-region.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state


class ProbabilisticRandomForest(BaseEstimator, ClassifierMixin):
    """A forest that PROPAGATES feature uncertainty (Reis et al., 2019).

    A normal forest treats every measurement as exact, but real features come with
    error bars (a noisy sensor, a blurred photometry). The probabilistic random forest
    treats each feature value as a GAUSSIAN and, at prediction time, sends a sample
    down each tree drawn from that per-feature distribution -- repeated over many draws
    and many trees, so the vote reflects both the forest's disagreement AND the input's
    own uncertainty. Fitting is ordinary bagging; the probabilistic part is the
    sampled, uncertainty-aware inference. ``feature_sigma`` is the per-feature noise.
    """

    def __init__(self, n_estimators=25, max_depth=None, n_samples=10,
                 feature_sigma=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.feature_sigma = feature_sigma
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self.classes_ = np.unique(y)
        n = len(X)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n)                     # bootstrap sample
            t = DecisionTreeClassifier(max_depth=self.max_depth,
                                       random_state=rng.randint(1 << 30))
            t.fit(X[idx], y[idx])
            self.trees_.append(t)
        self._rng_seed = rng.randint(1 << 30)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        rng = check_random_state(self._rng_seed)
        sigma = np.atleast_1d(self.feature_sigma)
        votes = np.zeros((len(X), len(self.classes_)))
        for t in self.trees_:
            for _ in range(self.n_samples):
                # perturb inputs by their measurement noise, then vote
                Xs = X + rng.randn(*X.shape) * sigma
                for i, c in enumerate(t.predict(Xs)):
                    votes[i, np.searchsorted(self.classes_, c)] += 1
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class BayesianCommitteeMachine(BaseEstimator, RegressorMixin):
    """Fuse regressors by their PRECISION, not a flat average (Tresp, 2000).

    Split a big regression across several experts (each trained on part of the data)
    and you must recombine them -- but a plain average trusts a wildly uncertain expert
    as much as a confident one. The Bayesian committee machine weights each expert's
    prediction by its INVERSE VARIANCE (precision) and corrects for the shared prior,
    so where an expert is unsure its vote fades and the confident experts dominate. It
    is the principled way to pool independent probabilistic regressors. Each base
    estimator must expose ``predict(X, return_std=True)``.
    """

    def __init__(self, estimators, prior_var=1.0):
        self.estimators = estimators
        self.prior_var = prior_var

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(0)
        parts = np.array_split(rng.permutation(len(X)), len(self.estimators))
        self.experts_ = []
        for est, idx in zip(self.estimators, parts):
            e = clone(est)
            e.fit(X[idx], y[idx])                          # each expert sees a shard
            self.experts_.append(e)
        return self

    def predict(self, X, return_std=False):
        X = check_array(X)
        M = len(self.experts_)
        prec = np.zeros(len(X)); mean_acc = np.zeros(len(X))
        for e in self.experts_:
            mu, sd = e.predict(X, return_std=True)
            p = 1.0 / (sd ** 2 + 1e-12)                    # expert precision
            prec += p; mean_acc += p * mu
        # subtract the (M-1) times over-counted prior precision
        total_prec = prec - (M - 1) / self.prior_var
        total_prec = np.maximum(total_prec, 1e-12)
        mean = mean_acc / total_prec
        if return_std:
            return mean, np.sqrt(1.0 / total_prec)
        return mean


class RegressionViaClassification(BaseEstimator, RegressorMixin):
    """Turn regression into classification over BINS (Torgo & Gama, 1997).

    Some targets are easier to CLASSIFY than to regress -- a classifier can carve a
    non-monotone, multi-modal response that a single regressor smooths over. This
    discretises the target into bins, trains a classifier to predict the bin, and
    decodes back to a number as the class-probability-weighted average of the bin
    centres. The soft decode keeps the estimate continuous while borrowing the
    classifier's flexible decision boundaries. ``n_bins`` sets the granularity;
    ``strategy`` is 'quantile' (equal counts) or 'uniform' (equal width).
    """

    def __init__(self, classifier=None, n_bins=10, strategy="quantile"):
        self.classifier = classifier
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.strategy == "quantile":
            self.edges_ = np.quantile(y, np.linspace(0, 1, self.n_bins + 1))
            self.edges_ = np.unique(self.edges_)
        else:
            self.edges_ = np.linspace(y.min(), y.max(), self.n_bins + 1)
        bins = np.clip(np.digitize(y, self.edges_[1:-1]), 0, len(self.edges_) - 2)
        self.centres_ = np.array([y[bins == b].mean() if np.any(bins == b)
                                  else 0.5 * (self.edges_[b] + self.edges_[b + 1])
                                  for b in range(len(self.edges_) - 1)])
        clf = self.classifier
        if clf is None:
            clf = DecisionTreeClassifier(max_depth=6)
        self.clf_ = clone(clf).fit(X, bins)
        self.bin_labels_ = self.clf_.classes_
        return self

    def predict(self, X):
        X = check_array(X)
        proba = self.clf_.predict_proba(X)
        centres = self.centres_[self.bin_labels_]
        return proba @ centres                             # soft-decoded expectation


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
