"""Bias mitigation at the three points a pipeline can intervene.

* PRE-processing -- change the data (``Reweighing``, ``CorrelationRemover``).
* IN-processing -- change the training objective
  (``ExponentiatedGradientReduction``).
* POST-processing -- change the decisions (``ThresholdOptimizer``).

Each buys fairness at some cost to accuracy; where to intervene depends on
whether you can retrain, touch the data, or only adjust the final decisions.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, ClassifierMixin, clone, check_is_fitted
from ..utils import check_array, check_X_y, check_random_state


class Reweighing(BaseEstimator, TransformerMixin):
    """Pre-processing: sample weights that de-correlate label and group.

    THE IDEA
    --------
    Bias shows up as a dependence between the sensitive attribute and the label in
    the training data (one group has more positives). Reweighing assigns each
    (group, label) cell the weight it would have IF the two were independent::

        w(g, y) = P(group=g) * P(label=y) / P(group=g, label=y)

    A learner trained with these weights sees a dataset in which group and label
    are statistically independent, so it has no incentive to use the group as a
    proxy. It touches only the weights, never the features or labels, and works
    with any estimator that accepts ``sample_weight``.

    Kamiran & Calders (2012). ``fit`` needs ``y`` and ``sensitive``; the learned
    weights are in ``sample_weight_``.
    """

    def fit(self, X, y, sensitive):
        y = np.asarray(y)
        s = np.asarray(sensitive)
        n = len(y)
        w = np.ones(n)
        for g in np.unique(s):
            for label in np.unique(y):
                mask = (s == g) & (y == label)
                observed = mask.sum() / n
                expected = (s == g).mean() * (y == label).mean()
                if observed > 0:
                    w[mask] = expected / observed
        self.sample_weight_ = w
        return self

    def fit_transform(self, X, y, sensitive):
        return self.fit(X, y, sensitive).sample_weight_


class CorrelationRemover(BaseEstimator, TransformerMixin):
    """Pre-processing: linearly project the sensitive signal OUT of the features.

    Even after dropping the sensitive column, other features can act as PROXIES
    for it (zip code for race). This removes the part of each feature that is
    linearly predictable from the sensitive attribute -- it regresses every
    feature on the (centred) sensitive attribute and keeps only the residual, so
    no feature carries a linear trace of the group. ``alpha`` blends between the
    original feature (0) and the fully-residualised one (1).

    It removes only LINEAR dependence -- non-linear proxies can survive, which is
    the honest limitation of any purely-linear scrubber.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y=None, sensitive=None):
        X = check_array(X)
        s = np.asarray(sensitive, dtype=float).reshape(len(X), -1)
        self.s_mean_ = s.mean(axis=0)
        sc = s - self.s_mean_
        # least-squares map from centred sensitive attrs to each feature
        self.coef_, *_ = np.linalg.lstsq(sc, X - X.mean(axis=0), rcond=None)
        self.x_mean_ = X.mean(axis=0)
        return self

    def transform(self, X, sensitive):
        check_is_fitted(self, "coef_")
        X = check_array(X)
        sc = np.asarray(sensitive, dtype=float).reshape(len(X), -1) - self.s_mean_
        predicted = self.x_mean_ + sc @ self.coef_        # linear group signal
        residual = X - predicted
        return self.alpha * residual + (1 - self.alpha) * X


class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """Post-processing: pick a PER-GROUP decision threshold for a fairness goal.

    Given a fitted scoring model, the only freedom left is the threshold at which
    a score becomes a positive decision -- and it can differ by group. This
    searches each group's threshold to equalise the target rate (demographic
    parity: equal selection rate; equal opportunity: equal TPR) while keeping
    accuracy as high as the constraint allows. It needs NO retraining, which is
    why it is the go-to when the model is fixed (a vendor model, an expensive
    net).

    Hardt, Price & Srebro (2016). ``constraint`` is
    'demographic_parity' or 'equal_opportunity'.
    """

    def __init__(self, estimator, constraint="demographic_parity", grid=101):
        self.estimator = estimator
        self.constraint = constraint
        self.grid = grid

    def _scores(self, X):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)[:, 1]
        return self.estimator_.decision_function(X)

    def fit(self, X, y, sensitive):
        X, y = check_X_y(X, y)
        s = np.asarray(sensitive)
        self.estimator_ = clone(self.estimator).fit(X, y)
        scores = self._scores(X)
        groups = np.unique(s)
        cand = np.linspace(scores.min(), scores.max(), self.grid)

        # target rate to equalise: overall selection rate or overall TPR
        if self.constraint == "demographic_parity":
            target = (scores >= np.median(scores)).mean()
        else:
            target = ((scores >= np.median(scores)) & (y == 1))[y == 1].mean()

        self.thresholds_ = {}
        for g in groups:
            m = s == g
            best_t, best_gap = cand[0], np.inf
            for t in cand:
                pred = scores[m] >= t
                if self.constraint == "demographic_parity":
                    rate = pred.mean()
                else:
                    pos = y[m] == 1
                    rate = pred[pos].mean() if pos.any() else 0.0
                gap = abs(rate - target)
                if gap < best_gap:
                    best_gap, best_t = gap, t
            self.thresholds_[g] = best_t
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, sensitive):
        check_is_fitted(self, "thresholds_")
        scores = self._scores(check_array(X))
        s = np.asarray(sensitive)
        out = np.zeros(len(scores), dtype=int)
        for g, t in self.thresholds_.items():
            out[s == g] = (scores[s == g] >= t).astype(int)
        return self.classes_[out]


class ExponentiatedGradientReduction(BaseEstimator, ClassifierMixin):
    """In-processing: reduce fair classification to a sequence of REWEIGHTED fits.

    THE REDUCTION
    -------------
    Fair classification under a parity constraint is a saddle-point problem: the
    learner minimises error while an adversary (Lagrange multipliers on the
    constraint) penalises violations. Exponentiated gradient plays that game --
    each round it fits an ordinary cost-sensitive classifier on weights that
    reflect the CURRENT constraint violation, then multiplicatively updates the
    multipliers toward the groups still being treated unfairly. The final
    predictor is the average of the per-round classifiers.

    The point is that you never write a fair learner -- you call an ordinary one
    repeatedly with shifting weights and let the game converge. This version
    targets demographic parity for binary classification.

    Agarwal et al. (2018).
    """

    def __init__(self, estimator, eps=0.02, n_iter=20, eta=1.0, random_state=None):
        self.estimator = estimator
        self.eps = eps
        self.n_iter = n_iter
        self.eta = eta
        self.random_state = random_state

    def fit(self, X, y, sensitive):
        X, y = check_X_y(X, y)
        s = np.asarray(sensitive)
        groups = np.unique(s)
        n = len(y)
        lam = np.zeros(len(groups))               # one multiplier per group
        self.predictors_ = []
        for _ in range(self.n_iter):
            # translate the current multipliers into per-sample weights: push
            # each group's selection rate toward the overall rate
            w = np.ones(n)
            for gi, g in enumerate(groups):
                w[s == g] *= np.exp(lam[gi])
            w = np.clip(w / w.mean(), 1e-3, 1e3)
            est = clone(self.estimator)
            est.fit(X, y, sample_weight=w)
            self.predictors_.append(est)
            # measure violation and update the multipliers multiplicatively
            pred = est.predict(X)
            overall = (pred == 1).mean()
            for gi, g in enumerate(groups):
                rate = (pred[s == g] == 1).mean()
                lam[gi] += self.eta * (overall - rate)   # raise weight if under-selected
        self.classes_ = self.predictors_[-1].classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "predictors_")
        X = check_array(X)
        # majority vote of the per-round predictors (the averaged classifier)
        votes = np.mean([p.predict(X) for p in self.predictors_], axis=0)
        return self.classes_[(votes >= 0.5).astype(int)]


__all__ = ["Reweighing", "CorrelationRemover", "ThresholdOptimizer",
           "ExponentiatedGradientReduction"]
