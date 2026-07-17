"""Ensembles: many weak models beating one strong one.

Every method here combines trees. They divide into two families that attack
opposite halves of the error, and knowing which is which is most of the
intuition.

BAGGING: ATTACK THE VARIANCE
----------------------------
A deep tree has low bias and high variance -- it captures the true shape but
a slightly different training set gives a visibly different tree. Averaging
independent estimates cancels that noise: k independent estimates of the same
quantity have 1/k the variance.

But we only have one dataset. Bagging manufactures diversity by resampling it
with replacement (``bootstrap``), so each tree sees a different ~63% of the
data. The trees are correlated, not independent, so the variance reduction is
partial -- and that ceiling is exactly what ``RandomForestClassifier``
attacks, by also hiding features at each split so the trees cannot all seize
on the same dominant one.

Bagging deep trees is therefore the rule: you want each member overfit and
noisy, because averaging is what fixes it. Bias is untouched -- the average of
many unbiased trees is still unbiased.

BOOSTING: ATTACK THE BIAS
-------------------------
Boosting fits members SEQUENTIALLY, each one targeting what the ensemble so
far got wrong.

* ``AdaBoostClassifier`` reweights the data: misclassified samples get heavier,
  so the next learner concentrates where the ensemble is failing.
* ``GradientBoostingRegressor`` and ``HistGradientBoosting*`` fit the next
  tree to the RESIDUAL -- more precisely, to the negative gradient of the loss.
  Gradient descent, but taking steps in function space rather than parameter
  space: each tree is one step, and ``learning_rate`` is the step size.

Boosting members must be WEAK (shallow trees, often depth 1-3). A strong
learner would fit the residual perfectly on the first step, leaving nothing to
correct and reproducing a single overfit model. Reducing bias by construction,
boosting can and does overfit, which is what ``learning_rate``, ``subsample``
and early stopping are for.

  bagging:  deep trees, in parallel, independent   -> cuts variance
  boosting: shallow trees, in sequence, dependent  -> cuts bias

STACKING
--------
``StackingClassifier`` learns HOW to combine rather than assuming an average:
a meta-model is trained on the members' predictions. The subtlety is that
those predictions must be out-of-fold -- a member's opinion about data it
trained on is far too optimistic, and a meta-model fed such predictions learns
to trust the biggest overfitter.

MIXTURE OF EXPERTS: A THIRD WAY
-------------------------------
``MixtureOfExpertsRegressor`` does not average and does not sequence. A learned
GATE routes each input to whichever member specialises in it, so the members are
not interchangeable -- averaging them would be actively wrong. Use it when the
data is genuinely a union of regimes. It is also the idea behind sparse MoE
layers in large language models.

  bagging:  deep trees, in parallel, independent   -> cuts variance
  boosting: shallow trees, in sequence, dependent  -> cuts bias
  stacking: learn how to combine                   -> cuts both, carefully
  mixture:  learn who to ASK                       -> fits piecewise structure

THE FAMOUS VARIANTS (``_boosting_variants.py``)
----------------------------------------------
XGBoost, LightGBM and CatBoost are this same gradient boosting plus one or two
specific ideas each, and the ideas are smaller than the branding suggests:

* ``GOSSRegressor`` -- LightGBM's sampling: keep the big gradients, sample the
  rest, reweight the survivors. Trains on ~30% of the data for ~the same score.
* ``exclusive_feature_bundles`` -- LightGBM's other half: pack mutually
  exclusive (one-hot) features into a single column, losing nothing.
* ``ordered_target_statistic`` -- CatBoost's fix for a target leak that hides
  inside a preprocessing step everyone thought was innocent. Worth reading even
  if you never use CatBoost.
* ``DARTRegressor`` -- dropout for trees, so the late ones still matter.
* ``NGBoostRegressor`` -- predicts a DISTRIBUTION, and demonstrates why learned
  uncertainty is only as honest as the mean model's generalisation.
"""
import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                    check_is_fitted)
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state, softmax
from ._hist_gb import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class _BaseBagging(BaseEstimator):
    """Bootstrap AGGregatING: fit members on resamples, then average.

    Sampling n items from n with replacement leaves each member roughly 63% of
    the distinct samples -- the chance of a given sample never being drawn is
    ``(1 - 1/n)^n``, which tends to ``1/e ~ 0.368``. The ~37% left out are that
    tree's "out-of-bag" set: data it never saw, and therefore a free validation
    set requiring no holdout at all. That is what ``oob_score`` uses.
    """

    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _fit_members(self, X, y, default, sample_weight=None):
        rng = check_random_state(self.random_state)
        n = len(X)
        n_draw = int(self.max_samples * n) if isinstance(self.max_samples, float) \
            else int(self.max_samples)
        base = self.estimator if self.estimator is not None else default
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, size=n_draw) if self.bootstrap \
                else rng.choice(n, size=n_draw, replace=False)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=rng.randint(0, 2 ** 31 - 1))
            if sample_weight is not None and "sample_weight" in \
                    est.fit.__code__.co_varnames:
                est.fit(X[idx], y[idx], sample_weight=np.asarray(sample_weight)[idx])
            else:
                est.fit(X[idx], y[idx])
            self.estimators_.append(est)


class BaggingClassifier(_BaseBagging, ClassifierMixin):
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._fit_members(X, y, DecisionTreeClassifier(), sample_weight)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        k = len(self.classes_)
        proba = np.zeros((len(X), k))
        for est in self.estimators_:
            p = est.predict_proba(X)
            cols = np.searchsorted(self.classes_, est.classes_)
            proba[:, cols] += p
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BaggingRegressor(_BaseBagging, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        self._fit_members(X, y, DecisionTreeRegressor(), sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.mean([est.predict(X) for est in self.estimators_], axis=0)


class _BaseForest(BaseEstimator):
    """Bagging plus feature subsampling at every split.

    WHY THE EXTRA RANDOMNESS HELPS
    ------------------------------
    Averaging k estimators with pairwise correlation ``rho`` leaves variance::

        rho * sigma^2  +  (1 - rho) * sigma^2 / k

    The second term vanishes as trees are added. The first does not -- it is a
    floor set by how alike the trees are. Adding trees past that point buys
    nothing.

    Bootstrapping alone leaves the trees quite correlated: one dominant feature
    gets picked at the root of nearly every tree. So a forest also hides a
    random subset of features at EACH split (``max_features``). Now the
    dominant feature is unavailable much of the time, other structure gets
    used, and ``rho`` falls -- lowering the floor itself.

    Each tree is individually WORSE for being denied its best feature. The
    ensemble is better. That trade is the whole idea.

    ``sqrt(d)`` features is the usual default for classification, ``d`` for
    regression, where there is typically less to gain.
    """

    def __init__(self, n_estimators=100, criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, ccp_alpha=0.0, monotonic_cst=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.random_state = random_state

    _bootstrap_samples = True

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimators_")
        return np.mean([t.feature_importances_ for t in self.estimators_], axis=0)

    def _fit_forest(self, X, y, tree_cls, criterion, sample_weight=None):
        if getattr(self, "warm_start", False) and hasattr(self, "estimators_"):
            existing = self.estimators_
            rng = self._rng
        else:
            existing = []
            rng = check_random_state(self.random_state)
            self._rng = rng
            self._oob_idx = []
        n = len(X)
        self.estimators_ = list(existing)
        w = None if sample_weight is None else np.asarray(sample_weight,
                                                          dtype=np.float64)
        for _ in range(self.n_estimators - len(existing)):
            tree = tree_cls(
                criterion=criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                ccp_alpha=self.ccp_alpha,
                monotonic_cst=self.monotonic_cst,
                random_state=rng.randint(0, 2 ** 31 - 1),
            )
            if self.bootstrap and self._bootstrap_samples:
                idx = rng.randint(0, n, size=n)
                tree.fit(X[idx], y[idx], sample_weight=None if w is None else w[idx])
                self._oob_idx.append(np.setdiff1d(np.arange(n), idx))
            else:
                tree.fit(X, y, sample_weight=w)
                self._oob_idx.append(np.array([], dtype=int))
            self.estimators_.append(tree)


class RandomForestClassifier(_BaseForest, ClassifierMixin):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, oob_score=False, warm_start=False,
                 ccp_alpha=0.0, monotonic_cst=None, random_state=None):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap, ccp_alpha,
                         monotonic_cst, random_state)
        self.oob_score = oob_score
        self.warm_start = warm_start

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, force_all_finite="allow-nan")
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._fit_forest(X, y, DecisionTreeClassifier, self.criterion,
                         sample_weight)
        if self.oob_score:
            k = len(self.classes_)
            votes = np.zeros((len(X), k))
            for tree, oob in zip(self.estimators_, self._oob_idx):
                if len(oob):
                    votes[oob] += tree.predict_proba(X[oob])
            covered = votes.sum(axis=1) > 0
            pred = self.classes_[np.argmax(votes[covered], axis=1)]
            self.oob_score_ = float(np.mean(pred == y[covered]))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, force_all_finite="allow-nan")
        proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            proba += tree.predict_proba(X)
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseForest, RegressorMixin):
    def __init__(self, n_estimators=100, criterion="squared_error",
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=1.0, bootstrap=True, oob_score=False,
                 warm_start=False, ccp_alpha=0.0, monotonic_cst=None,
                 random_state=None):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap, ccp_alpha,
                         monotonic_cst, random_state)
        self.oob_score = oob_score
        self.warm_start = warm_start

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite="allow-nan")
        self._fit_forest(X, y, DecisionTreeRegressor, self.criterion,
                         sample_weight)
        if self.oob_score:
            preds = np.zeros(len(X))
            counts = np.zeros(len(X))
            for tree, oob in zip(self.estimators_, self._oob_idx):
                if len(oob):
                    preds[oob] += tree.predict(X[oob])
                    counts[oob] += 1
            covered = counts > 0
            from ..metrics import r2_score
            self.oob_score_ = r2_score(y[covered], preds[covered] / counts[covered])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, force_all_finite="allow-nan")
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)


class ExtraTreesClassifier(RandomForestClassifier):
    """Extremely randomised trees: no bootstrap, more randomness per split.

    Pushes the decorrelation argument further. Each tree sees all the data --
    no bootstrap -- but randomness at the splits keeps them diverse. Trading
    even more individual accuracy for even less correlation, this often matches
    a random forest while fitting faster, since less effort goes into choosing
    thresholds carefully.
    """
    _bootstrap_samples = False


class ExtraTreesRegressor(RandomForestRegressor):
    _bootstrap_samples = False


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """Adaptive boosting: reweight the data toward what the ensemble gets wrong.

    Each round fits a weak learner on weighted data, then:

    1. measures its weighted error ``err``;
    2. gives it a vote ``alpha = log((1-err)/err) + log(K-1)`` -- large when the
       learner is accurate, zero at chance, and NEGATIVE if it is worse than
       chance (in which case believing its opposite is informative);
    3. multiplies the weight of every misclassified sample by ``exp(alpha)``,
       so the next learner is forced to attend to them.

    Points that are easy get quietly ignored; the ensemble's attention
    concentrates on the boundary. The ``log(K-1)`` term is what makes this SAMME
    rather than the original binary AdaBoost: with K classes, chance is 1/K, not
    1/2, so the bar a learner must clear to earn a positive vote is lower.

    AdaBoost is exactly forward stagewise additive modelling under EXPONENTIAL
    loss, which explains its one real weakness: ``exp`` punishes a badly
    misclassified point enormously, so a mislabelled sample can capture the
    whole ensemble's attention. Gradient boosting with a gentler loss is the
    usual answer.
    """

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n = len(X)
        rng = check_random_state(self.random_state)
        base = self.estimator if self.estimator is not None else \
            DecisionTreeClassifier(max_depth=1)
        w = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.estimator_weights_ = []
        for _ in range(self.n_estimators):
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=rng.randint(0, 2 ** 31 - 1))
            if "sample_weight" in est.fit.__code__.co_varnames:
                est.fit(X, y_idx, sample_weight=w)
            else:  # fall back to weighted resampling
                idx = rng.choice(n, size=n, p=w)
                est.fit(X[idx], y_idx[idx])
            pred = est.predict(X)
            err = np.sum(w * (pred != y_idx)) / w.sum()
            if err >= 1.0 - 1.0 / k:
                continue
            err = max(err, 1e-10)
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(k - 1))
            if alpha <= 0:
                break
            w *= np.exp(alpha * (pred != y_idx))
            w /= w.sum()
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            if err < 1e-9:
                break
        if not self.estimators_:  # degenerate: keep one stump
            est = clone(base).fit(X, y_idx)
            self.estimators_.append(est)
            self.estimator_weights_.append(1.0)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        k = len(self.classes_)
        scores = np.zeros((len(X), k))
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X).astype(int)
            scores[np.arange(len(X)), pred] += alpha
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """Gradient descent in function space, one tree per step.

    THE IDEA
    --------
    To minimise a loss over a FUNCTION rather than a parameter vector, ask what
    gradient descent would say. The derivative of squared loss with respect to
    the current prediction is ``pred - y``, so the negative gradient is
    ``y - pred``: the residual. Fitting a tree to the residuals and adding a
    small multiple of it is one step of gradient descent, where the "parameter"
    being updated is the whole prediction function::

        F_{m+1}(x) = F_m(x) + learning_rate * tree_m(x)

    Squared loss makes the negative gradient literally the residual, which is
    why it is usually introduced as "fit the next tree to the errors". For
    other losses the target is the gradient rather than the residual -- the
    algorithm is unchanged, which is the point of the framing.

    THE KNOBS ARE ALL THE SAME KNOB
    -------------------------------
    ``learning_rate`` shrinks each step. Small steps mean more trees to reach
    the same fit, but a smoother path and better generalisation -- the standard
    trade is a low rate and many trees. It interacts with ``n_estimators``
    directly: halving one roughly wants doubling the other.

    ``subsample`` < 1 fits each tree on a random portion of the data
    ("stochastic gradient boosting"), which both decorrelates the trees and
    speeds things up.

    ``max_depth`` caps interaction order: depth 1 (a stump) is an additive
    model with no interactions at all, depth 2 allows pairwise, and so on.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, warm_start=False,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.warm_start = warm_start
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        n = len(X)
        w = None if sample_weight is None else np.asarray(sample_weight,
                                                          dtype=np.float64)
        if self.warm_start and hasattr(self, "estimators_"):
            rng = self._rng
            pred = self.init_ + self.learning_rate * np.sum(
                [t.predict(X) for t in self.estimators_], axis=0)
        else:
            rng = check_random_state(self.random_state)
            self._rng = rng
            self.init_ = float(np.average(y, weights=w))
            pred = np.full(n, self.init_)
            self.estimators_ = []
        n_sub = int(self.subsample * n)
        for _ in range(self.n_estimators - len(self.estimators_)):
            resid = y - pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 2 ** 31 - 1))
            if self.subsample < 1.0:
                idx = rng.choice(n, size=n_sub, replace=False)
                tree.fit(X[idx], resid[idx],
                         sample_weight=None if w is None else w[idx])
            else:
                tree.fit(X, resid, sample_weight=w)
            pred += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        pred = np.full(len(X), self.init_)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)
        return pred


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting for classification: K trees per round, one per class.

    Trees output numbers, not probabilities, so the ensemble builds a SCORE per
    class and softmax turns the scores into probabilities at the end -- exactly
    the arrangement in ``LogisticRegression``, but with the linear score
    replaced by a sum of trees.

    Under multinomial deviance the negative gradient for class k is::

        y_k - p_k

    the same "observed minus predicted" as logistic regression, so each round
    fits one tree per class to that residual. K trees per round is the cost of
    letting classes have independent score functions.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n = len(X)
        Y = np.eye(k)[y_idx]
        prior = np.clip(Y.mean(axis=0), 1e-12, None)
        self.init_ = np.log(prior)
        F = np.tile(self.init_, (n, 1))
        self.estimators_ = []
        n_sub = int(self.subsample * n)
        for _ in range(self.n_estimators):
            P = softmax(F, axis=1)
            round_trees = []
            for c in range(k):
                resid = Y[:, c] - P[:, c]
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=rng.randint(0, 2 ** 31 - 1))
                if self.subsample < 1.0:
                    idx = rng.choice(n, size=n_sub, replace=False)
                    tree.fit(X[idx], resid[idx])
                else:
                    tree.fit(X, resid)
                F[:, c] += self.learning_rate * tree.predict(X)
                round_trees.append(tree)
            self.estimators_.append(round_trees)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        F = np.tile(self.init_, (len(X), 1))
        for round_trees in self.estimators_:
            for c, tree in enumerate(round_trees):
                F[:, c] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting="hard", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.estimators_ = [(name, clone(est).fit(X, y))
                            for name, est in self.estimators]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        preds = np.array([est.predict(X) for _, est in self.estimators_])
        out = []
        for col in preds.T:
            vals, counts = np.unique(col, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.array(out)

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        proba = np.average([est.predict_proba(check_array(X))
                            for _, est in self.estimators_], axis=0,
                           weights=self.weights)
        return proba




class _BaseStacking(BaseEstimator):
    """Fit base estimators, then a meta-learner on their cross-val predictions."""

    def __init__(self, estimators, final_estimator=None, cv=5,
                 passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def _meta_features(self, X, y):
        from ..model_selection import _check_cv
        is_clf = self._estimator_type == "classifier"
        cv = _check_cv(self.cv, y, classifier=is_clf)
        blocks = []
        for _, est in self.estimators:
            width = self._block_width(est, y)
            oof = np.zeros((len(X), width))
            for train, test in cv.split(X, y):
                fitted = clone(est).fit(X[train], y[train])
                oof[test] = self._transform_one(fitted, X[test])
            blocks.append(oof)
        return np.hstack(blocks)

    def fit(self, X, y):
        X, y = check_X_y(X, y) if self._estimator_type == "classifier" \
            else check_X_y(X, y, y_numeric=True)
        self._prepare(y)
        meta = self._meta_features(X, y)
        if self.passthrough:
            meta = np.hstack([meta, X])
        # base estimators are refit on the full data for prediction time
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.final_estimator_ = clone(
            self.final_estimator if self.final_estimator is not None
            else self._default_final()).fit(meta, y)
        return self

    def _build_meta(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        meta = np.hstack([self._transform_one(est, X) for est in self.estimators_])
        return np.hstack([meta, X]) if self.passthrough else meta

    def transform(self, X):
        return self._build_meta(X)


class StackingClassifier(_BaseStacking, ClassifierMixin):
    _estimator_type = "classifier"

    def _prepare(self, y):
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

    def _default_final(self):
        from ..linear_model import LogisticRegression
        return LogisticRegression()

    def _block_width(self, est, y):
        # drop one column for binary targets: the probabilities are redundant
        k = len(np.unique(y))
        return 1 if k == 2 else k

    @staticmethod
    def _transform_one(est, X):
        proba = est.predict_proba(X)
        return proba[:, 1:] if proba.shape[1] == 2 else proba

    def predict(self, X):
        return self.final_estimator_.predict(self._build_meta(X))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._build_meta(X))


class StackingRegressor(_BaseStacking, RegressorMixin):
    _estimator_type = "regressor"

    def _prepare(self, y):
        pass

    def _default_final(self):
        from ..linear_model import Ridge
        return Ridge()

    def _block_width(self, est, y):
        return 1

    @staticmethod
    def _transform_one(est, X):
        return est.predict(X)[:, None]

    def predict(self, X):
        return self.final_estimator_.predict(self._build_meta(X))


class VotingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.estimators_ = [(name, clone(est).fit(X, y))
                            for name, est in self.estimators]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        preds = np.column_stack([est.predict(X) for _, est in self.estimators_])
        return np.average(preds, axis=1, weights=self.weights)


from ._extra import AdaBoostRegressor, RandomTreesEmbedding
from ._boosting_variants import (
    GOSSRegressor, OrderedBoostingRegressor, DARTRegressor, NGBoostRegressor,
    goss_sample, exclusive_feature_bundles, bundle_features,
    ordered_target_statistic,
)
from ._mixture_of_experts import (
    MixtureOfExpertsClassifier, MixtureOfExpertsRegressor,
)

__all__ = [
    "BaggingClassifier", "BaggingRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "VotingClassifier", "VotingRegressor",
    "StackingClassifier", "StackingRegressor",
    "RandomTreesEmbedding",
    "GOSSRegressor", "OrderedBoostingRegressor", "DARTRegressor",
    "NGBoostRegressor", "goss_sample", "exclusive_feature_bundles",
    "bundle_features", "ordered_target_statistic",
    "MixtureOfExpertsClassifier", "MixtureOfExpertsRegressor",
]
