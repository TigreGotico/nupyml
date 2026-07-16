"""Ensemble methods: bagging, random forests, boosting."""
import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                    check_is_fitted)
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state, softmax
from ._hist_gb import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class _BaseBagging(BaseEstimator):
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
    """Forest without bootstrap (extra randomness comes from max_features)."""
    _bootstrap_samples = False


class ExtraTreesRegressor(RandomForestRegressor):
    _bootstrap_samples = False


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """SAMME AdaBoost over decision stumps (or a given base estimator)."""

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
    """Gradient boosting with multinomial deviance (one tree per class/round)."""

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


__all__ = [
    "BaggingClassifier", "BaggingRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "VotingClassifier", "VotingRegressor",
    "StackingClassifier", "StackingRegressor",
]
