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

    def _fit_members(self, X, y, default):
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
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)


class BaggingClassifier(_BaseBagging, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._fit_members(X, y, DecisionTreeClassifier())
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
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self._fit_members(X, y, DecisionTreeRegressor())
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.mean([est.predict(X) for est in self.estimators_], axis=0)


class _BaseForest(BaseEstimator):
    def __init__(self, n_estimators=100, criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    _bootstrap_samples = True

    def _fit_forest(self, X, y, tree_cls, criterion):
        rng = check_random_state(self.random_state)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = tree_cls(
                criterion=criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.randint(0, 2 ** 31 - 1),
            )
            if self.bootstrap and self._bootstrap_samples:
                idx = rng.randint(0, n, size=n)
                tree.fit(X[idx], y[idx])
            else:
                tree.fit(X, y)
            self.estimators_.append(tree)


class RandomForestClassifier(_BaseForest, ClassifierMixin):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, random_state=None):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap, random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._fit_forest(X, y, DecisionTreeClassifier, self.criterion)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            proba += tree.predict_proba(X)
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseForest, RegressorMixin):
    def __init__(self, n_estimators=100, criterion="squared_error",
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=1.0, bootstrap=True, random_state=None):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap, random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self._fit_forest(X, y, DecisionTreeRegressor, self.criterion)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
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
            # weighted fit via weighted resampling
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
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(X)
        self.init_ = float(y.mean())
        pred = np.full(n, self.init_)
        self.estimators_ = []
        n_sub = int(self.subsample * n)
        for _ in range(self.n_estimators):
            resid = y - pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 2 ** 31 - 1))
            if self.subsample < 1.0:
                idx = rng.choice(n, size=n_sub, replace=False)
                tree.fit(X[idx], resid[idx])
            else:
                tree.fit(X, resid)
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
    def __init__(self, estimators, voting="hard"):
        self.estimators = estimators
        self.voting = voting

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
        proba = np.mean([est.predict_proba(check_array(X))
                         for _, est in self.estimators_], axis=0)
        return proba


__all__ = [
    "BaggingClassifier", "BaggingRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "VotingClassifier",
]
