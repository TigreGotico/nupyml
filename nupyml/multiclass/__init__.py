"""Meta-estimators that turn binary learners into multiclass/multioutput ones."""
import itertools

import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                    check_is_fitted)
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, softmax


def _decision_or_proba(est, X):
    """Continuous confidence score for the positive class."""
    if hasattr(est, "decision_function"):
        d = est.decision_function(X)
        return d if d.ndim == 1 else d[:, 1]
    return est.predict_proba(X)[:, 1]


class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        self.estimators_ = []
        for c in range(len(self.classes_)):
            binary = (y_idx == c).astype(int)
            self.estimators_.append(clone(self.estimator).fit(X, binary))
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.column_stack([_decision_or_proba(e, X)
                                for e in self.estimators_])

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        proba = np.column_stack([e.predict_proba(X)[:, 1]
                                 for e in self.estimators_])
        total = proba.sum(axis=1, keepdims=True)
        return proba / np.where(total > 0, total, 1.0)


class OneVsOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        self.estimators_ = {}
        for a, b in itertools.combinations(range(len(self.classes_)), 2):
            mask = (y_idx == a) | (y_idx == b)
            binary = (y_idx[mask] == b).astype(int)
            self.estimators_[(a, b)] = clone(self.estimator).fit(X[mask], binary)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        k = len(self.classes_)
        votes = np.zeros((len(X), k))
        confidences = np.zeros((len(X), k))
        for (a, b), est in self.estimators_.items():
            score = _decision_or_proba(est, X)
            pred_b = est.predict(X) == 1
            votes[pred_b, b] += 1
            votes[~pred_b, a] += 1
            confidences[:, b] += score
            confidences[:, a] -= score
        # break vote ties with accumulated confidence
        return votes + confidences / (3 * (np.abs(confidences).max() + 1))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class OutputCodeClassifier(BaseEstimator, ClassifierMixin):
    """Error-correcting output codes."""

    def __init__(self, estimator, code_size=1.5, random_state=None):
        self.estimator = estimator
        self.code_size = code_size
        self.random_state = random_state

    def fit(self, X, y):
        from ..utils import check_random_state
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n_bits = max(2, int(k * self.code_size))
        # a decision_function is signed, so its code book must be too, else the
        # decoding distance compares unbounded scores against 0/1 targets
        self._signed = hasattr(self.estimator, "decision_function")
        low = -1.0 if self._signed else 0.0
        # the error-correcting property comes from row separation: keep the
        # best-separated candidate, rejecting duplicate rows (two classes made
        # indistinguishable) and constant columns (a one-label subproblem)
        from scipy.spatial.distance import pdist
        best_code, best_sep = None, -1.0
        for _ in range(100):
            code = np.where(rng.uniform(size=(k, n_bits)) > 0.5, 1.0, low)
            if any(len(np.unique(code[:, j])) < 2 for j in range(n_bits)):
                continue
            sep = pdist(code, metric="hamming").min()
            if sep > best_sep:
                best_code, best_sep = code, sep
        if best_code is None or best_sep <= 0:
            raise ValueError(
                f"Could not build a code book with distinct rows for {k} "
                f"classes in {n_bits} bits; increase code_size")
        self.code_book_ = best_code
        self.code_separation_ = float(best_sep)
        self.estimators_ = [clone(self.estimator).fit(X, self.code_book_[y_idx, bit])
                            for bit in range(n_bits)]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        # soft per-bit scores retain the confidence that hard 0/1 codes throw
        # away, which is what makes the error-correcting distance meaningful
        codes = np.column_stack([_decision_or_proba(e, X)
                                 for e in self.estimators_])
        from scipy.spatial.distance import cdist
        dist = cdist(codes, self.code_book_, metric="euclidean")
        return self.classes_[np.argmin(dist, axis=1)]


class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """One independent classifier per output column."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, Y):
        X = check_array(X)
        Y = np.asarray(Y)
        self.estimators_ = [clone(self.estimator).fit(X, Y[:, j])
                            for j in range(Y.shape[1])]
        self.classes_ = [e.classes_ for e in self.estimators_]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.column_stack([e.predict(X) for e in self.estimators_])

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return [e.predict_proba(X) for e in self.estimators_]

    def score(self, X, Y):
        Y = np.asarray(Y)
        return float(np.mean(self.predict(X) == Y))


class MultiOutputRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, Y):
        X = check_array(X)
        Y = np.asarray(Y, dtype=np.float64)
        self.estimators_ = [clone(self.estimator).fit(X, Y[:, j])
                            for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.column_stack([e.predict(X) for e in self.estimators_])

    def score(self, X, Y):
        from ..metrics import r2_score
        Y = np.asarray(Y, dtype=np.float64)
        pred = self.predict(X)
        return float(np.mean([r2_score(Y[:, j], pred[:, j])
                              for j in range(Y.shape[1])]))


class ClassifierChain(BaseEstimator, ClassifierMixin):
    """Chain of binary classifiers; each sees earlier predictions as features."""

    def __init__(self, estimator, order=None, random_state=None):
        self.estimator = estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, Y):
        from ..utils import check_random_state
        X = check_array(X)
        Y = np.asarray(Y)
        n_outputs = Y.shape[1]
        if self.order is None:
            self.order_ = np.arange(n_outputs)
        elif self.order == "random":
            self.order_ = check_random_state(self.random_state).permutation(n_outputs)
        else:
            self.order_ = np.asarray(self.order)
        self.estimators_ = []
        augmented = X
        for j in self.order_:
            est = clone(self.estimator).fit(augmented, Y[:, j])
            self.estimators_.append(est)
            augmented = np.hstack([augmented, Y[:, [j]]])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        out = np.zeros((len(X), len(self.order_)))
        augmented = X
        for est, j in zip(self.estimators_, self.order_):
            pred = est.predict(augmented)
            out[:, j] = pred
            augmented = np.hstack([augmented, pred[:, None]])
        return out

    def score(self, X, Y):
        return float(np.mean(self.predict(X) == np.asarray(Y)))


__all__ = ["OneVsRestClassifier", "OneVsOneClassifier", "OutputCodeClassifier",
           "MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain"]
