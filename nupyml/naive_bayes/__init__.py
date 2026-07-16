"""Naive Bayes classifiers."""
import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, softmax


class _BaseNB(BaseEstimator, ClassifierMixin):
    def _joint_log_likelihood(self, X):
        raise NotImplementedError

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        from scipy.special import logsumexp
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


class GaussianNB(_BaseNB):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        d = X.shape[1]
        self.theta_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        self.class_prior_ = np.zeros(k)
        for c in range(k):
            mask = y_idx == c
            wc = w[mask]
            self.theta_[c] = np.average(X[mask], axis=0, weights=wc)
            self.var_[c] = np.average((X[mask] - self.theta_[c]) ** 2, axis=0,
                                      weights=wc)
            self.class_prior_[c] = wc.sum() / w.sum()
        self.var_ += self.var_smoothing * X.var(axis=0).max()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit via per-class running sums."""
        X, y = check_X_y(X, y)
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        if not hasattr(self, "_sums"):
            if classes is None:
                raise ValueError("classes must be passed on the first call")
            self.classes_ = np.asarray(classes)
            self._le = LabelEncoder()
            self._le.classes_ = self.classes_
            k, d = len(self.classes_), X.shape[1]
            self._sums = np.zeros((k, d))
            self._sq_sums = np.zeros((k, d))
            self._counts = np.zeros(k)
            self._var_floor = 0.0
        y_idx = self._le.transform(y)
        for c in range(len(self.classes_)):
            mask = y_idx == c
            if mask.any():
                wc = w[mask][:, None]
                self._sums[c] += (X[mask] * wc).sum(axis=0)
                self._sq_sums[c] += (X[mask] ** 2 * wc).sum(axis=0)
                self._counts[c] += w[mask].sum()
        self._var_floor = max(self._var_floor,
                              self.var_smoothing * X.var(axis=0).max())
        seen = self._counts > 0
        counts = np.where(seen, self._counts, 1.0)[:, None]
        self.theta_ = self._sums / counts
        self.var_ = np.maximum(self._sq_sums / counts - self.theta_ ** 2, 0.0) \
            + self._var_floor
        self.class_prior_ = np.where(seen, self._counts, 0.0) / self._counts.sum()
        self.class_prior_ = np.maximum(self.class_prior_, 1e-12)
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "theta_")
        X = check_array(X)
        jll = np.zeros((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            log_prob = -0.5 * (np.log(2 * np.pi * self.var_[c])
                               + (X - self.theta_[c]) ** 2 / self.var_[c]).sum(axis=1)
            jll[:, c] = np.log(self.class_prior_[c]) + log_prob
        return jll


class _DiscreteNB(_BaseNB):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _count(self, X, Y):
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=True)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        if sample_weight is not None:
            Y = Y * np.asarray(sample_weight, dtype=np.float64)[:, None]
        self.class_count_ = Y.sum(axis=0)
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        self._fit_counts(X, Y)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=True)
        if not hasattr(self, "_acc_fc"):
            if classes is None:
                raise ValueError("classes must be passed on the first call")
            self.classes_ = np.asarray(classes)
            self._le = LabelEncoder()
            self._le.classes_ = self.classes_
            self._acc_fc = None
            self._acc_cc = np.zeros(len(self.classes_))
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        if sample_weight is not None:
            Y = Y * np.asarray(sample_weight, dtype=np.float64)[:, None]
        import scipy.sparse as _sp
        Xd = np.asarray(X.todense()) if _sp.issparse(X) else X
        fc = Y.T @ Xd
        self._acc_fc = fc if self._acc_fc is None else self._acc_fc + fc
        self._acc_cc += Y.sum(axis=0)
        self.class_count_ = self._acc_cc
        self.class_log_prior_ = np.log(self._acc_cc / self._acc_cc.sum())
        self._fit_from_accumulated(self._acc_fc, self._acc_cc)
        return self

    def _fit_from_accumulated(self, fc, cc):
        raise NotImplementedError


class MultinomialNB(_DiscreteNB):
    def _fit_counts(self, X, Y):
        fc = np.asarray(Y.T @ X if not sp.issparse(X) else (sp.csr_matrix(Y.T) @ X).todense())
        self._fit_from_accumulated(np.asarray(fc), None)

    def _fit_from_accumulated(self, fc, cc):
        fc = fc + self.alpha
        self.feature_log_prob_ = np.log(fc) - np.log(fc.sum(axis=1, keepdims=True))

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "feature_log_prob_")
        X = check_array(X, accept_sparse=True)
        jll = X @ self.feature_log_prob_.T
        return np.asarray(jll) + self.class_log_prior_


class ComplementNB(_DiscreteNB):
    def _fit_counts(self, X, Y):
        Xd = np.asarray(X.todense()) if sp.issparse(X) else X
        self._fit_from_accumulated(Y.T @ Xd, Y.sum(axis=0))

    def _fit_from_accumulated(self, fc, cc):
        comp = fc.sum(axis=0, keepdims=True) - fc + self.alpha
        logw = np.log(comp / comp.sum(axis=1, keepdims=True))
        self.feature_log_prob_ = -logw

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "feature_log_prob_")
        X = check_array(X, accept_sparse=True)
        return np.asarray(X @ self.feature_log_prob_.T)


class BernoulliNB(_DiscreteNB):
    def __init__(self, alpha=1.0, binarize=0.0):
        super().__init__(alpha=alpha)
        self.binarize = binarize

    def _fit_counts(self, X, Y):
        Xd = np.asarray(X.todense()) if sp.issparse(X) else X
        if self.binarize is not None:
            Xd = (Xd > self.binarize).astype(np.float64)
        self._fit_from_accumulated(Y.T @ Xd, Y.sum(axis=0))

    def _fit_from_accumulated(self, fc, cc):
        fc = fc + self.alpha
        cc = cc[:, None] + 2 * self.alpha
        self.feature_log_prob_ = np.log(fc / cc)
        self._neg_log_prob = np.log(1 - fc / cc)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        import scipy.sparse as _sp
        Xd = np.asarray(X.todense()) if _sp.issparse(X) else np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            Xd = (Xd > self.binarize).astype(np.float64)
        return super().partial_fit(Xd, y, classes=classes,
                                   sample_weight=sample_weight)

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "feature_log_prob_")
        X = check_array(X, accept_sparse=True)
        Xd = np.asarray(X.todense()) if sp.issparse(X) else X
        if self.binarize is not None:
            Xd = (Xd > self.binarize).astype(np.float64)
        jll = Xd @ self.feature_log_prob_.T + (1 - Xd) @ self._neg_log_prob.T
        return jll + self.class_log_prior_


class CategoricalNB(_BaseNB):
    """Naive Bayes for categorical features: each feature has its own
    conditional distribution over its own categories."""

    def __init__(self, alpha=1.0, min_categories=None):
        self.alpha = alpha
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        Xi = X.astype(np.int64)
        if (Xi < 0).any():
            raise ValueError("CategoricalNB requires non-negative integer features")
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k, d = len(self.classes_), X.shape[1]
        self.class_count_ = np.array([w[y_idx == c].sum() for c in range(k)])
        self.class_log_prior_ = np.log(self.class_count_ / w.sum())
        self.n_categories_ = np.array([
            max(int(Xi[:, j].max()) + 1,
                self.min_categories or 0) for j in range(d)])
        self.category_log_prob_ = []
        for j in range(d):
            n_cat = self.n_categories_[j]
            counts = np.zeros((k, n_cat))
            np.add.at(counts, (y_idx, Xi[:, j]), w)
            counts += self.alpha
            self.category_log_prob_.append(
                np.log(counts / counts.sum(axis=1, keepdims=True)))
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "category_log_prob_")
        Xi = check_array(X).astype(np.int64)
        jll = np.tile(self.class_log_prior_, (len(Xi), 1))
        for j, log_prob in enumerate(self.category_log_prob_):
            # unseen categories carry no evidence rather than crashing
            idx = np.clip(Xi[:, j], 0, log_prob.shape[1] - 1)
            jll += log_prob[:, idx].T
        return jll


__all__ = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB",
           "CategoricalNB"]
