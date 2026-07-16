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

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        d = X.shape[1]
        self.theta_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        self.class_prior_ = np.zeros(k)
        for c in range(k):
            Xc = X[y_idx == c]
            self.theta_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0)
            self.class_prior_[c] = len(Xc) / len(X)
        self.var_ += self.var_smoothing * X.var(axis=0).max()
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

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        self.class_count_ = Y.sum(axis=0)
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        self._fit_counts(X, Y)
        return self


class MultinomialNB(_DiscreteNB):
    def _fit_counts(self, X, Y):
        fc = np.asarray(Y.T @ X if not sp.issparse(X) else (sp.csr_matrix(Y.T) @ X).todense())
        fc = np.asarray(fc) + self.alpha
        self.feature_log_prob_ = np.log(fc) - np.log(fc.sum(axis=1, keepdims=True))

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "feature_log_prob_")
        X = check_array(X, accept_sparse=True)
        jll = X @ self.feature_log_prob_.T
        return np.asarray(jll) + self.class_log_prior_


class ComplementNB(_DiscreteNB):
    def _fit_counts(self, X, Y):
        Xd = np.asarray(X.todense()) if sp.issparse(X) else X
        fc = Y.T @ Xd
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
        fc = Y.T @ Xd + self.alpha
        cc = Y.sum(axis=0)[:, None] + 2 * self.alpha
        self.feature_log_prob_ = np.log(fc / cc)
        self._neg_log_prob = np.log(1 - fc / cc)

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "feature_log_prob_")
        X = check_array(X, accept_sparse=True)
        Xd = np.asarray(X.todense()) if sp.issparse(X) else X
        if self.binarize is not None:
            Xd = (Xd > self.binarize).astype(np.float64)
        jll = Xd @ self.feature_log_prob_.T + (1 - Xd) @ self._neg_log_prob.T
        return jll + self.class_log_prior_


__all__ = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB"]
