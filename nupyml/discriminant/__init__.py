"""Discriminant analysis: model each class as a Gaussian, then apply Bayes.

Like ``GaussianNB``, these are generative: model ``P(x | class)`` per class and
turn it around with Bayes' rule. Unlike naive Bayes, they estimate the FULL
covariance, so they see correlations between features rather than assuming they
do not exist.

The only difference between the two is one question: do the classes share a
covariance?

* ``LinearDiscriminantAnalysis`` says yes. Every class has its own mean but
  one pooled covariance. In the Bayes rule, the quadratic terms then CANCEL
  between classes and the boundary comes out LINEAR -- that is where the name
  comes from, and it is a consequence, not a design choice.
* ``QuadraticDiscriminantAnalysis`` says no. Each class keeps its own
  covariance, nothing cancels, and the boundaries are quadratic -- ellipses and
  parabolas. More flexible, but it estimates a full covariance per class, which
  needs enough samples in every class to be non-singular.

That is a bias-variance choice in the open: LDA is more constrained and more
stable on little data; QDA fits more shapes and needs more data to do it.

LDA also doubles as a supervised dimensionality reduction. Where PCA finds
directions of maximum variance -- ignoring labels entirely -- LDA finds
directions maximising BETWEEN-class scatter relative to WITHIN-class scatter:
directions that pull classes apart. It yields at most ``n_classes - 1`` of them,
since that is the dimension the class means span.
"""
import numpy as np
from scipy.special import logsumexp

from ..base import BaseEstimator, ClassifierMixin, TransformerMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin,
                                 TransformerMixin):
    def __init__(self, n_components=None, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n, d = X.shape
        self.priors_ = np.bincount(y_idx, minlength=k) / n
        self.means_ = np.array([X[y_idx == c].mean(axis=0) for c in range(k)])
        # pooled within-class covariance
        Sw = np.zeros((d, d))
        for c in range(k):
            diff = X[y_idx == c] - self.means_[c]
            Sw += diff.T @ diff
        Sw /= n - k
        Sw += self.reg * np.eye(d)
        self.covariance_ = Sw
        Sw_inv = np.linalg.inv(Sw)
        self.coef_ = self.means_ @ Sw_inv
        self.intercept_ = (-0.5 * np.einsum("cd,dc->c", self.coef_, self.means_.T)
                           + np.log(self.priors_))
        # transform directions: eigenvectors of Sw^-1 Sb
        overall = X.mean(axis=0)
        Sb = np.zeros((d, d))
        for c in range(k):
            diff = (self.means_[c] - overall)[:, None]
            Sb += self.priors_[c] * (diff @ diff.T)
        vals, vecs = np.linalg.eig(Sw_inv @ Sb)
        order = np.argsort(-vals.real)
        n_comp = self.n_components or min(k - 1, d)
        self.scalings_ = vecs[:, order[:n_comp]].real
        self._overall_mean = overall
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        scores = check_array(X) @ self.coef_.T + self.intercept_
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def transform(self, X):
        check_is_fitted(self, "scalings_")
        return (check_array(X) - self._overall_mean) @ self.scalings_


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, reg_param=1e-6):
        self.reg_param = reg_param

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n, d = X.shape
        self.priors_ = np.bincount(y_idx, minlength=k) / n
        self.means_ = np.array([X[y_idx == c].mean(axis=0) for c in range(k)])
        self.covariances_ = []
        for c in range(k):
            Xc = X[y_idx == c]
            cov = np.cov(Xc.T, bias=False) if len(Xc) > 1 else np.eye(d)
            cov = np.atleast_2d(cov) + self.reg_param * np.eye(d)
            self.covariances_.append(cov)
        return self

    def _log_posterior(self, X):
        check_is_fitted(self, "means_")
        X = check_array(X)
        k = len(self.classes_)
        ll = np.empty((len(X), k))
        for c in range(k):
            cov = self.covariances_[c]
            L = np.linalg.cholesky(cov)
            diff = X - self.means_[c]
            sol = np.linalg.solve(L, diff.T)
            maha = (sol ** 2).sum(axis=0)
            logdet = 2 * np.log(np.diag(L)).sum()
            ll[:, c] = -0.5 * (maha + logdet) + np.log(self.priors_[c])
        return ll

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X):
        lp = self._log_posterior(X)
        return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]
