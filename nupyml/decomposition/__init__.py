"""Matrix decomposition: PCA, TruncatedSVD, NMF, FastICA, KernelPCA."""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        X = check_array(X)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components or min(n, d)
        if isinstance(k, float):
            ratios = S ** 2 / (S ** 2).sum()
            k = int(np.searchsorted(np.cumsum(ratios), k) + 1)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = (S[:k] ** 2) / (n - 1)
        self.explained_variance_ratio_ = (S[:k] ** 2) / (S ** 2).sum()
        self.n_components_ = k
        self.n_features_in_ = d
        self.n_features_out_ = k
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        Xt = (X - self.mean_) @ self.components_.T
        if self.whiten:
            Xt /= np.sqrt(self.explained_variance_)
        return Xt

    def inverse_transform(self, Xt):
        check_is_fitted(self, "components_")
        Xt = np.asarray(Xt, dtype=np.float64)
        if self.whiten:
            Xt = Xt * np.sqrt(self.explained_variance_)
        return Xt @ self.components_ + self.mean_


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        if sp.issparse(X):
            U, S, Vt = sp.linalg.svds(X, k=self.n_components)
            order = np.argsort(-S)
            U, S, Vt = U[:, order], S[order], Vt[order]
        else:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            U, S, Vt = U[:, :self.n_components], S[:self.n_components], \
                Vt[:self.n_components]
        self.components_ = Vt
        self.singular_values_ = S
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(Vt)
        return U * S

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, accept_sparse=True)
        return np.asarray(X @ self.components_.T)


class NMF(BaseEstimator, TransformerMixin):
    """Non-negative matrix factorization via multiplicative updates."""

    _estimator_tags = {"requires_positive_X": True}

    def __init__(self, n_components=2, max_iter=500, tol=1e-5, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        if (X < 0).any():
            raise ValueError("NMF input must be non-negative")
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components
        scale = np.sqrt(X.mean() / k)
        W = np.abs(rng.normal(scale=scale, size=(n, k)))
        H = np.abs(rng.normal(scale=scale, size=(k, d)))
        eps = 1e-12
        prev_err = np.inf
        for it in range(self.max_iter):
            H *= (W.T @ X) / (W.T @ W @ H + eps)
            W *= (X @ H.T) / (W @ H @ H.T + eps)
            if it % 10 == 0:
                err = np.linalg.norm(X - W @ H)
                if abs(prev_err - err) < self.tol * max(prev_err, 1.0):
                    break
                prev_err = err
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H))
        self.n_iter_ = it + 1
        self.n_features_in_ = d
        self.n_features_out_ = k
        return W

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        H = self.components_
        rng = check_random_state(self.random_state)
        W = np.abs(rng.normal(scale=np.sqrt(X.mean() / len(H)),
                              size=(len(X), len(H))))
        eps = 1e-12
        for _ in range(200):
            W *= (X @ H.T) / (W @ H @ H.T + eps)
        return W

    def inverse_transform(self, W):
        check_is_fitted(self, "components_")
        return np.asarray(W) @ self.components_


class FastICA(BaseEstimator, TransformerMixin):
    """FastICA with the logcosh contrast and symmetric decorrelation."""

    def __init__(self, n_components=None, max_iter=200, tol=1e-5,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components or d
        self.mean_ = X.mean(axis=0)
        Xc = (X - self.mean_).T                       # (d, n)
        # whiten
        cov = Xc @ Xc.T / n
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(-eigval)[:k]
        D = eigval[order]
        E = eigvec[:, order]
        K = (E / np.sqrt(D)).T                        # (k, d)
        Xw = K @ Xc                                   # (k, n)
        W = rng.normal(size=(k, k))

        def sym_decorrelate(W):
            s, u = np.linalg.eigh(W @ W.T)
            return (u / np.sqrt(s)) @ u.T @ W

        W = sym_decorrelate(W)
        for _ in range(self.max_iter):
            WX = W @ Xw
            g = np.tanh(WX)
            g_prime = 1 - g ** 2
            W_new = g @ Xw.T / n - g_prime.mean(axis=1)[:, None] * W
            W_new = sym_decorrelate(W_new)
            lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if lim < self.tol:
                break
        self.whitening_ = K
        self.unmixing_ = W
        self.components_ = W @ K
        self.mixing_ = np.linalg.pinv(self.components_)
        self.n_features_in_ = d
        self.n_features_out_ = k
        return (self.components_ @ Xc).T

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        return (X - self.mean_) @ self.components_.T


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, kernel="rbf", gamma=None, degree=3):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def _kernel(self, X, Y):
        gamma = self.gamma or 1.0 / X.shape[1]
        if self.kernel == "rbf":
            return np.exp(-gamma * cdist(X, Y) ** 2)
        if self.kernel == "poly":
            return (gamma * X @ Y.T + 1.0) ** self.degree
        if self.kernel == "linear":
            return X @ Y.T
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def fit(self, X, y=None):
        X = check_array(X)
        self._X_fit = X
        n = len(X)
        K = self._kernel(X, X)
        one_n = np.full((n, n), 1.0 / n)
        Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        vals, vecs = scipy.linalg.eigh(Kc)
        order = np.argsort(-vals)[:self.n_components]
        self.eigenvalues_ = np.maximum(vals[order], 0)
        self.eigenvectors_ = vecs[:, order]
        self._K_fit_rows = K.mean(axis=0)
        self._K_fit_all = K.mean()
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_components
        return self

    def transform(self, X):
        check_is_fitted(self, "eigenvectors_")
        X = check_array(X)
        K = self._kernel(X, self._X_fit)
        Kc = (K - K.mean(axis=1, keepdims=True) - self._K_fit_rows
              + self._K_fit_all)
        nonzero = self.eigenvalues_ > 1e-12
        alphas = np.zeros_like(self.eigenvectors_)
        alphas[:, nonzero] = (self.eigenvectors_[:, nonzero]
                              / np.sqrt(self.eigenvalues_[nonzero]))
        return Kc @ alphas


from ._sparse import (  # noqa: E402
    SparsePCA, DictionaryLearning, SparseCoder, MiniBatchNMF,
    LatentDirichletAllocation,
)

__all__ = ["PCA", "TruncatedSVD", "NMF", "FastICA", "KernelPCA",
           "SparsePCA", "DictionaryLearning", "SparseCoder", "MiniBatchNMF",
           "LatentDirichletAllocation"]
