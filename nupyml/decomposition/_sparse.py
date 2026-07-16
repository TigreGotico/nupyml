"""Sparse coding, dictionary learning, and topic modelling."""
import numpy as np
import scipy.special

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _lasso_cd(X, D, alpha, max_iter=100, tol=1e-4, init=None):
    """Solve min_C 0.5||X - C D||^2 + alpha||C||_1 over the code C."""
    n, k = len(X), len(D)
    C = np.zeros((n, k)) if init is None else init.copy()
    G = D @ D.T                       # (k, k) gram
    Dx = X @ D.T                      # (n, k)
    diag = np.maximum(np.diag(G), 1e-12)
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(k):
            old = C[:, j].copy()
            residual = Dx[:, j] - C @ G[:, j] + C[:, j] * G[j, j]
            C[:, j] = _soft_threshold(residual, alpha) / diag[j]
            max_delta = max(max_delta, np.abs(C[:, j] - old).max())
        if max_delta < tol:
            break
    return C


class SparsePCA(BaseEstimator, TransformerMixin):
    """PCA with an L1 penalty on the components (alternating minimization)."""

    def __init__(self, n_components=None, alpha=1.0, ridge_alpha=0.01,
                 max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        k = self.n_components or min(X.shape)
        # init from the dense PCA solution
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        code = U[:, :k] * S[:k]
        D = Vt[:k]
        prev = np.inf
        for it in range(self.max_iter):
            # dictionary step with an L1 penalty, then code step (ridge)
            D = _lasso_cd(Xc.T, code.T, self.alpha, max_iter=20).T
            norms = np.linalg.norm(D, axis=1, keepdims=True)
            D = D / np.where(norms > 0, norms, 1.0)
            gram = D @ D.T + self.ridge_alpha * np.eye(k)
            code = np.linalg.solve(gram, D @ Xc.T).T
            err = np.linalg.norm(Xc - code @ D)
            if abs(prev - err) < self.tol * max(prev, 1.0):
                break
            prev = err
        self.components_ = D
        self.error_ = float(err)
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X) - self.mean_
        D = self.components_
        gram = D @ D.T + self.ridge_alpha * np.eye(len(D))
        return np.linalg.solve(gram, D @ X.T).T


class DictionaryLearning(BaseEstimator, TransformerMixin):
    """Learn an overcomplete dictionary with sparse codes (alternating)."""

    def __init__(self, n_components=None, alpha=1.0, max_iter=50, tol=1e-6,
                 transform_alpha=None, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.transform_alpha = transform_alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components or d
        D = rng.normal(size=(k, d))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        code = np.zeros((n, k))
        prev = np.inf
        for it in range(self.max_iter):
            code = _lasso_cd(X, D, self.alpha, max_iter=30, init=code)
            # block-coordinate dictionary update (Mairal et al. 2009)
            A = code.T @ code
            B = code.T @ X
            for j in range(k):
                if A[j, j] < 1e-12:
                    D[j] = rng.normal(size=d)
                    D[j] /= np.linalg.norm(D[j])
                    continue
                u = (B[j] - A[j] @ D + A[j, j] * D[j]) / A[j, j]
                norm = np.linalg.norm(u)
                D[j] = u / max(norm, 1.0)
            err = np.linalg.norm(X - code @ D)
            if abs(prev - err) < self.tol * max(prev, 1.0):
                break
            prev = err
        self.components_ = D
        self.error_ = float(err)
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        alpha = self.transform_alpha if self.transform_alpha is not None \
            else self.alpha
        return _lasso_cd(X, self.components_, alpha, max_iter=100)


class SparseCoder(BaseEstimator, TransformerMixin):
    """Sparse codes against a fixed, user-supplied dictionary."""

    def __init__(self, dictionary, transform_alpha=1.0, max_iter=100):
        self.dictionary = dictionary
        self.transform_alpha = transform_alpha
        self.max_iter = max_iter

    def fit(self, X=None, y=None):
        self.components_ = np.asarray(self.dictionary, dtype=np.float64)
        return self

    def transform(self, X):
        D = np.asarray(self.dictionary, dtype=np.float64)
        return _lasso_cd(check_array(X), D, self.transform_alpha,
                         max_iter=self.max_iter)


class MiniBatchNMF(BaseEstimator, TransformerMixin):
    """NMF fit on mini-batches with running sufficient statistics."""

    def __init__(self, n_components=2, batch_size=100, max_iter=100,
                 forget_factor=0.7, random_state=None):
        self.n_components = n_components
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.forget_factor = forget_factor
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        if (X < 0).any():
            raise ValueError("MiniBatchNMF input must be non-negative")
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components
        scale = np.sqrt(X.mean() / k)
        H = np.abs(rng.normal(scale=scale, size=(k, d)))
        A = np.zeros((k, k))
        B = np.zeros((k, d))
        eps = 1e-12
        for it in range(self.max_iter):
            batch = X[rng.choice(n, size=min(self.batch_size, n), replace=False)]
            W = np.abs(rng.normal(scale=scale, size=(len(batch), k)))
            for _ in range(20):
                W *= (batch @ H.T) / (W @ H @ H.T + eps)
            rho = self.forget_factor
            A = rho * A + W.T @ W
            B = rho * B + W.T @ batch
            for _ in range(5):
                H *= B / (A @ H + eps)
                H = np.maximum(H, 0.0)
        self.components_ = H
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        rng = check_random_state(self.random_state)
        H = self.components_
        W = np.abs(rng.normal(scale=np.sqrt(max(X.mean(), 1e-12) / len(H)),
                              size=(len(X), len(H))))
        eps = 1e-12
        for _ in range(200):
            W *= (X @ H.T) / (W @ H @ H.T + eps)
        return W


class LatentDirichletAllocation(BaseEstimator, TransformerMixin):
    """Variational Bayes LDA (Hoffman et al. 2010), batch updates."""

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, max_iter=20, mean_change_tol=1e-3,
                 max_doc_update_iter=100, random_state=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_iter = max_iter
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.random_state = random_state

    @staticmethod
    def _dirichlet_expectation(alpha):
        """E[log theta] under a Dirichlet parameterized by alpha."""
        if alpha.ndim == 1:
            return scipy.special.psi(alpha) - scipy.special.psi(alpha.sum())
        return scipy.special.psi(alpha) - scipy.special.psi(
            alpha.sum(axis=1, keepdims=True))

    def _e_step(self, X, lam):
        n, d = X.shape
        k = self.n_components
        rng = check_random_state(self.random_state)
        Elogbeta = self._dirichlet_expectation(lam)
        expElogbeta = np.exp(Elogbeta)
        gamma = rng.gamma(100.0, 1.0 / 100.0, size=(n, k))
        sstats = np.zeros((k, d))
        for i in range(n):
            counts = X[i]
            ids = np.nonzero(counts)[0]
            if len(ids) == 0:
                continue
            cts = counts[ids]
            gammad = gamma[i]
            expElogthetad = np.exp(self._dirichlet_expectation(gammad))
            expElogbetad = expElogbeta[:, ids]
            phinorm = expElogthetad @ expElogbetad + 1e-100
            for _ in range(self.max_doc_update_iter):
                last = gammad
                gammad = (self.alpha_ + expElogthetad
                          * ((cts / phinorm) @ expElogbetad.T))
                expElogthetad = np.exp(self._dirichlet_expectation(gammad))
                phinorm = expElogthetad @ expElogbetad + 1e-100
                if np.mean(np.abs(gammad - last)) < self.mean_change_tol:
                    break
            gamma[i] = gammad
            sstats[:, ids] += np.outer(expElogthetad, cts / phinorm)
        sstats *= expElogbeta
        return gamma, sstats

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        import scipy.sparse as sp
        if sp.issparse(X):
            X = np.asarray(X.todense())
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components
        self.alpha_ = self.doc_topic_prior or 1.0 / k
        self.eta_ = self.topic_word_prior or 1.0 / k
        lam = rng.gamma(100.0, 1.0 / 100.0, size=(k, d))
        for it in range(self.max_iter):
            gamma, sstats = self._e_step(X, lam)
            lam = self.eta_ + sstats
        self.components_ = lam
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, accept_sparse=True)
        import scipy.sparse as sp
        if sp.issparse(X):
            X = np.asarray(X.todense())
        gamma, _ = self._e_step(X, self.components_)
        return gamma / gamma.sum(axis=1, keepdims=True)


__all__ = ["SparsePCA", "DictionaryLearning", "SparseCoder", "MiniBatchNMF",
           "LatentDirichletAllocation"]
