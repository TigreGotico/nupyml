"""A sparse Bayesian kernel regressor with a predictive distribution"""
import numpy as np
from scipy.spatial.distance import cdist
from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array, check_random_state


class RelevanceVectorMachine(BaseEstimator, RegressorMixin):
    """A sparse Bayesian kernel regressor with a predictive distribution
    (Tipping, 2001).

    The SVM gives a sparse predictor but no probabilities, and its sparsity is a
    side effect of the hinge loss. The RVM starts from a Bayesian model with an
    INDIVIDUAL precision (``alpha_i``) on every basis weight and maximises the
    evidence for those precisions. Most ``alpha_i`` diverge to infinity, forcing
    their weight to exactly zero -- so the model keeps only a handful of RELEVANCE
    VECTORS, usually far fewer than the SVM's support vectors, and yields a full
    predictive mean AND variance. Kernel (RBF) basis; ``predict(..., return_std=True)``
    gives the error bars.
    """

    def __init__(self, gamma=1.0, max_iter=500, tol=1e-5, threshold=1e5):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold

    def _design(self, X, Xr):
        K = np.exp(-self.gamma * cdist(X, Xr, "sqeuclidean"))
        return np.column_stack([np.ones(len(X)), K])       # + bias column

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float).ravel()
        self.X_train_ = X
        Phi = self._design(X, X)
        n, m = Phi.shape
        alpha = np.ones(m)
        beta = 1.0 / (np.var(y) + 1e-6)
        active = np.ones(m, bool)
        for _ in range(self.max_iter):
            Pa = Phi[:, active]
            A = np.diag(alpha[active])
            Sigma = np.linalg.inv(A + beta * Pa.T @ Pa)
            mu = beta * Sigma @ Pa.T @ y
            gamma_i = 1.0 - alpha[active] * np.diag(Sigma)
            old = alpha[active].copy()
            new_alpha = gamma_i / (mu ** 2 + 1e-12)
            alpha[active] = new_alpha
            rss = np.sum((y - Pa @ mu) ** 2)
            beta = (n - gamma_i.sum()) / (rss + 1e-12)
            active = alpha < self.threshold                # prune diverged weights
            if not active.any():
                active[0] = True
            if np.max(np.abs(np.log(new_alpha + 1e-12)
                             - np.log(old + 1e-12))) < self.tol:
                break
        self.active_ = active
        self.alpha_, self.beta_ = alpha, beta
        Pa = Phi[:, active]
        self.Sigma_ = np.linalg.inv(np.diag(alpha[active]) + beta * Pa.T @ Pa)
        self.mu_ = beta * self.Sigma_ @ Pa.T @ y
        # relevance vectors = retained training points (drop the bias column)
        rv = np.where(active[1:])[0]
        self.relevance_vectors_ = X[rv]
        self.n_relevance_ = len(rv)
        return self

    def predict(self, X, return_std=False):
        Phi = self._design(check_array(X), self.X_train_)[:, self.active_]
        mean = Phi @ self.mu_
        if not return_std:
            return mean
        var = 1.0 / self.beta_ + np.einsum("ij,jk,ik->i", Phi, self.Sigma_, Phi)
        return mean, np.sqrt(var)


__all__ = ["RelevanceVectorMachine"]
