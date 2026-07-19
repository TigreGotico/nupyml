"""Approximate the posterior by matching MOMENTS, one factor at a time"""
import numpy as np
from scipy.stats import norm
from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_array, check_random_state


class ExpectationPropagationClassifier(BaseEstimator, ClassifierMixin):
    """Approximate the posterior by matching MOMENTS, one factor at a time
    (Minka, 2001).

    A probit classifier's posterior over weights is non-Gaussian (a Gaussian prior
    times many probit likelihood factors). EP approximates each awkward factor by a
    Gaussian "site", then repeatedly refines each site so that, combined with all
    the others, it matches the first two MOMENTS of the true tilted distribution.
    Unlike the Laplace approximation (which matches only curvature at the mode), EP
    matches means and variances globally, and is usually more accurate. The product
    of the sites gives a Gaussian posterior over the weights.
    """

    def __init__(self, prior_var=10.0, max_iter=50, tol=1e-4):
        self.prior_var = prior_var
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        X1 = np.column_stack([X, np.ones(len(X))])       # bias term
        n, d = X1.shape
        # site parameters (natural form): tau (precision), nu (precision*mean)
        tau = np.zeros(n); nu = np.zeros(n)
        Sigma = self.prior_var * np.eye(d)
        mu = np.zeros(d)
        for _ in range(self.max_iter):
            tau_old = tau.copy()
            for i in range(n):
                xi = X1[i]
                s_xi = Sigma @ xi
                var_i = xi @ s_xi
                # cavity distribution (remove site i)
                tau_c = 1.0 / var_i - tau[i]
                if tau_c <= 0:
                    continue
                nu_c = (xi @ mu) / var_i - nu[i]
                mu_c = nu_c / tau_c
                var_c = 1.0 / tau_c
                # tilted moment matching through the probit likelihood
                z = t[i] * mu_c / np.sqrt(1 + var_c)
                ratio = norm.pdf(z) / (norm.cdf(z) + 1e-12)
                mu_hat = mu_c + t[i] * var_c * ratio / np.sqrt(1 + var_c)
                var_hat = var_c - (var_c ** 2 * ratio *
                                   (z + ratio)) / (1 + var_c)
                # update site i
                new_tau = 1.0 / var_hat - tau_c
                new_nu = mu_hat / var_hat - nu_c
                dtau = new_tau - tau[i]
                tau[i], nu[i] = new_tau, new_nu
                # rank-1 update of the global posterior
                s_xi = Sigma @ xi
                Sigma = Sigma - (dtau / (1 + dtau * (xi @ s_xi))) * np.outer(s_xi, s_xi)
                mu = Sigma @ (X1.T @ nu)
            if np.abs(tau - tau_old).max() < self.tol:
                break
        self.mean_ = mu
        self.cov_ = Sigma
        return self

    def decision_function(self, X):
        X1 = np.column_stack([check_array(X), np.ones(len(X))])
        m = X1 @ self.mean_
        v = np.einsum("ij,jk,ik->i", X1, self.cov_, X1)
        return m / np.sqrt(1 + v)                         # probit-averaged score

    def predict_proba(self, X):
        p = norm.cdf(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


__all__ = ["ExpectationPropagationClassifier"]
