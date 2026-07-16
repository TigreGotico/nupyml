"""Variational Bayesian Gaussian mixture (Dirichlet / Dirichlet-process priors)."""
import numpy as np
import scipy.linalg
import scipy.special
from scipy.special import betaln, digamma, gammaln, logsumexp

from ..base import BaseEstimator, ClusterMixin, DensityMixin, check_is_fitted
from ..utils import check_array, check_random_state


class BayesianGaussianMixture(BaseEstimator, DensityMixin, ClusterMixin):
    """Full-covariance VB-GMM. Unneeded components have their weight driven
    toward zero, so ``n_components`` is an upper bound rather than a count."""

    def __init__(self, n_components=1, covariance_type="full", max_iter=100,
                 tol=1e-3, reg_covar=1e-6,
                 weight_concentration_prior_type="dirichlet_process",
                 weight_concentration_prior=None, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.random_state = random_state

    # ---------------- priors ----------------
    def _init_priors(self, X):
        n, d = X.shape
        k = self.n_components
        self._d = d
        self.weight_concentration_prior_ = (
            self.weight_concentration_prior
            if self.weight_concentration_prior is not None else 1.0 / k)
        self.mean_precision_prior_ = 1.0
        self.mean_prior_ = X.mean(axis=0)
        self.degrees_of_freedom_prior_ = float(d)
        self.covariance_prior_ = np.atleast_2d(np.cov(X.T)) + \
            self.reg_covar * np.eye(d)

    def _init_params(self, X, rng):
        from ..cluster import KMeans
        n, d = X.shape
        k = self.n_components
        km = KMeans(n_clusters=k, n_init=1,
                    random_state=rng.randint(0, 2 ** 31 - 1)).fit(X)
        resp = np.zeros((n, k))
        resp[np.arange(n), km.labels_] = 1.0
        resp += 0.01 * rng.uniform(size=(n, k))
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    # ---------------- VB updates ----------------
    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
        xk = (resp.T @ X) / nk[:, None]
        # weight concentration
        if self.weight_concentration_prior_type == "dirichlet_process":
            self.weight_concentration_ = (
                1.0 + nk,
                (self.weight_concentration_prior_
                 + np.hstack([np.cumsum(nk[::-1])[-2::-1], 0])))
        else:
            self.weight_concentration_ = self.weight_concentration_prior_ + nk
        # mean / precision
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = ((self.mean_precision_prior_ * self.mean_prior_
                        + nk[:, None] * xk) / self.mean_precision_[:, None])
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk
        self.covariances_ = np.empty((self.n_components, d, d))
        for c in range(self.n_components):
            diff = X - xk[c]
            sk = (resp[:, c][:, None] * diff).T @ diff / nk[c]
            dm = xk[c] - self.mean_prior_
            self.covariances_[c] = (
                self.covariance_prior_ + nk[c] * sk
                + (nk[c] * self.mean_precision_prior_ / self.mean_precision_[c])
                * np.outer(dm, dm))
            self.covariances_[c] /= self.degrees_of_freedom_[c]
            self.covariances_[c] += self.reg_covar * np.eye(d)
        self._nk = nk

    def _estimate_log_prob(self, X):
        """E_q[log N(x | mu_k, Lambda_k)] under the Normal-Wishart posterior.

        ``covariances_`` already carries the 1/nu factor, so the Mahalanobis
        term must not be scaled by nu again.
        """
        n, d = X.shape
        k = self.n_components
        log_prob = np.empty((n, k))
        for c in range(k):
            L = scipy.linalg.cholesky(self.covariances_[c], lower=True)
            diff = X - self.means_[c]
            sol = scipy.linalg.solve_triangular(L, diff.T, lower=True)
            maha = (sol ** 2).sum(axis=0)
            log_det = 2 * np.log(np.diag(L)).sum()
            nu = self.degrees_of_freedom_[c]
            log_gauss = -0.5 * (d * np.log(2 * np.pi) + maha + log_det
                                + d * np.log(nu))
            # E[log |Lambda_k|] under the Wishart posterior
            log_lambda = (d * np.log(2)
                          + digamma(0.5 * (nu - np.arange(d))).sum())
            log_prob[:, c] = log_gauss + 0.5 * (log_lambda
                                                - d / self.mean_precision_[c])
        return log_prob

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            a, b = self.weight_concentration_
            digamma_sum = digamma(a + b)
            log_v = digamma(a) - digamma_sum
            log_1mv = digamma(b) - digamma_sum
            return log_v + np.hstack([0, np.cumsum(log_1mv[:-1])])
        return (digamma(self.weight_concentration_)
                - digamma(self.weight_concentration_.sum()))

    def _e_step(self, X):
        weighted = self._estimate_log_prob(X) + self._estimate_log_weights()
        log_norm = logsumexp(weighted, axis=1, keepdims=True)
        log_resp = weighted - log_norm
        return np.exp(log_resp), log_resp, float(log_norm.mean())

    @staticmethod
    def _log_wishart_norm(nu, log_det_precisions_chol, d):
        return -(nu * log_det_precisions_chol
                 + nu * d * 0.5 * np.log(2.0)
                 + np.sum(gammaln(0.5 * (nu - np.arange(d)[:, None])), axis=0))

    def _lower_bound(self, log_resp):
        """Variational lower bound (ELBO), up to terms constant in the params.

        Tracking this rather than the mean log-likelihood matters: the
        likelihood plateaus while the weight concentrations are still
        sharpening, so a likelihood-based tol stops far too early.
        """
        d = self._d
        log_det_prec_chol = np.empty(self.n_components)
        for c in range(self.n_components):
            L = scipy.linalg.cholesky(self.covariances_[c], lower=True)
            # log|precision_chol| = -log|cov_chol|
            log_det_prec_chol[c] = -np.log(np.diag(L)).sum()
        log_det_prec_chol = log_det_prec_chol - 0.5 * d * np.log(
            self.degrees_of_freedom_)
        log_wishart = self._log_wishart_norm(
            self.degrees_of_freedom_, log_det_prec_chol, d).sum()
        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -np.sum(betaln(*self.weight_concentration_))
        else:
            log_norm_weight = float(
                gammaln(self.weight_concentration_.sum())
                - gammaln(self.weight_concentration_).sum())
        return float(-np.sum(np.exp(log_resp) * log_resp)
                     - log_wishart - log_norm_weight
                     - 0.5 * d * np.sum(np.log(self.mean_precision_)))

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self._init_priors(X)
        resp = self._init_params(X, rng)
        prev = -np.inf
        for it in range(self.max_iter):
            self._m_step(X, resp)
            resp, log_resp, _ = self._e_step(X)
            bound = self._lower_bound(log_resp)
            if abs(bound - prev) < self.tol:
                break
            prev = bound
        self.n_iter_ = it + 1
        self.lower_bound_ = bound
        # effective mixture weights
        if self.weight_concentration_prior_type == "dirichlet_process":
            a, b = self.weight_concentration_
            v = a / (a + b)
            self.weights_ = v * np.hstack([1, np.cumprod(1 - v)[:-1]])
            self.weights_ /= self.weights_.sum()
        else:
            self.weights_ = (self.weight_concentration_
                             / self.weight_concentration_.sum())
        self.labels_ = resp.argmax(axis=1)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "means_")
        resp, _, _ = self._e_step(check_array(X))
        return resp

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score_samples(self, X):
        check_is_fitted(self, "means_")
        X = check_array(X)
        weighted = self._estimate_log_prob(X) + self._estimate_log_weights()
        return logsumexp(weighted, axis=1)

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())


__all__ = ["BayesianGaussianMixture"]
