"""Gaussian mixture models fit with log-space EM."""
import numpy as np
from scipy.special import logsumexp

from ..base import BaseEstimator, ClusterMixin, DensityMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _log_gaussian(X, mean, cov, cov_type, reg):
    n, d = X.shape
    if cov_type == "full":
        cov = cov + reg * np.eye(d)
        L = np.linalg.cholesky(cov)
        diff = X - mean
        sol = np.linalg.solve(L, diff.T)
        maha = (sol ** 2).sum(axis=0)
        logdet = 2 * np.log(np.diag(L)).sum()
    elif cov_type == "diag":
        var = cov + reg
        diff = X - mean
        maha = ((diff ** 2) / var).sum(axis=1)
        logdet = np.log(var).sum()
    elif cov_type == "spherical":
        var = cov + reg
        maha = ((X - mean) ** 2).sum(axis=1) / var
        logdet = d * np.log(var)
    else:
        raise ValueError(f"Unknown covariance_type: {cov_type!r}")
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


class GaussianMixture(BaseEstimator, DensityMixin, ClusterMixin):
    def __init__(self, n_components=1, covariance_type="full", max_iter=100,
                 tol=1e-3, n_init=1, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _init_params(self, X, rng):
        from ..cluster import KMeans
        km = KMeans(n_clusters=self.n_components, n_init=1,
                    random_state=rng.randint(0, 2 ** 31 - 1)).fit(X)
        means = km.cluster_centers_
        k, d = self.n_components, X.shape[1]
        weights = np.bincount(km.labels_, minlength=k).astype(float) / len(X)
        weights = np.maximum(weights, 1e-6)
        weights /= weights.sum()
        if self.covariance_type == "full":
            covs = np.array([np.cov(X.T) + self.reg_covar * np.eye(d)] * k)
        elif self.covariance_type == "diag":
            covs = np.tile(X.var(axis=0) + self.reg_covar, (k, 1))
        else:
            covs = np.full(k, X.var() + self.reg_covar)
        return weights, means, covs

    def _e_step(self, X, weights, means, covs):
        k = self.n_components
        log_prob = np.empty((len(X), k))
        for c in range(k):
            log_prob[:, c] = _log_gaussian(X, means[c], covs[c],
                                           self.covariance_type, self.reg_covar)
        weighted = log_prob + np.log(weights)
        log_norm = logsumexp(weighted, axis=1, keepdims=True)
        return weighted - log_norm, float(log_norm.mean())

    def _m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 1e-10
        weights = nk / len(X)
        means = (resp.T @ X) / nk[:, None]
        k, d = self.n_components, X.shape[1]
        if self.covariance_type == "full":
            covs = np.empty((k, d, d))
            for c in range(k):
                diff = X - means[c]
                covs[c] = (resp[:, c][:, None] * diff).T @ diff / nk[c]
        elif self.covariance_type == "diag":
            covs = ((resp.T @ (X ** 2)) / nk[:, None]) - means ** 2
            covs = np.maximum(covs, 0)
        else:
            full_diag = ((resp.T @ (X ** 2)) / nk[:, None]) - means ** 2
            covs = np.maximum(full_diag.mean(axis=1), 0)
        return weights, means, covs

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        best = None
        for _ in range(self.n_init):
            weights, means, covs = self._init_params(X, rng)
            prev_ll = -np.inf
            for it in range(self.max_iter):
                log_resp, ll = self._e_step(X, weights, means, covs)
                weights, means, covs = self._m_step(X, log_resp)
                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll
            if best is None or ll > best[0]:
                best = (ll, weights, means, covs, it + 1)
        self.lower_bound_, self.weights_, self.means_, self.covariances_, \
            self.n_iter_ = best
        _, log_resp = self.lower_bound_, None
        lr, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        self.labels_ = lr.argmax(axis=1)
        return self

    def _estimate_log_resp(self, X):
        check_is_fitted(self, "means_")
        X = check_array(X)
        log_resp, _ = self._e_step(X, self.weights_, self.means_,
                                   self.covariances_)
        return log_resp

    def predict_proba(self, X):
        return np.exp(self._estimate_log_resp(X))

    def predict(self, X):
        return self._estimate_log_resp(X).argmax(axis=1)

    def score_samples(self, X):
        check_is_fitted(self, "means_")
        X = check_array(X)
        k = self.n_components
        log_prob = np.empty((len(X), k))
        for c in range(k):
            log_prob[:, c] = _log_gaussian(X, self.means_[c],
                                           self.covariances_[c],
                                           self.covariance_type, self.reg_covar)
        return logsumexp(log_prob + np.log(self.weights_), axis=1)

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())

    def sample(self, n_samples=1, random_state=None):
        check_is_fitted(self, "means_")
        rng = check_random_state(random_state)
        counts = rng.multinomial(n_samples, self.weights_)
        d = self.means_.shape[1]
        out = []
        comps = []
        for c, n_c in enumerate(counts):
            if n_c == 0:
                continue
            if self.covariance_type == "full":
                cov = self.covariances_[c]
            elif self.covariance_type == "diag":
                cov = np.diag(self.covariances_[c])
            else:
                cov = self.covariances_[c] * np.eye(d)
            out.append(rng.multivariate_normal(self.means_[c], cov, size=n_c))
            comps.append(np.full(n_c, c))
        return np.vstack(out), np.concatenate(comps)

    def bic(self, X):
        X = check_array(X)
        k, d = self.n_components, X.shape[1]
        if self.covariance_type == "full":
            cov_params = k * d * (d + 1) / 2
        elif self.covariance_type == "diag":
            cov_params = k * d
        else:
            cov_params = k
        n_params = k - 1 + k * d + cov_params
        return float(-2 * self.score(X) * len(X) + n_params * np.log(len(X)))

    def aic(self, X):
        X = check_array(X)
        k, d = self.n_components, X.shape[1]
        if self.covariance_type == "full":
            cov_params = k * d * (d + 1) / 2
        elif self.covariance_type == "diag":
            cov_params = k * d
        else:
            cov_params = k
        n_params = k - 1 + k * d + cov_params
        return float(-2 * self.score(X) * len(X) + 2 * n_params)


__all__ = ["GaussianMixture"]
