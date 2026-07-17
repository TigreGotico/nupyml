"""Gaussian mixture models: clustering as density estimation.

THE MODEL
---------
Assume the data was GENERATED like this: pick a component k with probability
``weight_k``, then draw a point from that component's Gaussian. Fitting means
recovering the weights, means and covariances from the points alone -- the
component labels were never observed.

This is what "soft clustering" means: a point does not belong to a cluster, it
has a POSTERIOR PROBABILITY of having come from each. A point midway between
two components honestly reports 50/50 instead of being forced to pick.

WHY THIS BEATS KMEANS
---------------------
KMeans is (almost) the special case where every covariance is a shared multiple
of the identity and assignments are hard. Letting each component keep its own
full covariance lets it be stretched, rotated and sized differently -- so a GMM
fits elongated and overlapping clusters that KMeans has no vocabulary for.

Being a real probability model, it also gives a likelihood, and therefore
``bic``/``aic`` for choosing the number of components, and ``sample`` for
generating new data. KMeans offers none of that.

WHY EM, AND WHY IT WORKS
------------------------
The likelihood contains a log of a sum -- ``log(sum_k w_k * N(x | mu_k))`` --
which does not separate, so there is nothing to solve directly. The trouble is
circular: knowing which component made each point would make the parameters
trivial, and knowing the parameters would make the assignment trivial.

EM cuts the circle by alternating:

* **E-step**: given the parameters, compute each point's posterior over
  components (the "responsibilities").
* **M-step**: given the responsibilities, update each component as a weighted
  fit -- every point contributes to every component, in proportion.

The guarantee is that each round cannot DECREASE the likelihood. EM is really
maximising a lower bound on the likelihood: the E-step makes the bound touch
the true likelihood at the current parameters, and the M-step maximises the
bound. Since the bound touches, improving it improves the real thing. That is
why EM converges monotonically -- to a local optimum, so ``n_init`` matters
here as much as in KMeans.

NUMERICAL REALITY
-----------------
Everything is computed in LOG space. A Gaussian density in 20 dimensions
underflows to exactly 0.0 for any point that is not very close to the mean, and
then a ratio of densities is 0/0. Working with log-densities and combining them
with ``logsumexp`` -- which factors out the largest term before exponentiating
-- keeps every quantity in a representable range. ``reg_covar`` guards the other
failure: a component collapsing onto a single point makes its covariance
singular and its likelihood infinite, which EM is otherwise happy to pursue.
"""
import numpy as np
from scipy.special import logsumexp

from ..base import BaseEstimator, ClusterMixin, DensityMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _log_gaussian(X, mean, cov, cov_type, reg):
    """Log-density of a multivariate Gaussian, via Cholesky.

    The density needs ``(x-mu)' inv(C) (x-mu)`` and ``log|C|``. Neither is
    computed the obvious way, because inverting a covariance and taking a
    determinant are both numerically poor and needlessly expensive.

    The Cholesky factor ``C = L L'`` gives both cheaply and stably:

    * the quadratic form becomes ``||solve(L, x-mu)||^2`` -- a triangular solve,
      no inverse ever formed;
    * ``log|C| = 2*sum(log(diag(L)))`` -- a sum of logs rather than a product of
      n numbers that would overflow.

    Cholesky also fails loudly (rather than returning garbage) if the covariance
    is not positive definite, which is a useful alarm.

    ``covariance_type`` trades flexibility for parameters: ``full`` is d(d+1)/2
    numbers per component and fits any ellipse; ``diag`` is d, axis-aligned
    only; ``spherical`` is 1, round only. With limited data, fewer parameters
    often generalise better.
    """
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
        """Responsibilities: the posterior probability of each component per point.

        Bayes' rule, in log space::

            log r_ik = log w_k + log N(x_i | mu_k, C_k) - log(sum over k)

        The subtracted term is the normaliser, and ``logsumexp`` computes it
        without ever exponentiating a large negative number. It also returns the
        mean log-likelihood, which is what convergence is judged on.
        """
        k = self.n_components
        log_prob = np.empty((len(X), k))
        for c in range(k):
            log_prob[:, c] = _log_gaussian(X, means[c], covs[c],
                                           self.covariance_type, self.reg_covar)
        weighted = log_prob + np.log(weights)
        log_norm = logsumexp(weighted, axis=1, keepdims=True)
        return weighted - log_norm, float(log_norm.mean())

    def _m_step(self, X, log_resp):
        """Refit every component, with each point weighted by its responsibility.

        The update is exactly the maximum-likelihood fit of a Gaussian, except
        that the "count" of points in a component is now a fractional sum of
        responsibilities::

            n_k    = sum_i r_ik
            mu_k   = sum_i r_ik * x_i / n_k
            C_k    = sum_i r_ik * (x_i - mu_k)(x_i - mu_k)' / n_k
            w_k    = n_k / n

        Set every responsibility to 0 or 1 and these collapse into the KMeans
        M-step -- which is the precise sense in which KMeans is hard-assignment
        EM.
        """
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
        """Bayesian Information Criterion. LOWER is better.

        ``-2*log-likelihood + n_params*log(n)``. Likelihood alone always favours
        more components -- a component per point fits perfectly -- so it cannot
        choose k. BIC charges for parameters, and the ``log(n)`` charge grows
        with the sample size, so BIC gets stricter as evidence accumulates.

        Sweeping k and taking the minimum BIC is the standard way to choose the
        number of components. ``aic`` charges a flat 2 per parameter instead,
        so it is more permissive and tends to pick larger models.
        """
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


from ._bayesian import BayesianGaussianMixture  # noqa: E402

__all__ = ["GaussianMixture", "BayesianGaussianMixture"]
