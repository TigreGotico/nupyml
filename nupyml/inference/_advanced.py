"""Probabilistic inference v2: adaptive HMC, particle VI, moment-matching, and
slice sampling.

Four inference methods beyond the Metropolis/Gibbs/HMC/VI core. NUTS removes HMC's
trajectory-length knob. SVGD transports a set of PARTICLES to the posterior with a
deterministic, gradient-based flow -- variational inference without a parametric
family. Expectation propagation approximates a posterior by MOMENT MATCHING one
factor at a time. Slice sampling is a tuning-free MCMC that adapts its step to the
local density.
"""
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm

from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_array, check_random_state


class NUTS(BaseEstimator):
    """HMC that picks its own trajectory length (Hoffman & Gelman, 2014).

    HMC's one awkward knob is how many leapfrog steps to take: too few and it
    barely moves, too many and it wastes work looping back on itself. The No-U-Turn
    Sampler removes it by doubling the trajectory -- forward and backward in time --
    until the endpoints start to approach each other (a "U-turn"), then samples a
    point from the whole balanced trajectory. No trajectory-length tuning, and it
    is the sampler behind Stan and PyMC. This is the recursive tree-building NUTS
    with a fixed step size.
    """

    def __init__(self, log_prob, grad_log_prob, step_size=0.1, max_tree_depth=10,
                 random_state=None):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.random_state = random_state

    def _leapfrog(self, theta, r, eps):
        r = r + 0.5 * eps * self.grad_log_prob(theta)
        theta = theta + eps * r
        r = r + 0.5 * eps * self.grad_log_prob(theta)
        return theta, r

    def _H(self, theta, r):
        return self.log_prob(theta) - 0.5 * r @ r

    def sample(self, x0, n_samples, burn_in=500):
        rng = check_random_state(self.random_state)
        theta = np.atleast_1d(np.asarray(x0, float))
        out = np.zeros((n_samples, len(theta)))
        for it in range(n_samples + burn_in):
            r0 = rng.normal(size=len(theta))
            logu = self._H(theta, r0) - rng.exponential()   # slice variable (log)
            tm = tp = theta; rm = rp = r0
            tprime = theta; n = 1; s = 1; j = 0
            while s == 1 and j < self.max_tree_depth:
                v = 1 if rng.rand() < 0.5 else -1
                if v == -1:
                    tm, rm, _, _, tcand, ncand, scand = self._build(tm, rm, logu, v, j, rng)
                else:
                    _, _, tp, rp, tcand, ncand, scand = self._build(tp, rp, logu, v, j, rng)
                if scand == 1 and rng.rand() < ncand / max(n, 1):
                    tprime = tcand
                n += ncand
                s = scand * self._no_uturn(tm, tp, rm, rp)
                j += 1
            theta = tprime
            if it >= burn_in:
                out[it - burn_in] = theta
        return out

    def _no_uturn(self, tm, tp, rm, rp):
        d = tp - tm
        return int((d @ rm >= 0) and (d @ rp >= 0))

    def _build(self, theta, r, logu, v, j, rng):
        if j == 0:
            tp, rp = self._leapfrog(theta, r, v * self.step_size)
            n = int(logu <= self._H(tp, rp))
            s = int(logu < self._H(tp, rp) + 1000)      # divergence guard
            return tp, rp, tp, rp, tp, n, s
        tm, rm, tpl, rpl, tcand, n, s = self._build(theta, r, logu, v, j - 1, rng)
        if s == 1:
            if v == -1:
                tm, rm, _, _, t2, n2, s2 = self._build(tm, rm, logu, v, j - 1, rng)
            else:
                _, _, tpl, rpl, t2, n2, s2 = self._build(tpl, rpl, logu, v, j - 1, rng)
            if n2 > 0 and rng.rand() < n2 / max(n + n2, 1):
                tcand = t2
            s = s2 * self._no_uturn(tm, tpl, rm, rpl)
            n = n + n2
        return tm, rm, tpl, rpl, tcand, n, s


class SVGD(BaseEstimator):
    """Move a set of PARTICLES to the posterior by a deterministic flow
    (Liu & Wang, 2016).

    Variational inference fits a parametric family; SVGD keeps a set of particles
    and pushes them, all at once, along the direction that most decreases the KL to
    the target -- the "Stein" direction. Each particle is drawn toward high density
    by the average gradient of its neighbours (weighted by an RBF kernel) while a
    repulsive kernel term spreads the particles out so they don't collapse to the
    mode. The result interpolates between a single MAP point (one particle) and full
    Bayesian sampling (many), with no accept/reject. ``bandwidth=None`` uses the
    median heuristic.
    """

    def __init__(self, grad_log_prob, n_particles=50, step_size=0.1, n_iter=500,
                 bandwidth=None, random_state=None):
        self.grad_log_prob = grad_log_prob
        self.n_particles = n_particles
        self.step_size = step_size
        self.n_iter = n_iter
        self.bandwidth = bandwidth
        self.random_state = random_state

    def _kernel(self, X):
        sq = cdist(X, X, "sqeuclidean")
        if self.bandwidth is None:
            med = np.median(pdist(X) ** 2) if len(X) > 1 else 1.0
            h = med / (np.log(len(X) + 1) + 1e-9) + 1e-9   # median heuristic
        else:
            h = self.bandwidth
        K = np.exp(-sq / h)
        # grad_{x_j} k(x_j, x_i) = K_ij * 2/h * (x_i - x_j)
        return K, h

    def sample(self, x0):
        rng = check_random_state(self.random_state)
        X = np.array(x0, float)
        if X.ndim == 1:
            X = X[:, None]
        n = len(X)
        for _ in range(self.n_iter):
            g = np.array([self.grad_log_prob(x) for x in X])   # (n, d)
            K, h = self._kernel(X)
            # phi_i = 1/n sum_j [ K_ji grad_j + grad_{x_j} K_ji ]
            attractive = K @ g
            repulsive = (K.sum(axis=1)[:, None] * X - K @ X) * (2.0 / h)
            phi = (attractive + repulsive) / n
            X = X + self.step_size * phi
        self.particles_ = X
        return X


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


class SliceSampler(BaseEstimator):
    """MCMC with no step-size to tune -- it finds its own (Neal, 2003).

    Metropolis needs a proposal width: too small mixes slowly, too large rejects
    everything, and the right value differs across the distribution. Slice sampling
    removes the choice. To sample ``x``, draw a height ``u`` uniformly under the
    density at the current point, then sample the next ``x`` uniformly from the
    "slice" -- the set of points whose density exceeds ``u`` -- found by STEPPING
    OUT an interval and SHRINKING it on rejection. Every proposal is accepted, and
    the interval adapts to the local scale automatically. Coordinate-wise for
    multiple dimensions.
    """

    def __init__(self, log_prob, width=1.0, max_stepout=50, random_state=None):
        self.log_prob = log_prob
        self.width = width
        self.max_stepout = max_stepout
        self.random_state = random_state

    def sample(self, x0, n_samples, burn_in=200):
        rng = check_random_state(self.random_state)
        x = np.atleast_1d(np.asarray(x0, float))
        d = len(x)
        out = np.zeros((n_samples, d))
        for it in range(n_samples + burn_in):
            for k in range(d):
                x = self._sample_1d(x, k, rng)
            if it >= burn_in:
                out[it - burn_in] = x
        return out

    def _sample_1d(self, x, k, rng):
        logy = self.log_prob(x) + np.log(rng.rand())     # height under the density
        # step out an interval [L, R] around the current coordinate
        left = x.copy(); right = x.copy()
        r = rng.rand()
        left[k] = x[k] - r * self.width
        right[k] = x[k] + (1 - r) * self.width
        j = 0
        while self.log_prob(left) > logy and j < self.max_stepout:
            left[k] -= self.width; j += 1
        j = 0
        while self.log_prob(right) > logy and j < self.max_stepout:
            right[k] += self.width; j += 1
        # shrink until a point inside the slice is found
        for _ in range(100):
            xk = rng.uniform(left[k], right[k])
            cand = x.copy(); cand[k] = xk
            if self.log_prob(cand) > logy:
                return cand
            if xk < x[k]:
                left[k] = xk
            else:
                right[k] = xk
        return x


__all__ = ["NUTS", "SVGD", "ExpectationPropagationClassifier", "SliceSampler"]
