"""Probabilistic v5: evidence estimation, diverse sampling, probabilistic
integration, and scalable GP inference.

Four more probabilistic methods. Nested sampling turns a posterior into a
one-dimensional integral to compute the Bayesian EVIDENCE. A determinantal point
process samples DIVERSE subsets by repulsion. Bayesian quadrature treats numerical
INTEGRATION as GP regression. The sparse variational GP scales Gaussian-process
regression to many points via a few inducing points.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class NestedSampling(BaseEstimator):
    """Compute the Bayesian EVIDENCE by shrinking the prior (Skilling, 2004).

    The evidence ``Z = ∫ L(θ) π(θ) dθ`` -- the normalising constant that model
    comparison needs -- is a nightmare integral in high dimensions. Nested sampling
    reparametrises it as a ONE-dimensional integral over the prior mass enclosed above
    each likelihood level. It keeps ``n_live`` points sampled from the prior, repeatedly
    removes the LOWEST-likelihood one (banking its shrinking slice of prior mass as
    evidence) and replaces it with a fresh point of higher likelihood. The accumulated
    slices give ``Z``, and the discarded points, weighted, give posterior samples.
    ``log_likelihood(theta)`` and a box prior.
    """

    def __init__(self, log_likelihood, bounds, n_live=100, max_iter=2000,
                 random_state=None):
        self.log_likelihood = log_likelihood
        self.bounds = bounds
        self.n_live = n_live
        self.max_iter = max_iter
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.asarray(self.bounds[0], float), np.asarray(self.bounds[1], float)
        d = len(np.atleast_1d(lo))
        live = rng.uniform(lo, hi, (self.n_live, d))
        live_ll = np.array([self.log_likelihood(x) for x in live])
        logZ = -np.inf
        log_width = np.log(1.0 - np.exp(-1.0 / self.n_live))
        samples, weights = [], []
        for i in range(self.max_iter):
            worst = live_ll.argmin()
            Lworst = live_ll[worst]
            logwt = log_width - i / self.n_live + Lworst   # weight of this slice
            logZ = np.logaddexp(logZ, logwt)
            samples.append(live[worst].copy()); weights.append(logwt)
            # replace the worst with a higher-likelihood draw from the prior
            for _ in range(200):
                cand = rng.uniform(lo, hi, d)
                if self.log_likelihood(cand) > Lworst:
                    live[worst] = cand
                    live_ll[worst] = self.log_likelihood(cand)
                    break
            if logwt < logZ - 8:                           # remaining mass negligible
                break
        self.log_evidence_ = logZ
        self.samples_ = np.array(samples)
        w = np.exp(np.array(weights) - logZ)
        self.posterior_weights_ = w / w.sum()
        self.posterior_mean_ = (self.posterior_weights_[:, None]
                                * self.samples_).sum(axis=0)
        return self


class DeterminantalPointProcess(BaseEstimator):
    """Sample DIVERSE subsets by repulsion (Kulesza & Taskar, 2012).

    Random sampling clumps; if you want a subset whose items are DIFFERENT from each
    other -- diverse search results, a representative summary, a spread-out minibatch
    -- you need repulsion. A determinantal point process makes the probability of a
    subset proportional to the DETERMINANT of the kernel submatrix on it, and a
    determinant is large exactly when the selected vectors are nearly orthogonal
    (dissimilar) and small when they are similar. So similar items rarely appear
    together. Exact sampling via the kernel's eigendecomposition. ``L`` is a PSD
    similarity kernel.
    """

    def __init__(self, L, random_state=None):
        self.L = np.asarray(L, float)
        self.random_state = random_state

    def sample(self):
        rng = check_random_state(self.random_state)
        vals, vecs = np.linalg.eigh(self.L)
        vals = np.clip(vals, 0, None)
        # phase 1: pick an eigenvector set with prob lambda/(lambda+1)
        included = rng.rand(len(vals)) < vals / (vals + 1.0)
        V = vecs[:, included]
        selected = []
        # phase 2: iteratively project out a chosen item
        while V.shape[1] > 0:
            probs = (V ** 2).sum(axis=1)
            probs = probs / probs.sum()
            i = rng.choice(len(probs), p=probs)
            selected.append(i)
            # find a vector in V with nonzero component at i, orthogonalise the rest
            j = np.argmax(np.abs(V[i]))
            Vj = V[:, j]
            V = np.delete(V, j, axis=1)
            if V.shape[1] > 0:
                V = V - np.outer(Vj, V[i] / Vj[i])
                # re-orthonormalise
                q, _ = np.linalg.qr(V) if V.shape[1] else (V, None)
                V = q
        return np.array(sorted(selected))


class BayesianQuadrature(BaseEstimator):
    """Integration as GP regression, WITH error bars (O'Hagan, 1991).

    Monte-Carlo integration converges as ``1/sqrt(n)`` and gives only a noisy point
    estimate. Bayesian quadrature instead puts a Gaussian-process prior on the
    integrand, conditions it on the function evaluations, and integrates the GP
    posterior -- which has a CLOSED FORM for an RBF kernel. The result is not just an
    estimate of the integral but a full posterior over it (a mean and a variance), so
    you know how much to trust it and where to sample next. Far more sample-efficient
    than Monte Carlo for smooth functions. Uniform measure on ``[a, b]`` here.
    """

    def __init__(self, length_scale=0.3, noise=1e-8):
        self.length_scale = length_scale
        self.noise = noise

    def _kernel(self, A, B):
        from scipy.spatial.distance import cdist
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.length_scale ** 2)

    def fit(self, X, y, a=0.0, b=1.0):
        X = check_array(X); y = np.asarray(y, float).ravel()
        self.X_ = X; self.a, self.b = a, b
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.Kinv_ = np.linalg.inv(K)
        # kernel mean z_i = ∫ k(x, x_i) dx over [a, b] (RBF integrates to erf terms)
        from scipy.special import erf
        l = self.length_scale
        z = (np.sqrt(np.pi / 2) * l
             * (erf((b - X[:, 0]) / (np.sqrt(2) * l))
                - erf((a - X[:, 0]) / (np.sqrt(2) * l))))
        self.z_ = z
        self.integral_ = z @ self.Kinv_ @ y
        # posterior variance of the integral
        zz = np.sqrt(np.pi) * l ** 2 * (
            (b - a) / l * erf((b - a) / (2 * l))
            + 2 / np.sqrt(np.pi) * (np.exp(-((b - a) ** 2) / (4 * l ** 2)) - 1))
        self.variance_ = max(zz - z @ self.Kinv_ @ z, 0.0)
        return self

    def integral(self, return_std=False):
        if return_std:
            return self.integral_, np.sqrt(self.variance_)
        return self.integral_


class SparseVariationalGP(BaseEstimator):
    """Gaussian-process regression that SCALES via inducing points (Titsias, 2009).

    Exact GP regression costs ``O(n^3)`` -- hopeless past a few thousand points. A
    sparse GP summarises the data with ``m << n`` INDUCING points and does inference
    through them, dropping the cost to ``O(n m^2)``. The variational formulation
    chooses the inducing outputs to best approximate the true posterior (minimising a
    KL bound), so it keeps GP-quality predictive means AND uncertainty at a fraction of
    the cost. Inducing points placed by k-means here; RBF kernel. Predicts mean and
    standard deviation.
    """

    def __init__(self, n_inducing=10, length_scale=1.0, signal_var=1.0,
                 noise=0.1, random_state=None):
        self.n_inducing = n_inducing
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise = noise
        self.random_state = random_state

    def _kernel(self, A, B):
        from scipy.spatial.distance import cdist
        return self.signal_var * np.exp(
            -0.5 * cdist(A, B, "sqeuclidean") / self.length_scale ** 2)

    def fit(self, X, y):
        from ..cluster import KMeans
        X = check_array(X); y = np.asarray(y, float).ravel()
        rng = check_random_state(self.random_state)
        m = min(self.n_inducing, len(X))
        self.Z_ = KMeans(n_clusters=m, random_state=rng).fit(X).cluster_centers_
        Kmm = self._kernel(self.Z_, self.Z_) + 1e-6 * np.eye(m)
        Knm = self._kernel(X, self.Z_)
        Kmm_inv = np.linalg.inv(Kmm)
        # Titsias predictive: Sigma = (Kmm + Kmn Knm / noise^2)^{-1}
        sig = self.noise ** 2
        A = Kmm + Knm.T @ Knm / sig
        A_inv = np.linalg.inv(A + 1e-6 * np.eye(m))
        self.mu_ = A_inv @ Knm.T @ y / sig               # inducing-point mean
        self.Kmm_inv_ = Kmm_inv
        self.A_inv_ = A_inv
        return self

    def predict(self, X, return_std=False):
        X = check_array(X)
        Ksm = self._kernel(X, self.Z_)
        mean = Ksm @ self.mu_
        if not return_std:
            return mean
        Kss = np.diag(self._kernel(X, X))
        var = Kss - np.einsum("ij,jk,ik->i", Ksm, self.Kmm_inv_, Ksm) \
            + np.einsum("ij,jk,ik->i", Ksm, self.A_inv_, Ksm) + self.noise ** 2
        return mean, np.sqrt(np.maximum(var, 1e-9))


__all__ = ["NestedSampling", "DeterminantalPointProcess", "BayesianQuadrature",
           "SparseVariationalGP"]
