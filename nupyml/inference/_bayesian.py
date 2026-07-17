"""Approximate Bayesian inference: variational, sequential MC, likelihood-free,
and a latent-variable GP.

The MCMC samplers (``mcmc``) draw from a posterior exactly-in-the-limit but
slowly. These trade exactness for speed or reach settings MCMC cannot: a
fast optimisation-based posterior (VI), a population sampler for hard posteriors
(SMC), inference with NO likelihood (ABC), and unsupervised nonlinear embedding
with uncertainty (GPLVM).
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class MeanFieldVI(BaseEstimator):
    """Variational inference (ADVI) -- turn posterior SAMPLING into OPTIMISATION.

    THE IDEA
    --------
    Instead of sampling the posterior, APPROXIMATE it with a simple distribution
    ``q`` -- here a mean-field Gaussian (independent per dimension) -- and optimise
    ``q``'s parameters to be as close as possible to the true posterior. Closeness
    is measured by the ELBO (evidence lower bound); maximising it minimises the
    KL divergence from ``q`` to the posterior. VI is far faster than MCMC and
    scales to large models -- at the cost of UNDER-estimating variance (the
    mean-field factorisation cannot represent posterior correlations).

    The ELBO is estimated by the reparameterisation trick (sample
    ``z = mu + sigma*eps``, differentiate through it); gradients w.r.t. the
    variational parameters are taken numerically here. ``log_prob`` is the
    unnormalised log-posterior. After fitting, ``mean_`` / ``std_`` describe ``q``.
    """

    def __init__(self, log_prob, dim, n_samples=20, lr=0.05, n_iter=500,
                 random_state=None):
        self.log_prob = log_prob
        self.dim = dim
        self.n_samples = n_samples
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def _grad_log_prob(self, z, h=1e-4):
        """Central finite-difference gradient of log_prob at z (per dimension)."""
        g = np.zeros(self.dim)
        for d in range(self.dim):
            zp = z.copy(); zp[d] += h
            zm = z.copy(); zm[d] -= h
            g[d] = (self.log_prob(zp) - self.log_prob(zm)) / (2 * h)
        return g

    def fit(self, X=None, y=None):
        rng = check_random_state(self.random_state)
        mu = np.zeros(self.dim)
        log_sigma = np.zeros(self.dim)
        # Adam on the variational params keeps the reparam-gradient ascent stable
        m_mu = v_mu = m_ls = v_ls = 0.0
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        for t in range(1, self.n_iter + 1):
            sigma = np.exp(np.clip(log_sigma, -8, 8))
            eps = rng.randn(self.n_samples, self.dim)
            gmu = np.zeros(self.dim); gls = np.zeros(self.dim)
            for e in eps:
                z = mu + sigma * e                    # reparameterised sample
                glp = self._grad_log_prob(z)
                gmu += glp                            # dELBO/dmu = E[grad log p]
                gls += glp * sigma * e                # chain rule through sigma
            gmu /= self.n_samples
            gls = gls / self.n_samples + 1.0          # + dEntropy/dlog_sigma = 1
            # Adam update for the mean and log-sigma parameter vectors
            m_mu = b1 * m_mu + (1 - b1) * gmu; v_mu = b2 * v_mu + (1 - b2) * gmu ** 2
            m_ls = b1 * m_ls + (1 - b1) * gls; v_ls = b2 * v_ls + (1 - b2) * gls ** 2
            mu += self.lr * (m_mu / (1 - b1 ** t)) / (np.sqrt(v_mu / (1 - b2 ** t)) + eps_a)
            log_sigma += self.lr * (m_ls / (1 - b1 ** t)) / (np.sqrt(v_ls / (1 - b2 ** t)) + eps_a)
            log_sigma = np.clip(log_sigma, -8, 8)
        self.mean_ = mu
        self.std_ = np.exp(log_sigma)
        return self


class SequentialMonteCarlo(BaseEstimator):
    """SMC sampler -- anneal from the prior to the posterior with a PARTICLE cloud.

    A single MCMC chain gets stuck in multimodal or tightly-constrained
    posteriors. SMC evolves a POPULATION of particles through a sequence of
    tempered distributions ``p(theta) * likelihood^beta`` as ``beta`` climbs from
    0 (the easy prior) to 1 (the posterior). At each step it REWEIGHTS particles by
    the incremental likelihood, RESAMPLES to kill low-weight ones and clone
    high-weight ones (fighting degeneracy), and MOVES them with a short MCMC
    kernel. The gradual tempering is what lets it cross regions a cold chain never
    would. Returns weighted posterior particles.
    """

    def __init__(self, log_prior, log_likelihood, dim, n_particles=500,
                 n_steps=20, random_state=None):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.dim = dim
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.random_state = random_state

    def fit(self, X=None, y=None):
        rng = check_random_state(self.random_state)
        # initialise from a broad Gaussian prior sample
        parts = rng.randn(self.n_particles, self.dim) * 3
        betas = np.linspace(0, 1, self.n_steps + 1)
        for k in range(1, len(betas)):
            dbeta = betas[k] - betas[k - 1]
            ll = np.array([self.log_likelihood(p) for p in parts])
            logw = dbeta * ll                         # incremental tempering weight
            logw -= logw.max()
            w = np.exp(logw); w /= w.sum()
            idx = rng.choice(self.n_particles, self.n_particles, p=w)   # resample
            parts = parts[idx]
            # MCMC move at the current temperature
            for _ in range(3):
                prop = parts + rng.randn(*parts.shape) * 0.3
                cur_lp = np.array([self.log_prior(p) + betas[k] * self.log_likelihood(p)
                                   for p in parts])
                prop_lp = np.array([self.log_prior(p) + betas[k] * self.log_likelihood(p)
                                    for p in prop])
                accept = np.log(rng.rand(self.n_particles)) < (prop_lp - cur_lp)
                parts[accept] = prop[accept]
        self.particles_ = parts
        self.posterior_mean_ = parts.mean(axis=0)
        return self


class ABC(BaseEstimator):
    """Approximate Bayesian Computation -- inference with NO likelihood function.

    Some models you can SIMULATE from but cannot write a likelihood for (complex
    stochastic simulators). ABC sidesteps the likelihood entirely: sample
    parameters from the prior, SIMULATE data, and ACCEPT the parameters if the
    simulated data's summary statistics are within ``epsilon`` of the observed
    ones. The accepted parameters approximate the posterior -- exactly so as
    ``epsilon -> 0``. The only requirements are a prior sampler, a simulator, and
    a distance on summaries; the likelihood never appears.

    ``prior_sampler() -> theta``; ``simulator(theta) -> summary``. ``fit(observed)``
    collects accepted parameters into ``posterior_samples_``.
    """

    def __init__(self, prior_sampler, simulator, epsilon, n_accept=200,
                 max_trials=100000, random_state=None):
        self.prior_sampler = prior_sampler
        self.simulator = simulator
        self.epsilon = epsilon
        self.n_accept = n_accept
        self.max_trials = max_trials
        self.random_state = random_state

    def fit(self, observed):
        rng = check_random_state(self.random_state)
        obs = np.asarray(observed, float)
        accepted, trials = [], 0
        while len(accepted) < self.n_accept and trials < self.max_trials:
            theta = self.prior_sampler(rng)
            sim = np.asarray(self.simulator(theta, rng), float)
            if np.linalg.norm(sim - obs) < self.epsilon:   # accept if close enough
                accepted.append(np.atleast_1d(theta))
            trials += 1
        self.posterior_samples_ = np.array(accepted)
        self.posterior_mean_ = self.posterior_samples_.mean(axis=0)
        self.acceptance_rate_ = len(accepted) / max(1, trials)
        return self


class GPLVM(BaseEstimator):
    """Gaussian Process Latent Variable Model -- nonlinear probabilistic PCA.

    PCA finds a LINEAR low-dimensional embedding. GPLVM finds a NONLINEAR one: it
    posits latent coordinates ``X`` (low-dim) that generate the observed
    high-dim ``Y`` through a Gaussian process, and OPTIMISES the latent
    coordinates to maximise the GP marginal likelihood of ``Y``. Because the
    mapping is a GP, the embedding can be arbitrarily curved (where PCA is stuck
    with a plane), and it comes with a principled probabilistic model. Here the
    latents are optimised by gradient ascent on the RBF-kernel log-marginal
    likelihood. ``fit`` returns the learned ``X_`` embedding.

    Lawrence (2005).
    """

    def __init__(self, n_components=2, length_scale=1.0, lr=0.01, n_iter=200,
                 random_state=None):
        self.n_components = n_components
        self.length_scale = length_scale
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def _kernel(self, X):
        sq = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
        return np.exp(-sq / (2 * self.length_scale ** 2))

    def _neg_log_marg(self, X, Y):
        n, d = Y.shape
        K = self._kernel(X) + 1e-6 * np.eye(len(X))
        Kinv = np.linalg.inv(K)
        sign, logdet = np.linalg.slogdet(K)
        # GP marginal likelihood of Y under latents X (summed over output dims)
        return 0.5 * d * logdet + 0.5 * np.trace(Kinv @ Y @ Y.T)

    def fit(self, Y):
        Y = check_array(Y)
        rng = check_random_state(self.random_state)
        # initialise the latents from PCA for a good starting point
        Yc = Y - Y.mean(0)
        U, S, _ = np.linalg.svd(Yc, full_matrices=False)
        X = U[:, :self.n_components] * S[:self.n_components]
        X = X / (X.std(0) + 1e-8)
        h = 1e-4
        for _ in range(self.n_iter):
            base = self._neg_log_marg(X, Y)
            grad = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Xp = X.copy(); Xp[i, j] += h
                    grad[i, j] = (self._neg_log_marg(Xp, Y) - base) / h
            # normalise the step: GPLVM's marginal likelihood is non-identifiable
            # up to rotation/scale, so a raw large step wanders off; a unit-norm
            # descent direction keeps it near the good PCA initialisation
            gnorm = np.linalg.norm(grad) + 1e-12
            X -= self.lr * grad / gnorm * np.sqrt(X.size)
        self.X_ = X
        return self

    def fit_transform(self, Y):
        return self.fit(Y).X_


__all__ = ["MeanFieldVI", "SequentialMonteCarlo", "ABC", "GPLVM"]
