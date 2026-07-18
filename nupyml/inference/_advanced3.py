"""Probabilistic v4: ensemble data assimilation, regime switching, Bayesian matrix
factorisation, and Rao-Blackwellised particle filtering.

Four more probabilistic models. The Ensemble Kalman Filter tracks a high-dimensional
state with a small ensemble instead of a huge covariance. The Markov-switching model
lets a series jump between regimes. Bayesian probabilistic matrix factorisation puts
a posterior on the latent factors. The Rao-Blackwellised particle filter marginalises
out the part of the state that stays Gaussian, so far fewer particles are needed.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class EnsembleKalmanFilter(BaseEstimator):
    """Track a state with a small ENSEMBLE, not a big covariance (Evensen, 1994).

    The Kalman filter needs the full ``n x n`` state covariance -- impossible when
    the state is a weather field with millions of variables. The Ensemble Kalman
    Filter represents the distribution by a handful of sample states (the ENSEMBLE)
    and reads the covariance it needs off their spread. Each step propagates every
    ensemble member through the (possibly nonlinear) dynamics, then nudges them
    toward the observation with a Kalman gain computed from the ENSEMBLE covariance.
    It is the workhorse of geophysical data assimilation. ``transition`` and
    ``observation`` are functions (or matrices) of the state.
    """

    def __init__(self, transition, observation, process_cov, obs_cov,
                 n_ensemble=50, random_state=None):
        self.transition = transition
        self.observation = observation
        self.process_cov = process_cov
        self.obs_cov = obs_cov
        self.n_ensemble = n_ensemble
        self.random_state = random_state

    def _apply(self, fn, x):
        return fn @ x if isinstance(fn, np.ndarray) else fn(x)

    def filter(self, observations, x0):
        rng = check_random_state(self.random_state)
        x0 = np.asarray(x0, float)
        d = len(x0)
        ens = rng.multivariate_normal(x0, np.eye(d), self.n_ensemble).T   # (d, N)
        means = []
        Q = np.atleast_2d(self.process_cov)
        R = np.atleast_2d(self.obs_cov)
        for y in np.atleast_1d(observations):
            y = np.atleast_1d(y)
            # forecast every member through the dynamics + process noise
            ens = np.column_stack([self._apply(self.transition, ens[:, i])
                                   + rng.multivariate_normal(np.zeros(d), Q)
                                   for i in range(self.n_ensemble)])
            hens = np.column_stack([np.atleast_1d(self._apply(self.observation,
                                    ens[:, i])) for i in range(self.n_ensemble)])
            xbar = ens.mean(axis=1, keepdims=True)
            hbar = hens.mean(axis=1, keepdims=True)
            A = ens - xbar; Hd = hens - hbar
            Pxy = A @ Hd.T / (self.n_ensemble - 1)        # ensemble cross-covariance
            Pyy = Hd @ Hd.T / (self.n_ensemble - 1) + R
            K = Pxy @ np.linalg.inv(Pyy)                  # ensemble Kalman gain
            for i in range(self.n_ensemble):              # perturbed-observation update
                innov = y + rng.multivariate_normal(np.zeros(len(y)), R) \
                    - hens[:, i]
                ens[:, i] = ens[:, i] + K @ innov
            means.append(ens.mean(axis=1))
        self.state_means_ = np.array(means)
        return self.state_means_


class MarkovSwitchingModel(BaseEstimator):
    """A series that JUMPS between regimes (Hamilton, 1989).

    Some series are not one process but several, switching between them -- calm vs
    crisis markets, expansion vs recession. The Markov-switching model fits a
    separate mean/variance (and optional AR) for each REGIME and a Markov transition
    matrix between them, then runs the Hamilton FILTER: a forward recursion that
    combines the transition probabilities with each regime's likelihood to give, at
    every time, the probability the series is in each regime. Parameters are fit by
    EM. ``predict_regimes`` returns the smoothed regime probabilities.
    """

    def __init__(self, n_regimes=2, max_iter=100, tol=1e-4, random_state=None):
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        rng = check_random_state(self.random_state)
        K, n = self.n_regimes, len(y)
        # init means spread across the data range, equal variances
        self.means_ = np.quantile(y, np.linspace(0.2, 0.8, K))
        self.vars_ = np.full(K, y.var())
        self.trans_ = np.full((K, K), 1.0 / K)
        self.pi_ = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            B = np.exp(-0.5 * (y[:, None] - self.means_) ** 2 / self.vars_) \
                / np.sqrt(2 * np.pi * self.vars_)          # (n, K) likelihoods
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((K, K))
            for t in range(n - 1):
                num = (alpha[t][:, None] * self.trans_
                       * B[t + 1][None, :] * beta[t + 1][None, :])
                xi += num / num.sum()
            self.trans_ = xi / xi.sum(axis=1, keepdims=True)
            self.pi_ = gamma[0]
            nk = gamma.sum(axis=0)
            self.means_ = (gamma * y[:, None]).sum(axis=0) / nk
            self.vars_ = np.maximum(
                (gamma * (y[:, None] - self.means_) ** 2).sum(axis=0) / nk, 1e-6)
            ll = np.log(c).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.smoothed_ = gamma
        return self

    def _forward(self, B):
        n, K = B.shape
        alpha = np.zeros((n, K)); c = np.zeros(n)
        alpha[0] = self.pi_ * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.trans_) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        return alpha, c

    def _backward(self, B, c):
        n, K = B.shape
        beta = np.zeros((n, K)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (self.trans_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def predict_regimes(self, y=None):
        return self.smoothed_


class BayesianPMF(BaseEstimator):
    """Matrix factorisation with a POSTERIOR on the factors (Salakhutdinov, 2008).

    Ordinary matrix factorisation gives point estimates of the user/item factors and
    so cannot say how uncertain a prediction is -- risky for a cold-start user with
    one rating. Bayesian PMF places Gaussian priors on the factors and samples their
    POSTERIOR by Gibbs sampling (each user/item factor is Gaussian given the others),
    averaging predictions over the samples. The result is a predictive DISTRIBUTION:
    the mean is the rating, and its spread widens for users/items with little data.
    Works on a dense rating matrix with NaN for missing entries.
    """

    def __init__(self, n_factors=5, n_samples=50, burn_in=20, alpha=2.0,
                 random_state=None):
        self.n_factors = n_factors
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.alpha = alpha                                 # observation precision
        self.random_state = random_state

    def fit(self, R):
        R = np.asarray(R, float)
        rng = check_random_state(self.random_state)
        n, m = R.shape
        k = self.n_factors
        mask = ~np.isnan(R)
        U = rng.randn(n, k) * 0.1
        V = rng.randn(m, k) * 0.1
        preds = []
        for it in range(self.n_samples + self.burn_in):
            U = self._sample_factors(R, mask, V, rng)      # p(U | V, R)
            V = self._sample_factors(R.T, mask.T, U, rng)  # p(V | U, R)
            if it >= self.burn_in:
                preds.append(U @ V.T)
        self.samples_ = np.array(preds)
        self.mean_ = self.samples_.mean(axis=0)
        self.std_ = self.samples_.std(axis=0)
        return self

    def _sample_factors(self, R, mask, W, rng):
        n, k = R.shape[0], W.shape[1]
        out = np.zeros((n, k))
        prior = np.eye(k)
        for i in range(n):
            obs = mask[i]
            if not obs.any():
                out[i] = rng.multivariate_normal(np.zeros(k), prior)
                continue
            Wi = W[obs]
            prec = prior + self.alpha * Wi.T @ Wi          # posterior precision
            cov = np.linalg.inv(prec)
            mean = cov @ (self.alpha * Wi.T @ R[i, obs])
            out[i] = rng.multivariate_normal(mean, cov)
        return out

    def predict(self, i, j):
        return self.mean_[i, j], self.std_[i, j]


class RaoBlackwellisedParticleFilter(BaseEstimator):
    """Marginalise out the LINEAR part, sample only the rest (Doucet, 2000).

    A plain particle filter needs exponentially many particles as the state grows.
    But often part of the state is conditionally LINEAR-GAUSSIAN given the rest -- a
    jump-Markov system where a discrete MODE switches between linear dynamics. The
    Rao-Blackwellised particle filter samples only the nonlinear/discrete part with
    particles and runs an EXACT Kalman filter for the linear part inside each
    particle, so variance collapses and a handful of particles suffice. Here: a
    linear-Gaussian state whose observation MODE (e.g. sensor on/off, gain level)
    is the sampled discrete variable.
    """

    def __init__(self, A, modes, process_cov, obs_cov, n_particles=100,
                 mode_trans=None, random_state=None):
        self.A = np.atleast_2d(A)
        self.modes = modes                                 # list of observation matrices
        self.process_cov = np.atleast_2d(process_cov)
        self.obs_cov = np.atleast_2d(obs_cov)
        self.n_particles = n_particles
        self.mode_trans = mode_trans
        self.random_state = random_state

    def filter(self, observations, x0, P0):
        rng = check_random_state(self.random_state)
        d = len(x0); M = len(self.modes)
        trans = self.mode_trans if self.mode_trans is not None \
            else np.full((M, M), 1.0 / M)
        modes = rng.randint(0, M, self.n_particles)        # particle discrete modes
        means = np.tile(np.asarray(x0, float), (self.n_particles, 1))
        covs = np.tile(np.atleast_2d(P0), (self.n_particles, 1, 1))
        weights = np.full(self.n_particles, 1.0 / self.n_particles)
        state_est, mode_est = [], []
        for y in np.atleast_1d(observations):
            y = np.atleast_1d(y)
            for p in range(self.n_particles):
                modes[p] = rng.choice(M, p=trans[modes[p]])   # sample the next mode
                H = np.atleast_2d(self.modes[modes[p]])
                # Kalman predict + update (exact, for the linear state)
                m = self.A @ means[p]
                P = self.A @ covs[p] @ self.A.T + self.process_cov
                S = H @ P @ H.T + self.obs_cov
                Kg = P @ H.T @ np.linalg.inv(S)
                innov = y - H @ m
                means[p] = m + Kg @ innov
                covs[p] = P - Kg @ H @ P
                weights[p] *= np.exp(-0.5 * innov @ np.linalg.inv(S) @ innov) \
                    / np.sqrt(np.linalg.det(2 * np.pi * S))
            weights = np.maximum(weights, 1e-300); weights /= weights.sum()
            state_est.append(weights @ means)
            mode_est.append(np.bincount(modes, weights, minlength=M).argmax())
            if 1.0 / (weights ** 2).sum() < self.n_particles / 2:   # resample
                idx = rng.choice(self.n_particles, self.n_particles, p=weights)
                modes, means, covs = modes[idx], means[idx], covs[idx]
                weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.state_means_ = np.array(state_est)
        self.mode_estimates_ = np.array(mode_est)
        return self.state_means_


__all__ = ["EnsembleKalmanFilter", "MarkovSwitchingModel", "BayesianPMF",
           "RaoBlackwellisedParticleFilter"]
