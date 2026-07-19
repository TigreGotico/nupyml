"""Marginalise out the LINEAR part, sample only the rest (Doucet, 2000)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


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


__all__ = ["RaoBlackwellisedParticleFilter"]
