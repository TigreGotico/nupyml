"""Track a state with a small ENSEMBLE, not a big covariance (Evensen, 1994)."""
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


__all__ = ["EnsembleKalmanFilter"]
