"""Move a set of PARTICLES to the posterior by a deterministic flow"""
import numpy as np
from scipy.spatial.distance import cdist, pdist
from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_array, check_random_state


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


__all__ = ["SVGD"]
