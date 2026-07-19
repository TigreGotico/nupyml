"""Contextual bandit by POSTERIOR SAMPLING of a linear reward model"""
import numpy as np
from ..utils import check_random_state


class LinearThompsonSampling:
    """Contextual bandit by POSTERIOR SAMPLING of a linear reward model
    (Agrawal & Goyal, 2013).

    LinUCB adds an optimism BONUS to each arm's predicted reward; linear Thompson
    sampling instead maintains a Bayesian posterior over the reward weights and
    SAMPLES a weight vector from it each round, acting greedily w.r.t. the sample.
    The sampling itself provides exploration -- arms the posterior is uncertain
    about occasionally get a high sampled value and are tried. It often matches or
    beats LinUCB and needs no bonus to tune; the Gaussian posterior over ``theta``
    has a closed-form update per observed (context, reward).
    """

    def __init__(self, n_features, alpha=1.0, v=1.0, random_state=None):
        self.d = n_features
        self.alpha = alpha
        self.v = v
        self._rng = check_random_state(random_state)
        self.A = np.eye(n_features) / alpha           # precision
        self.b = np.zeros(n_features)

    def select(self, contexts):
        """contexts: (n_arms, n_features). Return the chosen arm index."""
        contexts = np.asarray(contexts, float)
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        theta = self._rng.multivariate_normal(mu, self.v ** 2 * A_inv)  # sample
        return int(np.argmax(contexts @ theta))

    def update(self, context, reward):
        x = np.asarray(context, float)
        self.A += np.outer(x, x)                       # Bayesian posterior update
        self.b += reward * x


__all__ = ["LinearThompsonSampling"]
