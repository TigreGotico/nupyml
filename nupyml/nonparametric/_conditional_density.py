"""Kernel conditional density estimation: the whole distribution p(y | x).

Regression predicts E[y | x] -- a single value. When the conditional distribution
is skewed, multimodal, or heteroscedastic, that mean is misleading. Conditional
KDE estimates the FULL density ``p(y | x)`` non-parametrically, so you can read
off modes, spread, and quantiles that change with ``x``.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array


class ConditionalKDE(BaseEstimator):
    """Nadaraya-Watson kernel estimate of the conditional density ``p(y | x)``.

    THE IDEA
    --------
    Weight each training point by how close its ``x`` is to the query (an
    x-kernel), then form a kernel density estimate of ``y`` using those weights::

        p(y | x) = sum_i w_i(x) K_h(y - y_i),   w_i(x) proportional to K_hx(x - x_i)

    So the conditional density at ``x`` is a LOCAL, x-weighted KDE over the
    training targets -- it uses mostly the neighbours of ``x`` and their ``y``
    values. This captures a conditional that is bimodal or whose spread changes
    with ``x``, which no mean-regressor can. ``pdf(x, y)`` evaluates the density;
    ``conditional_mean`` and ``sample`` follow from the same weights.
    """

    def __init__(self, bandwidth_x=1.0, bandwidth_y=0.5):
        self.bandwidth_x = bandwidth_x
        self.bandwidth_y = bandwidth_y

    def fit(self, X, y):
        self.X_ = check_array(X)
        self.y_ = np.asarray(y, float).ravel()
        return self

    def _weights(self, x):
        from scipy.spatial.distance import cdist
        d2 = cdist(np.atleast_2d(x), self.X_) ** 2
        w = np.exp(-0.5 * d2 / self.bandwidth_x ** 2)
        return w / (w.sum(axis=1, keepdims=True) + 1e-12)   # (n_query, n_train)

    def pdf(self, x, y_grid):
        """Conditional density of each value in ``y_grid`` given a single ``x``."""
        check_is_fitted(self, "X_")
        w = self._weights(np.atleast_2d(x))[0]
        y_grid = np.atleast_1d(y_grid)
        # x-weighted KDE of the training y's, evaluated on the grid
        diff = (y_grid[:, None] - self.y_[None, :]) / self.bandwidth_y
        kern = np.exp(-0.5 * diff ** 2) / (self.bandwidth_y * np.sqrt(2 * np.pi))
        return kern @ w

    def conditional_mean(self, X):
        w = self._weights(check_array(X))
        return w @ self.y_

    def sample(self, x, n=1, random_state=None):
        from ..utils import check_random_state
        rng = check_random_state(random_state)
        w = self._weights(np.atleast_2d(x))[0]
        idx = rng.choice(len(self.y_), size=n, p=w)         # pick a neighbour...
        return self.y_[idx] + rng.randn(n) * self.bandwidth_y   # ...and jitter


__all__ = ["ConditionalKDE"]
