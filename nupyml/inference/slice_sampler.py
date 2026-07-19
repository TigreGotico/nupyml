"""MCMC with no step-size to tune -- it finds its own (Neal, 2003)."""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_array, check_random_state


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


__all__ = ["SliceSampler"]
