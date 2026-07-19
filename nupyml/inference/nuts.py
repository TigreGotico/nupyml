"""HMC that picks its own trajectory length (Hoffman & Gelman, 2014)."""
import numpy as np
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


__all__ = ["NUTS"]
