"""Rectified Adam: fix Adam's unreliable, high-variance early steps."""
import numpy as np
from .optim import Optimizer


class RAdam(Optimizer):
    """Rectified Adam: fix Adam's unreliable, high-variance early steps.

    THE PROBLEM
    -----------
    Adam's adaptive term (dividing by ``sqrt(v)``) is computed from very few
    samples in the first steps, so its VARIANCE is huge -- the early updates are
    erratic, which is why Adam so often needs a manual learning-rate WARMUP to not
    blow up at the start.

    THE FIX
    -------
    RAdam estimates when the variance of the adaptive term is trustworthy and
    TURNS THE ADAPTIVITY OFF until then, falling back to plain momentum (SGD-like)
    for the first few steps and switching adaptivity on once enough gradient
    history exists. It is, in effect, an automatic, principled warmup -- the
    rectification term is derived, not tuned -- so RAdam often trains stably with
    no warmup schedule at all.

    Liu et al. (2019).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.betas
        # rho_inf and rho_t track how many effective samples the variance estimate
        # has -- the machinery that decides when adaptivity is safe
        rho_inf = 2 / (1 - b2) - 1
        rho_t = rho_inf - 2 * self._t * b2 ** self._t / (1 - b2 ** self._t)
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self._t)
            if rho_t > 4:
                # variance is trustworthy: use the adaptive step, with the
                # derived rectification factor r
                v_hat = np.sqrt(v / (1 - b2 ** self._t))
                r = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf)
                            / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                p.data -= self.lr * r * m_hat / (v_hat + self.eps)
            else:
                # too early to trust the adaptivity: plain momentum, the automatic
                # warmup
                p.data -= self.lr * m_hat


__all__ = ["RAdam"]
