"""Adam that adapts to the BELIEF in the gradient (Zhuang et al., 2020)."""
import numpy as np
from .optim import Optimizer


class AdaBelief(Optimizer):
    """Adam that adapts to the BELIEF in the gradient (Zhuang et al., 2020).

    Adam divides by the second moment of the gradient (its size). AdaBelief divides
    by the variance of the gradient AROUND its running mean -- ``(g - m)^2`` instead
    of ``g^2``. So a steadily-pointing gradient (small deviation from ``m``) gets a
    LARGE step even when its magnitude is large, and a noisy, direction-flipping
    one gets a small step. It reads the CONFIDENCE in the direction, not just the
    magnitude -- yielding faster convergence and better generalisation than Adam
    with a one-line change.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._s = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, s in zip(self.params, self._m, self._s):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            diff = g - m                              # deviation from the belief
            s *= self.b2; s += (1 - self.b2) * diff ** 2
            mhat = m / (1 - self.b1 ** self._t)
            shat = s / (1 - self.b2 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(shat) + self.eps)


__all__ = ["AdaBelief"]
