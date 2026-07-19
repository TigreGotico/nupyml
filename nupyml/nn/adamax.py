"""Adam with the INFINITY norm for the second moment (Kingma & Ba, 2015)."""
import numpy as np
from .optim import Optimizer


class Adamax(Optimizer):
    """Adam with the INFINITY norm for the second moment (Kingma & Ba, 2015).

    Adam's ``v`` is an L2 (squared) average of gradients. Adamax replaces it with a
    running MAX of the gradient magnitude (an L∞ norm), which is more stable when
    gradients are sparse or occasionally huge -- the max is not blown up by one
    outlier the way a squared average is. A simple, robust Adam variant.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._u = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, u in zip(self.params, self._m, self._u):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            np.maximum(self.b2 * u, np.abs(g), out=u)   # L-infinity running max
            mhat = m / (1 - self.b1 ** self._t)
            p.data -= self.lr * mhat / (u + self.eps)


__all__ = ["Adamax"]
