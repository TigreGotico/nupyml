"""Adam with a NON-DECREASING second-moment (Reddi et al., 2018)."""
import numpy as np
from .optim import Optimizer, LRScheduler


class AMSGrad(Optimizer):
    """Adam with a NON-DECREASING second-moment (Reddi et al., 2018).

    Adam can fail to converge because its ``v`` (second moment) can shrink, letting
    a rare large-gradient direction get a huge step. AMSGrad keeps the running
    MAXIMUM of ``v`` and divides by that, so the effective learning rate is
    monotonically non-increasing per parameter -- the small fix that restores the
    convergence guarantee Adam's original proof was missing.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._vhat = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v, vhat in zip(self.params, self._m, self._v, self._vhat):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            np.maximum(vhat, v, out=vhat)           # the running max -> monotone LR
            mhat = m / (1 - self.b1 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


__all__ = ["AMSGrad"]
