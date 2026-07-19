"""Adam that TRANSITIONS into SGD via dynamic learning-rate bounds"""
import numpy as np
from .optim import Optimizer


class AdaBound(Optimizer):
    """Adam that TRANSITIONS into SGD via dynamic learning-rate bounds
    (Luo et al., 2019).

    Adam converges fast but generalises worse than SGD; the culprit is a handful
    of extreme per-parameter learning rates late in training. AdaBound clips each
    parameter's effective step into a ``[lower, upper]`` band that starts wide
    (pure Adam) and, over training, both bounds converge to the SGD rate. So it is
    Adam early (fast) and SGD late (generalises) -- a smooth handover, no switch
    point to tune.
    """

    def __init__(self, params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999),
                 gamma=1e-3, eps=1e-8):
        super().__init__(params, lr)
        self.final_lr = final_lr
        self.b1, self.b2 = betas
        self.gamma = gamma
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        # the band shrinks toward final_lr as t grows
        lower = self.final_lr * (1 - 1 / (self.gamma * self._t + 1))
        upper = self.final_lr * (1 + 1 / (self.gamma * self._t))
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            step = self.lr / (np.sqrt(vhat) + self.eps)
            step = np.clip(step, lower, upper)         # bound each per-param step
            p.data -= step * mhat


__all__ = ["AdaBound"]
