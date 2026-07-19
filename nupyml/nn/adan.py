"""Adaptive Nesterov Momentum -- look-ahead on the gradient DIFFERENCE"""
import numpy as np
from .optim import Optimizer


class Adan(Optimizer):
    """Adaptive Nesterov Momentum -- look-ahead on the gradient DIFFERENCE
    (Xie, 2022).

    Adam smooths the gradient and its square. Adan additionally smooths the
    gradient's CHANGE between steps (``g_t - g_{t-1}``), a discrete stand-in for
    the Nesterov look-ahead, and folds it into both the update direction and the
    second-moment scaling. In practice it reaches a given loss in noticeably fewer
    steps than AdamW across vision and language models, with the same decoupled
    weight decay.
    """

    def __init__(self, params, lr=0.01, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]  # of grad diffs
        self._n = [np.zeros_like(p.data) for p in self.params]  # second moment
        self._prev_g = [None for _ in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2, b3 = self.betas
        for i, (p, m, v, n) in enumerate(zip(self.params, self._m, self._v,
                                             self._n)):
            if p.grad is None:
                continue
            g = p.grad
            prev = self._prev_g[i]
            diff = g - prev if prev is not None else np.zeros_like(g)
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * diff
            comb = g + (1 - b2) * diff
            n *= b3; n += (1 - b3) * comb * comb
            bc1 = 1 - b1 ** self._t
            bc2 = 1 - b2 ** self._t
            bc3 = 1 - b3 ** self._t
            update = (m / bc1 + (1 - b2) * v / bc2) / (np.sqrt(n / bc3) + self.eps)
            p.data -= self.lr * update
            p.data /= (1 + self.lr * self.weight_decay)   # decoupled decay
            self._prev_g[i] = g.copy()


__all__ = ["Adan"]
