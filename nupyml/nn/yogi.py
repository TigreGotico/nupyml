"""Control Adam's second-moment GROWTH additively (Zaheer et al., 2018)."""
import numpy as np
from .optim import Optimizer


class Yogi(Optimizer):
    """Control Adam's second-moment GROWTH additively (Zaheer et al., 2018).

    Adam's ``v`` can grow (or shrink) fast, causing the effective learning rate to
    swing and sometimes stopping convergence. Yogi changes the ``v`` update from
    multiplicative to ADDITIVE, gated by a sign::

        v <- v - (1 - b2) * sign(v - g^2) * g^2

    so ``v`` only ever changes by a controlled amount per step -- it cannot spike
    on one large gradient. Same cost as Adam, steadier adaptation.
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.full_like(p.data, 1e-6) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            g2 = g ** 2
            v -= (1 - self.b2) * np.sign(v - g2) * g2   # additive, sign-gated
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(np.abs(vhat)) + self.eps)


__all__ = ["Yogi"]
