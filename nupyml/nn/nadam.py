"""Adam with Nesterov momentum: look ahead before you step."""
import numpy as np
from .optim import Optimizer


class Nadam(Optimizer):
    """Adam with Nesterov momentum: look ahead before you step.

    Plain momentum steps along the accumulated velocity. NESTEROV momentum peeks
    one step ahead -- it evaluates the gradient AT the point the velocity is about
    to carry it to -- which lets it correct an overshoot before committing. Nadam
    folds that lookahead into Adam's first moment. The gain over Adam is usually
    small but real, especially early in training; it is Adam plus a touch of
    foresight.

    Dozat (2016).
    """

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self._t)
            v_hat = v / (1 - b2 ** self._t)
            # the Nesterov term blends the corrected momentum with the current
            # gradient, so the effective direction already anticipates the step
            m_nesterov = b1 * m_hat + (1 - b1) * g / (1 - b1 ** self._t)
            p.data -= self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)


__all__ = ["Nadam"]
