"""Adagrad's fix that needs NO learning rate (Zeiler, 2012)."""
import numpy as np
from .optim import Optimizer, LRScheduler


class Adadelta(Optimizer):
    """Adagrad's fix that needs NO learning rate (Zeiler, 2012).

    Adagrad's accumulator grows without bound, so its step decays to zero.
    Adadelta uses a DECAYING average of squared gradients (like RMSprop) AND a
    decaying average of squared UPDATES, forming the step from their ratio::

        step = -(sqrt(E[dx^2] + eps) / sqrt(E[g^2] + eps)) * g

    The two running averages have the same units, so they cancel -- which is why
    Adadelta has no ``lr`` to tune at all. It self-scales the step from the recent
    history of both gradients and updates.
    """

    def __init__(self, params, rho=0.95, eps=1e-6):
        super().__init__(params, lr=1.0)
        self.rho = rho
        self.eps = eps
        self._Eg = [np.zeros_like(p.data) for p in self.params]
        self._Edx = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, Eg, Edx in zip(self.params, self._Eg, self._Edx):
            if p.grad is None:
                continue
            g = p.grad
            Eg *= self.rho; Eg += (1 - self.rho) * g ** 2
            dx = -np.sqrt(Edx + self.eps) / np.sqrt(Eg + self.eps) * g
            Edx *= self.rho; Edx += (1 - self.rho) * dx ** 2
            p.data += dx


__all__ = ["Adadelta"]
