"""Adam's adaptivity with ONE second-moment number per layer (Ginsburg, 2019)."""
import numpy as np
from .optim import Optimizer
from .lars_larc import LARS


class NovoGrad(Optimizer):
    """Adam's adaptivity with ONE second-moment number per layer (Ginsburg, 2019).

    Adam stores a second moment for every weight -- as much memory as the model
    itself. NovoGrad keeps a single SCALAR second moment per layer (the running
    norm of that layer's gradient), normalises the gradient by it, then applies
    momentum and DECOUPLED weight decay. The result is layer-wise adaptive like
    LARS but with Adam-style smoothing, at a fraction of Adam's memory -- and it
    tends to be more robust to the initial learning rate.
    """

    def __init__(self, params, lr=0.01, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [0.0 for _ in self.params]
        self._init = [False for _ in self.params]

    def step(self):
        b1, b2 = self.betas
        for i, (p, m) in enumerate(zip(self.params, self._m)):
            if p.grad is None:
                continue
            g = p.grad
            gnorm2 = float(np.sum(g * g))            # scalar second moment
            if not self._init[i]:
                self._v[i] = gnorm2
                self._init[i] = True
            else:
                self._v[i] = b2 * self._v[i] + (1 - b2) * gnorm2
            gn = g / (np.sqrt(self._v[i]) + self.eps)
            gn = gn + self.weight_decay * p.data     # decoupled decay
            m *= b1
            m += (1 - b1) * gn
            p.data -= self.lr * m


__all__ = ["NovoGrad"]
