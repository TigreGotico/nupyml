"""Sublinear-memory Adam by FACTORING the second moment (Shazeer & Stern, 2018)."""
import numpy as np
from .optim import Optimizer


class Adafactor(Optimizer):
    """Sublinear-memory Adam by FACTORING the second moment (Shazeer & Stern, 2018).

    Adam stores a full second-moment tensor -- as much memory as the weights
    themselves, painful for huge matrices. Adafactor stores only per-ROW and
    per-COLUMN averages of a 2-D weight's squared gradients and reconstructs the
    estimate as their outer product, cutting the second-moment memory from
    ``O(m*n)`` to ``O(m + n)``. This factored approximation is what let large
    transformers train under tight memory. Falls back to full moment for vectors.
    """

    def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-30):
        super().__init__(params, lr)
        self.beta2 = beta2
        self.eps = eps
        self._state = []
        for p in self.params:
            if p.data.ndim == 2:
                m, n = p.data.shape
                self._state.append({"r": np.zeros(m), "c": np.zeros(n)})
            else:
                self._state.append({"v": np.zeros_like(p.data)})

    def step(self):
        for p, st in zip(self.params, self._state):
            if p.grad is None:
                continue
            g2 = p.grad ** 2 + self.eps
            if "v" in st:
                st["v"] = self.beta2 * st["v"] + (1 - self.beta2) * g2
                denom = np.sqrt(st["v"])
            else:
                st["r"] = self.beta2 * st["r"] + (1 - self.beta2) * g2.mean(axis=1)
                st["c"] = self.beta2 * st["c"] + (1 - self.beta2) * g2.mean(axis=0)
                # rank-1 reconstruction of the second moment from row/col averages
                est = np.outer(st["r"], st["c"]) / (st["r"].mean() + self.eps)
                denom = np.sqrt(est)
            p.data -= self.lr * p.grad / (denom + 1e-8)


__all__ = ["Adafactor"]
