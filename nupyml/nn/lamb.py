"""Layer-wise Adaptive Moments for large-BATCH training (You et al., 2020)."""
import numpy as np
from .optim import Optimizer, LRScheduler


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments for large-BATCH training (You et al., 2020).

    Training with very large batches needs large learning rates, which destabilise
    Adam. LAMB fixes this with a per-LAYER TRUST RATIO: it computes the Adam update
    for a parameter tensor, then rescales it by ``||weights|| / ||update||`` so the
    step is a bounded fraction of the weight's own magnitude, regardless of the raw
    gradient scale. That layer-local normalisation is what let BERT train in 76
    minutes -- it keeps every layer's relative step sane at batch sizes where a
    global learning rate cannot.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            update = mhat / (np.sqrt(vhat) + self.eps) + self.weight_decay * p.data
            w_norm = np.linalg.norm(p.data)
            u_norm = np.linalg.norm(update)
            trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
            p.data -= self.lr * trust * update      # step scaled to the weight's size


__all__ = ["LAMB"]
