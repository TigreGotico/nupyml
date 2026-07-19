"""Layer-wise Adaptive Rate Scaling -- one trust ratio per layer (You, 2017)."""
import numpy as np
from .optim import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling -- one trust ratio per layer (You, 2017).

    When you scale batch size up to thousands, a single global learning rate is
    wrong for every layer at once: early layers have tiny weights and huge
    gradients, late layers the reverse. LARS sets each layer's effective rate from
    its own TRUST RATIO ``||w|| / ||g + wd·w||`` -- so the step is always a fixed
    fraction of the weight's magnitude, regardless of how big or small that layer's
    gradients happen to be. This is what let ResNet train with batch size 32k.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-4,
                 eps=1e-8, clip=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps
        self.clip = clip                             # True -> LARC
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self._v):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            w_norm = np.linalg.norm(p.data)
            g_norm = np.linalg.norm(g)
            if w_norm > 0 and g_norm > 0:
                trust = w_norm / (g_norm + self.eps)
                # LARC clips the local rate so it never exceeds the global lr
                local_lr = min(trust, 1.0) if self.clip else trust
            else:
                local_lr = 1.0
            v *= self.momentum
            v += local_lr * g
            p.data -= self.lr * v

class LARC(LARS):
    """LARS with the trust ratio CLIPPED to 1 (Ginsburg, 2018).

    Plain LARS can hand a layer a local rate far above the global one early in
    training, which occasionally blows up. LARC clips the trust ratio at 1 so a
    layer's step is never larger than the global learning rate would give -- it can
    only ever slow a layer DOWN, never speed it past the global budget. Safer, and
    the usual default in practice.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-4, eps=1e-8):
        super().__init__(params, lr, momentum, weight_decay, eps, clip=True)


__all__ = ["LARS", "LARC"]
