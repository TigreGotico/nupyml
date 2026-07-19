"""Per-parameter learning rates from the ACCUMULATED squared gradient."""
import numpy as np
from .optim import Optimizer


class Adagrad(Optimizer):
    """Per-parameter learning rates from the ACCUMULATED squared gradient.

    THE IDEA, AND WHY IT PRECEDES ADAM
    ----------------------------------
    Give each parameter its own step size, shrinking with how much gradient it has
    seen: divide by ``sqrt(sum of past squared gradients)``. Rare features (small
    accumulated gradient) keep large steps; common ones settle down. This made
    Adagrad excellent for SPARSE data -- text, one-hot features -- where most
    parameters are updated seldom and deserve to move far when they are.

    THE FATAL FLAW
    --------------
    The accumulator only ever GROWS, so the effective learning rate marches
    monotonically to zero and training grinds to a halt before converging. Every
    adaptive optimizer since -- RMSprop, Adam -- exists to fix exactly this by
    using a DECAYING average instead of a running sum. Adagrad is here as the
    origin of per-parameter rates and the cautionary tale that motivated the rest.

    Duchi, Hazan & Singer (2011).
    """

    def __init__(self, params, lr=0.01, eps=1e-10):
        super().__init__(params, lr)
        self.eps = eps
        self._sum_sq = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, s in zip(self.params, self._sum_sq):
            if p.grad is None:
                continue
            s += p.grad ** 2                    # accumulates forever -> LR decays
            p.data -= self.lr * p.grad / (np.sqrt(s) + self.eps)


__all__ = ["Adagrad"]
