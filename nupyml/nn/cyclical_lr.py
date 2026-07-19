"""Triangular cycling between a low and high LR (Smith, 2017)."""
import numpy as np
from .optim import Optimizer, LRScheduler


class CyclicalLR(LRScheduler):
    """Triangular cycling between a low and high LR (Smith, 2017).

    Instead of only decreasing, the LR sweeps UP and down between ``base_lr`` and
    ``max_lr`` over ``2*step_size`` epochs. The periodic high phase helps the
    optimiser escape saddle points and poor local minima; the low phase lets it
    settle. Cheap, and often removes the need to tune a fixed LR.
    """

    def __init__(self, optimizer, max_lr, step_size=2000):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.step_size = step_size

    def get_lr(self):
        cycle = np.floor(1 + self.epoch / (2 * self.step_size))
        x = np.abs(self.epoch / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * max(0.0, 1 - x)


__all__ = ["CyclicalLR"]
