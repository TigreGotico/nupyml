"""The 1cycle policy: warm up to a peak, then anneal below the start"""
import numpy as np
from .optim import Optimizer, LRScheduler


class OneCycleLR(LRScheduler):
    """The 1cycle policy: warm up to a peak, then anneal below the start
    (Smith & Topin, 2019).

    One rise from ``base_lr`` to ``max_lr`` over the first ``pct_start`` of
    training, then a cosine anneal all the way down to nearly zero. The single
    warmup-then-decay hump enables "super-convergence" -- training in far fewer
    epochs than a fixed schedule -- and pairs the LR rise with a momentum dip
    (omitted here; LR is the part that matters most).
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.warm = int(pct_start * total_steps)

    def get_lr(self):
        if self.epoch < self.warm:                  # linear warmup to the peak
            return self.base_lr + (self.max_lr - self.base_lr) * self.epoch / max(1, self.warm)
        # cosine anneal from the peak down to ~0
        prog = (self.epoch - self.warm) / max(1, self.total_steps - self.warm)
        return self.max_lr * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))


__all__ = ["OneCycleLR"]
