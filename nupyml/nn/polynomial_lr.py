"""Decay the LR to zero along a polynomial of the training fraction."""
from .optim import Optimizer, LRScheduler


class PolynomialLR(LRScheduler):
    """Decay the LR to zero along a polynomial of the training fraction.

    ``lr = base_lr * (1 - epoch/total)^power``. ``power=1`` is linear decay
    (common for transformers); higher powers hold the rate up longer then drop.
    """

    def __init__(self, optimizer, total_iters, power=1.0):
        super().__init__(optimizer)
        self.total_iters = total_iters
        self.power = power

    def get_lr(self):
        frac = min(self.epoch, self.total_iters) / self.total_iters
        return self.base_lr * (1 - frac) ** self.power


__all__ = ["PolynomialLR"]
