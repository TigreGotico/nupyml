"""Multiply the LR by ``gamma`` every epoch -- smooth geometric decay."""
from .optim import Optimizer, LRScheduler


class ExponentialLR(LRScheduler):
    """Multiply the LR by ``gamma`` every epoch -- smooth geometric decay."""

    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** self.epoch


__all__ = ["ExponentialLR"]
