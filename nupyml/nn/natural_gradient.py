"""Precondition the gradient by the (diagonal) FISHER information."""
import numpy as np
from .optim import Optimizer


class NaturalGradient(Optimizer):
    """Precondition the gradient by the (diagonal) FISHER information.

    Plain gradient descent measures distance in raw PARAMETER space, so its steps
    depend on arbitrary parameterisation. The natural gradient measures distance in
    DISTRIBUTION space (how much the model's output changes) via the Fisher
    information matrix ``F``, and steps as ``F^{-1} grad`` -- the steepest descent
    that is invariant to reparameterisation. Here ``F`` is approximated by the
    running average of the squared gradients (the empirical diagonal Fisher) with a
    damping term, giving a cheap curvature-aware step.
    """

    def __init__(self, params, lr=0.1, damping=1e-3, decay=0.95):
        super().__init__(params, lr)
        self.damping = damping
        self.decay = decay
        self._fisher = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, f in zip(self.params, self._fisher):
            if p.grad is None:
                continue
            f *= self.decay
            f += (1 - self.decay) * p.grad ** 2        # empirical diagonal Fisher
            p.data -= self.lr * p.grad / (f + self.damping)


__all__ = ["NaturalGradient"]
