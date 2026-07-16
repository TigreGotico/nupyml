"""Optimizers and learning-rate schedulers."""
import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self._velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += g
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=False):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.betas
        bc1 = 1 - b1 ** self._t
        bc2 = 1 - b2 ** self._t
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.weight_decay and self.decoupled:
                update = update + self.weight_decay * p.data
            p.data -= self.lr * update


class AdamW(Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self._sq = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, sq in zip(self.params, self._sq):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            sq *= self.alpha
            sq += (1 - self.alpha) * g * g
            p.data -= self.lr * g / (np.sqrt(sq) + self.eps)


def clip_grad_norm(params, max_norm):
    params = [p for p in params if p.grad is not None]
    total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
    if total > max_norm and total > 0:
        scale = max_norm / total
        for p in params:
            p.grad *= scale
    return total


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        t = min(self.epoch, self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * t / self.T_max))


class WarmupLR(LRScheduler):
    """Linear warmup then inverse-sqrt decay (transformer-style)."""

    def __init__(self, optimizer, warmup_steps):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps

    def get_lr(self):
        step = max(self.epoch, 1)
        return self.base_lr * min(step / self.warmup_steps,
                                  np.sqrt(self.warmup_steps / step))


__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "RMSprop", "clip_grad_norm",
           "LRScheduler", "StepLR", "CosineAnnealingLR", "WarmupLR"]
