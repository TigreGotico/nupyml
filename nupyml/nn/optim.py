"""Optimizers: what to do with a gradient once you have it.

The gradient says which way is downhill RIGHT HERE. It does not say how far to
step, and it is measured at a point you are about to leave. Every optimizer
below is a different answer to those two problems, and they form a fairly
straight line of reasoning:

``SGD``
    Step against the gradient. Simple, and the step size is entirely your
    problem: too large diverges, too small crawls, and the right value differs
    per problem and per epoch.

``SGD(momentum=...)``
    Accumulate a velocity instead of stepping fresh each time. In a ravine --
    steep across, shallow along -- plain SGD bounces between the walls and
    barely advances. Momentum cancels the oscillation (opposite gradients
    subtract) and accumulates the consistent direction (aligned gradients add).

``RMSprop``
    Notices that one global step size cannot suit every parameter: a rarely-
    active feature's weight needs bigger steps than a constantly-active one's.
    So divide each parameter's step by a running RMS of its own recent
    gradients -- a per-parameter learning rate, adapted automatically.

``Adam``
    Both at once: momentum for the direction, RMSprop-style scaling for the
    size, plus a bias correction (see below). The default for good reason.

``AdamW``
    Fixes how Adam handles weight decay -- see its docstring; the distinction
    is subtle and matters.

None of them escape a local minimum by design; they are all local methods. What
saves neural networks is that in high dimensions most critical points are
saddles rather than minima, and noise plus momentum carries you off a saddle.
"""
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
    """Stochastic gradient descent, optionally with momentum.

    Plain: ``w -= lr * g``.

    With momentum, keep a decaying running sum of past gradients and step along
    THAT::

        v = momentum * v + g
        w -= lr * v

    Read ``momentum=0.9`` as "remember roughly the last 10 gradients"
    (``1/(1-0.9)``). Consistent directions reinforce; oscillating ones cancel.
    It also carries you across flat regions where the gradient nearly vanishes.

    Nesterov's variant evaluates the gradient at where the momentum is ABOUT to
    take you, rather than where you are -- a look-ahead that lets it start
    braking before overshooting, rather than after.
    """

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
    """Adaptive Moment Estimation: momentum and per-parameter scaling together.

    Track two running averages per parameter -- the mean of the gradient
    (``m``, first moment) and the mean of its square (``v``, second moment) --
    and step::

        w -= lr * m_hat / (sqrt(v_hat) + eps)

    The division is the adaptive part: a parameter whose gradients are
    consistently large gets a large ``v`` and so a SMALLER step, while a
    parameter with rare, small gradients takes relatively larger ones. The step
    is roughly scale-invariant, which is why Adam's default ``lr`` works across
    wildly different problems where SGD's would not.

    THE BIAS CORRECTION
    -------------------
    ``m`` and ``v`` start at zero, so early on they are biased toward zero --
    with ``beta2=0.999``, ``v`` after one step is a thousandth of the true
    value, and dividing by its square root would give an enormous first step.
    Dividing by ``1 - beta^t`` corrects this exactly; the factor decays to 1 and
    the correction quietly disappears.

    ``eps`` only prevents division by zero for a parameter that has seen no
    gradient at all.
    """

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
    """Adam with weight decay DECOUPLED from the adaptive scaling.

    L2 regularisation and weight decay are the same thing under plain SGD, and
    people used them interchangeably for years. Under Adam they are not.

    Adding ``alpha*w`` to the GRADIENT (L2) means the penalty passes through the
    ``1/sqrt(v)`` scaling like everything else. A parameter with large gradients
    gets a large ``v``, which SHRINKS its effective decay -- so the weights that
    most need regularising are regularised least. Exactly backwards.

    AdamW instead subtracts ``lr * alpha * w`` from the weight directly, after
    the adaptive step, so every parameter decays at the same rate regardless of
    its gradient history. This is why AdamW is standard for transformers.
    """

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
    """Rescale all gradients so their combined norm never exceeds ``max_norm``.

    A single freak batch, or a recurrent net's exploding gradient, can produce
    one enormous step that destroys weights it took hours to learn. Clipping
    caps the step SIZE while preserving its DIRECTION -- the whole gradient
    vector is scaled by one factor, so the relative contributions are untouched.

    Essentially mandatory for RNNs, where repeatedly multiplying by the same
    recurrent matrix makes gradients grow exponentially in sequence length.
    """
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
    """Decay the learning rate along a cosine curve from ``lr`` to ``eta_min``.

    Large steps early explore; small steps late settle into a minimum instead of
    bouncing around it. The cosine shape spends longer at both the high and low
    ends than a straight line would, with a smooth transition between -- which
    empirically beats step decay, and has no cliff for training to lurch at.
    """

    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        t = min(self.epoch, self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * t / self.T_max))


class WarmupLR(LRScheduler):
    """Ramp the learning rate UP, then decay it as ``1/sqrt(step)``.

    Warmup looks backwards until you consider Adam's state: at step 1 the
    second-moment estimate is based on a single gradient, so its per-parameter
    scaling is essentially noise. Taking full-sized steps on that basis can wreck
    a model in the first few batches. Starting small gives the moment estimates
    time to become meaningful.

    It matters most where the gradients are least stationary early on -- which
    is exactly why transformer training treats warmup as mandatory rather than
    optional.
    """

    def __init__(self, optimizer, warmup_steps):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps

    def get_lr(self):
        step = max(self.epoch, 1)
        return self.base_lr * min(step / self.warmup_steps,
                                  np.sqrt(self.warmup_steps / step))


__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "RMSprop", "clip_grad_norm",
           "LRScheduler", "StepLR", "CosineAnnealingLR", "WarmupLR"]
