"""Optimizer zoo v2: adaptive-method refinements, gradient surgery, and
natural-gradient preconditioning.

These join the base optimizers (SGD/Adam/AdamW/RMSprop/Adagrad/Nadam/RAdam/Lion/
Lookahead/Adadelta/AMSGrad/LAMB). Each targets a specific failure of Adam.
"""
import numpy as np

from .optim import Optimizer


class AdaBelief(Optimizer):
    """Adam that adapts to the BELIEF in the gradient (Zhuang et al., 2020).

    Adam divides by the second moment of the gradient (its size). AdaBelief divides
    by the variance of the gradient AROUND its running mean -- ``(g - m)^2`` instead
    of ``g^2``. So a steadily-pointing gradient (small deviation from ``m``) gets a
    LARGE step even when its magnitude is large, and a noisy, direction-flipping
    one gets a small step. It reads the CONFIDENCE in the direction, not just the
    magnitude -- yielding faster convergence and better generalisation than Adam
    with a one-line change.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._s = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, s in zip(self.params, self._m, self._s):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            diff = g - m                              # deviation from the belief
            s *= self.b2; s += (1 - self.b2) * diff ** 2
            mhat = m / (1 - self.b1 ** self._t)
            shat = s / (1 - self.b2 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(shat) + self.eps)


class Adamax(Optimizer):
    """Adam with the INFINITY norm for the second moment (Kingma & Ba, 2015).

    Adam's ``v`` is an L2 (squared) average of gradients. Adamax replaces it with a
    running MAX of the gradient magnitude (an L∞ norm), which is more stable when
    gradients are sparse or occasionally huge -- the max is not blown up by one
    outlier the way a squared average is. A simple, robust Adam variant.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._u = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, u in zip(self.params, self._m, self._u):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            np.maximum(self.b2 * u, np.abs(g), out=u)   # L-infinity running max
            mhat = m / (1 - self.b1 ** self._t)
            p.data -= self.lr * mhat / (u + self.eps)


class Yogi(Optimizer):
    """Control Adam's second-moment GROWTH additively (Zaheer et al., 2018).

    Adam's ``v`` can grow (or shrink) fast, causing the effective learning rate to
    swing and sometimes stopping convergence. Yogi changes the ``v`` update from
    multiplicative to ADDITIVE, gated by a sign::

        v <- v - (1 - b2) * sign(v - g^2) * g^2

    so ``v`` only ever changes by a controlled amount per step -- it cannot spike
    on one large gradient. Same cost as Adam, steadier adaptation.
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.full_like(p.data, 1e-6) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            g2 = g ** 2
            v -= (1 - self.b2) * np.sign(v - g2) * g2   # additive, sign-gated
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(np.abs(vhat)) + self.eps)


class AdaBound(Optimizer):
    """Adam that TRANSITIONS into SGD via dynamic learning-rate bounds
    (Luo et al., 2019).

    Adam converges fast but generalises worse than SGD; the culprit is a handful
    of extreme per-parameter learning rates late in training. AdaBound clips each
    parameter's effective step into a ``[lower, upper]`` band that starts wide
    (pure Adam) and, over training, both bounds converge to the SGD rate. So it is
    Adam early (fast) and SGD late (generalises) -- a smooth handover, no switch
    point to tune.
    """

    def __init__(self, params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999),
                 gamma=1e-3, eps=1e-8):
        super().__init__(params, lr)
        self.final_lr = final_lr
        self.b1, self.b2 = betas
        self.gamma = gamma
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        # the band shrinks toward final_lr as t grows
        lower = self.final_lr * (1 - 1 / (self.gamma * self._t + 1))
        upper = self.final_lr * (1 + 1 / (self.gamma * self._t))
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            step = self.lr / (np.sqrt(vhat) + self.eps)
            step = np.clip(step, lower, upper)         # bound each per-param step
            p.data -= step * mhat


class Adafactor(Optimizer):
    """Sublinear-memory Adam by FACTORING the second moment (Shazeer & Stern, 2018).

    Adam stores a full second-moment tensor -- as much memory as the weights
    themselves, painful for huge matrices. Adafactor stores only per-ROW and
    per-COLUMN averages of a 2-D weight's squared gradients and reconstructs the
    estimate as their outer product, cutting the second-moment memory from
    ``O(m*n)`` to ``O(m + n)``. This factored approximation is what let large
    transformers train under tight memory. Falls back to full moment for vectors.
    """

    def __init__(self, params, lr=1e-2, beta2=0.999, eps=1e-30):
        super().__init__(params, lr)
        self.beta2 = beta2
        self.eps = eps
        self._state = []
        for p in self.params:
            if p.data.ndim == 2:
                m, n = p.data.shape
                self._state.append({"r": np.zeros(m), "c": np.zeros(n)})
            else:
                self._state.append({"v": np.zeros_like(p.data)})

    def step(self):
        for p, st in zip(self.params, self._state):
            if p.grad is None:
                continue
            g2 = p.grad ** 2 + self.eps
            if "v" in st:
                st["v"] = self.beta2 * st["v"] + (1 - self.beta2) * g2
                denom = np.sqrt(st["v"])
            else:
                st["r"] = self.beta2 * st["r"] + (1 - self.beta2) * g2.mean(axis=1)
                st["c"] = self.beta2 * st["c"] + (1 - self.beta2) * g2.mean(axis=0)
                # rank-1 reconstruction of the second moment from row/col averages
                est = np.outer(st["r"], st["c"]) / (st["r"].mean() + self.eps)
                denom = np.sqrt(est)
            p.data -= self.lr * p.grad / (denom + 1e-8)


def pcgrad(task_grads):
    """PCGrad: project away CONFLICTING task gradients (Yu et al., 2020).

    In multi-task learning, gradients from different tasks can point in opposing
    directions -- one task's update undoes another's. PCGrad detects a conflict (a
    negative dot product between two task gradients) and PROJECTS one gradient onto
    the normal plane of the other, removing only the conflicting component. The
    de-conflicted gradients are then summed. This stops tasks from fighting and is
    the standard fix for negative transfer.

    ``task_grads`` is a list of flat gradient vectors (one per task); returns the
    combined gradient.
    """
    import copy
    grads = [np.array(g, float) for g in task_grads]
    n = len(grads)
    proj = [g.copy() for g in grads]
    rng_order = list(range(n))
    for i in range(n):
        for j in rng_order:
            if i == j:
                continue
            dot = proj[i] @ grads[j]
            if dot < 0:                                # conflict -> project out
                proj[i] -= dot / (grads[j] @ grads[j] + 1e-12) * grads[j]
    return np.sum(proj, axis=0)


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


__all__ = ["AdaBelief", "Adamax", "Yogi", "AdaBound", "Adafactor", "pcgrad",
           "NaturalGradient"]
