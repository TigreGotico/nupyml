"""More optimizers: the ideas that came after Adam.

Adam (in ``optim.py``) is the workhorse, but it is not the end of the story.
Each optimizer here is one specific improvement on it, and reading them as a
sequence of fixes is more useful than memorising update rules.
"""
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


class Nadam(Optimizer):
    """Adam with Nesterov momentum: look ahead before you step.

    Plain momentum steps along the accumulated velocity. NESTEROV momentum peeks
    one step ahead -- it evaluates the gradient AT the point the velocity is about
    to carry it to -- which lets it correct an overshoot before committing. Nadam
    folds that lookahead into Adam's first moment. The gain over Adam is usually
    small but real, especially early in training; it is Adam plus a touch of
    foresight.

    Dozat (2016).
    """

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self._t)
            v_hat = v / (1 - b2 ** self._t)
            # the Nesterov term blends the corrected momentum with the current
            # gradient, so the effective direction already anticipates the step
            m_nesterov = b1 * m_hat + (1 - b1) * g / (1 - b1 ** self._t)
            p.data -= self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)


class RAdam(Optimizer):
    """Rectified Adam: fix Adam's unreliable, high-variance early steps.

    THE PROBLEM
    -----------
    Adam's adaptive term (dividing by ``sqrt(v)``) is computed from very few
    samples in the first steps, so its VARIANCE is huge -- the early updates are
    erratic, which is why Adam so often needs a manual learning-rate WARMUP to not
    blow up at the start.

    THE FIX
    -------
    RAdam estimates when the variance of the adaptive term is trustworthy and
    TURNS THE ADAPTIVITY OFF until then, falling back to plain momentum (SGD-like)
    for the first few steps and switching adaptivity on once enough gradient
    history exists. It is, in effect, an automatic, principled warmup -- the
    rectification term is derived, not tuned -- so RAdam often trains stably with
    no warmup schedule at all.

    Liu et al. (2019).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.betas
        # rho_inf and rho_t track how many effective samples the variance estimate
        # has -- the machinery that decides when adaptivity is safe
        rho_inf = 2 / (1 - b2) - 1
        rho_t = rho_inf - 2 * self._t * b2 ** self._t / (1 - b2 ** self._t)
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self._t)
            if rho_t > 4:
                # variance is trustworthy: use the adaptive step, with the
                # derived rectification factor r
                v_hat = np.sqrt(v / (1 - b2 ** self._t))
                r = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf)
                            / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                p.data -= self.lr * r * m_hat / (v_hat + self.eps)
            else:
                # too early to trust the adaptivity: plain momentum, the automatic
                # warmup
                p.data -= self.lr * m_hat


class Lion:
    """Lion: a sign-based optimizer, discovered by program SEARCH.

    THE SURPRISE
    ------------
    Found by symbolic program search over optimizer space, Lion updates using only
    the SIGN of an interpolated momentum::

        update = sign(beta1 * momentum + (1 - beta1) * gradient)
        w -= lr * (update + weight_decay * w)

    Because the step is ``sign(...)``, every parameter moves by the SAME magnitude
    -- only the direction varies. That makes it memory-light (one momentum buffer,
    no second moment like Adam's ``v``) and it matches or beats Adam on large
    models at lower memory cost. It is also a striking example of the field's
    recent turn: an optimizer nobody designed, found by a machine, that works.

    The sign means the learning rate must be much SMALLER than Adam's (the step is
    never scaled down by a small gradient), and weight decay matters more.

    Chen et al. (2023).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        b1, b2 = self.betas
        for p, m in zip(self.params, self._m):
            if p.grad is None:
                continue
            g = p.grad
            # step direction is the SIGN of an interpolation of momentum and grad
            update = np.sign(b1 * m + (1 - b1) * g)
            p.data -= self.lr * (update + self.weight_decay * p.data)
            # the momentum buffer uses the OTHER interpolation -- Lion decouples
            # the direction from the memory update
            m *= b2
            m += (1 - b2) * g


class Lookahead:
    """Wrap any optimizer: let it explore, then pull back toward a slow average.

    THE IDEA
    --------
    Lookahead is a META-optimizer. It keeps a "fast" inner optimizer (Adam, SGD,
    anything) and a "slow" copy of the weights. The fast optimizer takes ``k``
    normal steps, exploring ahead; then the slow weights move a FRACTION of the
    way toward where the fast ones landed, and the fast weights are reset to the
    slow ones. Explore ``k`` steps, commit a little, repeat.

    The slow average damps the oscillations that fast optimizers make in sharp or
    noisy loss landscapes, giving more stable convergence and less sensitivity to
    the inner optimizer's hyperparameters -- often better final accuracy for
    almost no extra cost. It composes with any optimizer, which is the elegant
    part: it improves the one you already use rather than replacing it.

    Zhang et al. (2019).
    """

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step = 0
        # the slow weights: a snapshot the fast optimizer is periodically pulled
        # back toward
        self._slow = [p.data.copy() for p in self.base.params]

    def zero_grad(self):
        self.base.zero_grad()

    def step(self):
        self.base.step()                        # a fast (inner) step
        self._step += 1
        if self._step % self.k == 0:
            # every k steps, move the slow weights partway to the fast ones and
            # reset the fast ones there -- the "look ahead, then commit" move
            for p, slow in zip(self.base.params, self._slow):
                slow += self.alpha * (p.data - slow)
                p.data[...] = slow


__all__ = ["Adagrad", "Nadam", "RAdam", "Lion", "Lookahead"]
