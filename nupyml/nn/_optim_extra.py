"""Additional optimizers, weight-averaging wrappers, and LR schedules.

These join the base ``optim`` module (SGD/Adam/AdamW/RMSprop/Adagrad/Nadam/RAdam/
Lion/Lookahead, Step/Cosine/Warmup schedules) with the gaps: Adadelta, AMSGrad,
LAMB; sharpness-aware minimisation, stochastic weight averaging, weight EMA; and
one-cycle / cyclical / polynomial / exponential schedules.
"""
import numpy as np

from .optim import Optimizer, LRScheduler


class Adadelta(Optimizer):
    """Adagrad's fix that needs NO learning rate (Zeiler, 2012).

    Adagrad's accumulator grows without bound, so its step decays to zero.
    Adadelta uses a DECAYING average of squared gradients (like RMSprop) AND a
    decaying average of squared UPDATES, forming the step from their ratio::

        step = -(sqrt(E[dx^2] + eps) / sqrt(E[g^2] + eps)) * g

    The two running averages have the same units, so they cancel -- which is why
    Adadelta has no ``lr`` to tune at all. It self-scales the step from the recent
    history of both gradients and updates.
    """

    def __init__(self, params, rho=0.95, eps=1e-6):
        super().__init__(params, lr=1.0)
        self.rho = rho
        self.eps = eps
        self._Eg = [np.zeros_like(p.data) for p in self.params]
        self._Edx = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, Eg, Edx in zip(self.params, self._Eg, self._Edx):
            if p.grad is None:
                continue
            g = p.grad
            Eg *= self.rho; Eg += (1 - self.rho) * g ** 2
            dx = -np.sqrt(Edx + self.eps) / np.sqrt(Eg + self.eps) * g
            Edx *= self.rho; Edx += (1 - self.rho) * dx ** 2
            p.data += dx


class AMSGrad(Optimizer):
    """Adam with a NON-DECREASING second-moment (Reddi et al., 2018).

    Adam can fail to converge because its ``v`` (second moment) can shrink, letting
    a rare large-gradient direction get a huge step. AMSGrad keeps the running
    MAXIMUM of ``v`` and divides by that, so the effective learning rate is
    monotonically non-increasing per parameter -- the small fix that restores the
    convergence guarantee Adam's original proof was missing.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._vhat = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v, vhat in zip(self.params, self._m, self._v, self._vhat):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            np.maximum(vhat, v, out=vhat)           # the running max -> monotone LR
            mhat = m / (1 - self.b1 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments for large-BATCH training (You et al., 2020).

    Training with very large batches needs large learning rates, which destabilise
    Adam. LAMB fixes this with a per-LAYER TRUST RATIO: it computes the Adam update
    for a parameter tensor, then rescales it by ``||weights|| / ||update||`` so the
    step is a bounded fraction of the weight's own magnitude, regardless of the raw
    gradient scale. That layer-local normalisation is what let BERT train in 76
    minutes -- it keeps every layer's relative step sane at batch sizes where a
    global learning rate cannot.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g ** 2
            mhat = m / (1 - self.b1 ** self._t)
            vhat = v / (1 - self.b2 ** self._t)
            update = mhat / (np.sqrt(vhat) + self.eps) + self.weight_decay * p.data
            w_norm = np.linalg.norm(p.data)
            u_norm = np.linalg.norm(update)
            trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
            p.data -= self.lr * trust * update      # step scaled to the weight's size


class SAM:
    """Sharpness-Aware Minimisation -- seek FLAT minima (Foret et al., 2021).

    A sharp minimum fits the training set but generalises poorly; a flat one is
    robust. SAM optimises the worst-case loss in a neighbourhood: ``first_step``
    climbs to the nearby point of highest loss (``+rho * g/||g||``), then
    ``second_step`` restores the weights and lets the BASE optimizer descend using
    the gradient measured THERE. Two forward/backward passes per update, buying a
    flatter, more generalisable minimum.

    Usage: ``forward+backward; opt.first_step(); forward+backward again;
    opt.second_step()``.
    """

    def __init__(self, base_optimizer, rho=0.05):
        self.base = base_optimizer
        self.params = base_optimizer.params
        self.rho = rho
        self._e = None

    def zero_grad(self):
        self.base.zero_grad()

    def first_step(self):
        grads = [p.grad for p in self.params]
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads if g is not None)) + 1e-12
        self._e = []
        for p in self.params:
            e = self.rho * (p.grad / norm) if p.grad is not None else 0.0
            self._e.append(e)
            p.data += e                             # ascend to the worst-case point

    def second_step(self):
        for p, e in zip(self.params, self._e):      # restore, then descend
            p.data -= e
        self.base.step()


class SWA:
    """Stochastic Weight Averaging -- average the weights along the SGD trajectory.

    Late in training with a cyclic/high learning rate, SGD bounces around the rim
    of a wide flat basin rather than sitting at one point. Averaging those weights
    lands you nearer the basin's CENTRE -- a flatter solution that generalises
    better than any single iterate, for free. Call ``update_parameters()`` at each
    collection point; ``swap_in()`` copies the running average into the model.
    """

    def __init__(self, params):
        self.params = list(params)
        self.averaged = [p.data.copy() for p in self.params]
        self.n = 0

    def update_parameters(self):
        self.n += 1
        for avg, p in zip(self.averaged, self.params):
            avg += (p.data - avg) / self.n          # running mean of the iterates

    def swap_in(self):
        for p, avg in zip(self.params, self.averaged):
            p.data[...] = avg


class EMA:
    """Exponential Moving Average of the weights -- a smoothed "shadow" model.

    Keep a decayed running average of the parameters
    (``shadow = decay*shadow + (1-decay)*weights``). Evaluating with the shadow
    weights is a cheap, standard trick that noticeably improves stability and
    final accuracy (and it is what diffusion / GAN training almost always report).
    """

    def __init__(self, params, decay=0.999):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.data.copy() for p in self.params]

    def update(self):
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s *= d; s += (1 - d) * p.data

    def copy_to(self):
        for p, s in zip(self.params, self.shadow):
            p.data[...] = s


# --- schedules ------------------------------------------------------------

class ExponentialLR(LRScheduler):
    """Multiply the LR by ``gamma`` every epoch -- smooth geometric decay."""

    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** self.epoch


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


__all__ = ["Adadelta", "AMSGrad", "LAMB", "SAM", "SWA", "EMA",
           "ExponentialLR", "PolynomialLR", "CyclicalLR", "OneCycleLR"]
