"""Optimizers v3: preconditioning and layer-wise adaptive scaling.

These target two ideas the Adam family does not cover: SECOND-ORDER structure
(Shampoo preconditions with the gradient's covariance, not just its diagonal) and
LAYER-WISE trust (LARS/LARC/NovoGrad scale each layer's step by the ratio of its
weight size to its gradient size, which is what makes very large batches trainable).
Adan adds a Nesterov-style look-ahead on top of Adam's two moments.
"""
import numpy as np

from .optim import Optimizer


def _matrix_power_neg_quarter(A, eps=1e-6):
    """A^(-1/4) for a symmetric PSD matrix, via its eigendecomposition."""
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * w ** -0.25) @ V.T


class Shampoo(Optimizer):
    """Precondition each gradient by the geometry of its OWN history (Gupta, 2018).

    Adam scales each weight independently (a diagonal preconditioner). Shampoo keeps
    the full second-moment structure but factorises it PER TENSOR DIMENSION: for a
    weight matrix ``G`` it accumulates a left covariance ``L += G Gᵀ`` and a right
    covariance ``R += Gᵀ G``, then steps along ``L^{-1/4} G R^{-1/4}``. That is a
    cheap stand-in for the full-matrix (Kronecker) preconditioner -- it captures how
    gradients correlate ACROSS rows and ACROSS columns without ever forming the
    giant full Hessian. On 1-D parameters it degrades gracefully to Adagrad.
    """

    def __init__(self, params, lr=0.01, eps=1e-4, update_freq=1, weight_decay=0.0):
        super().__init__(params, lr)
        self.eps = eps
        self.update_freq = update_freq
        self.weight_decay = weight_decay
        self._pre = []            # per-param preconditioner state
        self._t = 0
        for p in self.params:
            if p.data.ndim == 2:
                m, n = p.data.shape
                self._pre.append({"L": np.eye(m) * eps, "R": np.eye(n) * eps,
                                  "Li": np.eye(m), "Ri": np.eye(n)})
            else:                 # diagonal (Adagrad) fallback
                self._pre.append({"acc": np.zeros_like(p.data) + eps})

    def step(self):
        self._t += 1
        for p, st in zip(self.params, self._pre):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if "acc" in st:
                st["acc"] += g * g
                p.data -= self.lr * g / np.sqrt(st["acc"])
                continue
            st["L"] += g @ g.T
            st["R"] += g.T @ g
            if self._t % self.update_freq == 0:      # inverse roots are the cost
                st["Li"] = _matrix_power_neg_quarter(st["L"], self.eps)
                st["Ri"] = _matrix_power_neg_quarter(st["R"], self.eps)
            p.data -= self.lr * (st["Li"] @ g @ st["Ri"])


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling -- one trust ratio per layer (You, 2017).

    When you scale batch size up to thousands, a single global learning rate is
    wrong for every layer at once: early layers have tiny weights and huge
    gradients, late layers the reverse. LARS sets each layer's effective rate from
    its own TRUST RATIO ``||w|| / ||g + wd·w||`` -- so the step is always a fixed
    fraction of the weight's magnitude, regardless of how big or small that layer's
    gradients happen to be. This is what let ResNet train with batch size 32k.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-4,
                 eps=1e-8, clip=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps
        self.clip = clip                             # True -> LARC
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self._v):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            w_norm = np.linalg.norm(p.data)
            g_norm = np.linalg.norm(g)
            if w_norm > 0 and g_norm > 0:
                trust = w_norm / (g_norm + self.eps)
                # LARC clips the local rate so it never exceeds the global lr
                local_lr = min(trust, 1.0) if self.clip else trust
            else:
                local_lr = 1.0
            v *= self.momentum
            v += local_lr * g
            p.data -= self.lr * v


class LARC(LARS):
    """LARS with the trust ratio CLIPPED to 1 (Ginsburg, 2018).

    Plain LARS can hand a layer a local rate far above the global one early in
    training, which occasionally blows up. LARC clips the trust ratio at 1 so a
    layer's step is never larger than the global learning rate would give -- it can
    only ever slow a layer DOWN, never speed it past the global budget. Safer, and
    the usual default in practice.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-4, eps=1e-8):
        super().__init__(params, lr, momentum, weight_decay, eps, clip=True)


class NovoGrad(Optimizer):
    """Adam's adaptivity with ONE second-moment number per layer (Ginsburg, 2019).

    Adam stores a second moment for every weight -- as much memory as the model
    itself. NovoGrad keeps a single SCALAR second moment per layer (the running
    norm of that layer's gradient), normalises the gradient by it, then applies
    momentum and DECOUPLED weight decay. The result is layer-wise adaptive like
    LARS but with Adam-style smoothing, at a fraction of Adam's memory -- and it
    tends to be more robust to the initial learning rate.
    """

    def __init__(self, params, lr=0.01, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [0.0 for _ in self.params]
        self._init = [False for _ in self.params]

    def step(self):
        b1, b2 = self.betas
        for i, (p, m) in enumerate(zip(self.params, self._m)):
            if p.grad is None:
                continue
            g = p.grad
            gnorm2 = float(np.sum(g * g))            # scalar second moment
            if not self._init[i]:
                self._v[i] = gnorm2
                self._init[i] = True
            else:
                self._v[i] = b2 * self._v[i] + (1 - b2) * gnorm2
            gn = g / (np.sqrt(self._v[i]) + self.eps)
            gn = gn + self.weight_decay * p.data     # decoupled decay
            m *= b1
            m += (1 - b1) * gn
            p.data -= self.lr * m


class Adan(Optimizer):
    """Adaptive Nesterov Momentum -- look-ahead on the gradient DIFFERENCE
    (Xie, 2022).

    Adam smooths the gradient and its square. Adan additionally smooths the
    gradient's CHANGE between steps (``g_t - g_{t-1}``), a discrete stand-in for
    the Nesterov look-ahead, and folds it into both the update direction and the
    second-moment scaling. In practice it reaches a given loss in noticeably fewer
    steps than AdamW across vision and language models, with the same decoupled
    weight decay.
    """

    def __init__(self, params, lr=0.01, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]  # of grad diffs
        self._n = [np.zeros_like(p.data) for p in self.params]  # second moment
        self._prev_g = [None for _ in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2, b3 = self.betas
        for i, (p, m, v, n) in enumerate(zip(self.params, self._m, self._v,
                                             self._n)):
            if p.grad is None:
                continue
            g = p.grad
            prev = self._prev_g[i]
            diff = g - prev if prev is not None else np.zeros_like(g)
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * diff
            comb = g + (1 - b2) * diff
            n *= b3; n += (1 - b3) * comb * comb
            bc1 = 1 - b1 ** self._t
            bc2 = 1 - b2 ** self._t
            bc3 = 1 - b3 ** self._t
            update = (m / bc1 + (1 - b2) * v / bc2) / (np.sqrt(n / bc3) + self.eps)
            p.data -= self.lr * update
            p.data /= (1 + self.lr * self.weight_decay)   # decoupled decay
            self._prev_g[i] = g.copy()


__all__ = ["Shampoo", "LARS", "LARC", "NovoGrad", "Adan"]
