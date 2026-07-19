"""Precondition each gradient by the geometry of its OWN history (Gupta, 2018)."""
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


__all__ = ["Shampoo"]
