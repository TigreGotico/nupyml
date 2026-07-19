"""Sharpness-Aware Minimisation -- seek FLAT minima (Foret et al., 2021)."""
import numpy as np


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


__all__ = ["SAM"]
