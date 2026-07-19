"""ONE robust loss with a tunable shape (Barron, 2019)."""
from ..autograd import Tensor
from .module import Module


class BarronLoss(Module):
    """ONE robust loss with a tunable shape (Barron, 2019).

    L2, Charbonnier, Cauchy, Geman-McClure, Welsch -- the robust losses are usually
    presented as separate choices, and picking one is guesswork. Barron's general
    loss is a single family with a continuous shape parameter ``alpha`` that
    RECOVERS all of them: ``alpha=2`` is squared error, ``alpha=1`` is a smooth L1
    (Charbonnier), ``alpha=0`` is Cauchy, ``alpha -> -inf`` is Welsch. Lower alpha
    means an outlier's influence saturates sooner. Because alpha is just a number,
    it can even be LEARNED, letting the model adapt its own robustness. ``c`` is the
    scale at which residuals start to be treated as outliers.
    """

    def __init__(self, alpha=1.0, c=1.0):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, pred, target):
        r = (Tensor._wrap(pred) - Tensor._wrap(target)) * (1.0 / self.c)
        a = self.alpha
        if a == 2.0:
            rho = 0.5 * (r * r)
        elif a == 0.0:                                   # Cauchy / Lorentzian
            rho = (0.5 * (r * r) + 1.0).log()
        else:
            b = abs(a - 2.0)
            rho = (b / a) * (((r * r) / b + 1.0) ** (a / 2.0) - 1.0)
        return rho.mean()


__all__ = ["BarronLoss"]
