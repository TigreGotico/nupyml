"""Cap an outlier's influence ENTIRELY once it is far enough (Beaton, 1974)."""
from ..autograd import Tensor
from .module import Module


class TukeyBiweightLoss(Module):
    """Cap an outlier's influence ENTIRELY once it is far enough (Beaton, 1974).

    Squared error lets one outlier dominate; Huber caps its GRADIENT to a constant
    but still lets it pull forever. Tukey's biweight goes further: past a threshold
    ``c`` the residual contributes a CONSTANT loss and ZERO gradient -- a gross
    outlier is not down-weighted, it is switched off. This "redescending" behaviour
    is the most robust of the M-estimators, at the cost of being non-convex (so it
    needs a decent initialisation, e.g. from a Huber fit).
    """

    def __init__(self, c=4.685):
        super().__init__()
        self.c = c

    def forward(self, pred, target):
        r = Tensor._wrap(pred) - Tensor._wrap(target)
        c2 = self.c ** 2
        # rho(r) = (c^2/6)[1 - (1-(r/c)^2)^3] for |r|<=c, else c^2/6
        inside = (1.0 - (r / self.c) * (r / self.c)).relu()   # 0 outside |r|<=c
        rho = (c2 / 6.0) * (1.0 - inside * inside * inside)
        return rho.mean()


__all__ = ["TukeyBiweightLoss"]
