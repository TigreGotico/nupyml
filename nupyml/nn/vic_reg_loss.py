"""Variance-Invariance-Covariance regularisation (Bardes et al., 2022)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class VICRegLoss(Module):
    """Variance-Invariance-Covariance regularisation (Bardes et al., 2022).

    A self-supervised loss for two augmented views that avoids collapse WITHOUT
    negatives or a target network -- purely by three explicit terms on the two
    views' embeddings ``za, zb``:

    * **Invariance** -- MSE between the paired views (they should agree).
    * **Variance** -- a hinge keeping each embedding dimension's std above 1, so
      the representation cannot collapse to a constant.
    * **Covariance** -- push the off-diagonal covariance to zero, so dimensions
      carry non-redundant information.

    The variance term is the anti-collapse mechanism SimCLR needs negatives for.
    """

    def __init__(self, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef

    def _var_cov(self, z):
        n, d = z.shape
        zc = z - z.mean(axis=0)
        std = (zc.var(axis=0) + 1e-4).sqrt()
        var_loss = (1.0 - std).relu().mean()           # hinge: keep std >= 1
        cov = (zc.T @ zc) * (1.0 / (n - 1))
        eye = np.eye(d)
        cov_loss = ((cov * Tensor(1.0 - eye)) ** 2).sum() * (1.0 / d)
        return var_loss, cov_loss

    def forward(self, za, zb):
        za, zb = Tensor._wrap(za), Tensor._wrap(zb)
        inv = ((za - zb) ** 2).mean()
        va, ca = self._var_cov(za)
        vb, cb = self._var_cov(zb)
        return (self.sim_coef * inv + self.var_coef * (va + vb)
                + self.cov_coef * (ca + cb))


__all__ = ["VICRegLoss"]
