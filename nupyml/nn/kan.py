"""Kolmogorov-Arnold Network: learnable activations on the EDGES (Liu, 2024)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class KAN(Module):
    """Kolmogorov-Arnold Network: learnable activations on the EDGES (Liu, 2024).

    An MLP puts fixed activations on the NODES and learns linear weights on the
    edges. A KAN inverts this: every edge carries a LEARNABLE univariate function,
    and nodes just sum. By the Kolmogorov-Arnold representation theorem, any
    multivariate function decomposes into sums of univariate ones -- KAN learns
    those univariate pieces directly. Here each edge function is a sum of RBF basis
    functions with learnable coefficients, plus a SiLU base path::

        f_ij(x) = w_base * silu(x) + sum_k c_ijk * rbf_k(x)

    so the network can shape each connection's response rather than picking one
    fixed nonlinearity. One KAN layer maps ``in_dim -> out_dim``.
    """

    def __init__(self, in_dim, out_dim, n_basis=8, grid_range=(-2, 2), rng=None):
        super().__init__()
        from ..utils import check_random_state
        r = check_random_state(rng)
        self.in_dim, self.out_dim, self.n_basis = in_dim, out_dim, n_basis
        self.centers = np.linspace(grid_range[0], grid_range[1], n_basis)
        self.width = (grid_range[1] - grid_range[0]) / n_basis
        # learnable spline coefficients per (out, in, basis) and a base weight
        from .module import Parameter
        self.coef = Parameter(r.randn(out_dim, in_dim, n_basis) * 0.1)
        self.w_base = Parameter(r.randn(out_dim, in_dim) * 0.1)

    def parameters(self):
        return [self.coef, self.w_base]

    def forward(self, x):
        x = Tensor._wrap(x)
        # RBF basis of each input feature: (b, in, n_basis)
        xe = x.reshape(x.shape[0], self.in_dim, 1)
        diff = xe - Tensor(self.centers.reshape(1, 1, -1))
        rbf = (-(diff * diff) * (1.0 / (2 * self.width ** 2))).exp()
        # spline term: coef (out,in,basis) x rbf (b,in,basis), summed -> (b, out)
        spline = (self.coef.reshape(1, self.out_dim, self.in_dim, self.n_basis)
                  * rbf.reshape(x.shape[0], 1, self.in_dim, self.n_basis))
        spline = spline.sum(axis=3).sum(axis=2)        # (b, out)
        base = (x.sigmoid() * x).reshape(x.shape[0], 1, self.in_dim) \
            * self.w_base.reshape(1, self.out_dim, self.in_dim)
        base = base.sum(axis=2)                         # (b, out)
        return spline + base


__all__ = ["KAN"]
