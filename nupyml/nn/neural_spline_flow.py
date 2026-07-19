"""A normalizing flow with expressive SPLINE couplings (Durkan et al., 2019)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


def _rq_spline_tensor(x_np, widths, heights, deriv, left=-3.0, right=3.0):
    """Autograd rational-quadratic spline of a column ``x_np``.

    ``widths``/``heights`` are (B, K) Tensors (softmaxed); ``deriv`` is (B, K+1)
    positive. Returns (transformed column Tensor, log-derivative Tensor). Only the
    discrete bin selection uses numpy; every value flows through the tape.
    """
    B, K = widths.shape
    span = right - left
    pref = np.triu(np.ones((K, K + 1)), 1)[:K, :K + 1]   # strict prefix-sum matrix
    pref = pref.T[:K + 1, :K].T if False else np.array(
        [[1.0 if j < m else 0.0 for m in range(K + 1)] for j in range(K)])
    cw = widths @ Tensor(pref)                           # (B, K+1) cumulative widths
    ch = heights @ Tensor(pref)
    xin = np.clip((x_np - left) / span, 1e-6, 1 - 1e-6)
    cwd = cw.data
    k = np.clip(np.array([np.searchsorted(cwd[i], xin[i], "right") - 1
                          for i in range(B)]), 0, K - 1)
    idx = np.arange(B)
    w = widths[idx, k]; h = heights[idx, k]
    x0 = cw[idx, k]; y0 = ch[idx, k]
    d0 = deriv[idx, k]; d1 = deriv[idx, k + 1]
    s = h / w
    xi = (Tensor(xin) - x0) / w
    omxi = 1.0 - xi
    num = h * (s * xi * xi + d0 * xi * omxi)
    den = s + (d0 + d1 - 2.0 * s) * xi * omxi
    y = y0 + num / den
    dnum = s * s * (d1 * xi * xi + 2.0 * s * xi * omxi + d0 * omxi * omxi)
    logdet = (dnum + 1e-9).log() - 2.0 * (den + 1e-9).log()
    return left + span * y, logdet


class NeuralSplineFlow(Module):
    """A normalizing flow with expressive SPLINE couplings (Durkan et al., 2019).

    Affine coupling layers (RealNVP) can only scale and shift, so many are needed to
    model a complex density. A neural spline flow replaces the affine map with a
    monotonic RATIONAL-QUADRATIC SPLINE whose knot positions and slopes are predicted
    by a network -- a far more flexible per-dimension transform that is still exactly
    invertible with a closed-form log-determinant. Fewer, more expressive layers fit
    sharp, multimodal densities that would take many affine couplings. Operates on
    the second half conditioned on the first, alternating.
    """

    def __init__(self, dim, n_flows=4, n_bins=8, hidden=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim, self.n_bins = dim, n_bins
        self.d = dim // 2
        self.flows = []
        out = (dim - self.d) * (3 * n_bins - 1)
        for _ in range(n_flows):
            self.flows.append([Linear(self.d, hidden, rng=r),
                               Linear(hidden, out, rng=r)])

    def parameters(self):
        return [p for f in self.flows for m in f for p in m.parameters()]

    def _params(self, net, n_dim, j):
        # slice out dim j's spline parameters (each a Tensor)
        base = j * (3 * self.n_bins - 1)
        K = self.n_bins
        widths = F.softmax(net[:, base:base + K], axis=1)
        heights = F.softmax(net[:, base + K:base + 2 * K], axis=1)
        raw_d = net[:, base + 2 * K:base + 3 * K - 1]
        deriv = (1.0 + raw_d.exp()).log() + 1e-3         # softplus, positive
        ones = Tensor(np.ones((deriv.shape[0], 1)))
        deriv = Tensor.concatenate([ones, deriv, ones], axis=1)   # pad slopes at ends
        return widths, heights, deriv

    def log_prob(self, x):
        z = Tensor._wrap(x)
        n = z.shape[0]
        total_logdet = Tensor(np.zeros(n))
        for f in self.flows:
            za, zb = z[:, :self.d], z[:, self.d:]
            net = f[1](f[0](za).relu())
            n_dim = self.dim - self.d
            cols, ld = [], Tensor(np.zeros(n))
            for j in range(n_dim):
                widths, heights, deriv = self._params(net, n_dim, j)
                yj, ldj = _rq_spline_tensor(zb.data[:, j], widths, heights, deriv)
                cols.append(yj.reshape(n, 1)); ld = ld + ldj
            z = Tensor.concatenate([za] + cols, axis=1)
            total_logdet = total_logdet + ld
            rev = np.arange(self.dim - 1, -1, -1)
            z = z[:, rev]                                # alternate the transformed half
        logpz = -0.5 * (z * z).sum(axis=1) - 0.5 * self.dim * np.log(2 * np.pi)
        return logpz + total_logdet

    def nll(self, x):
        return -self.log_prob(x).mean()


__all__ = ["NeuralSplineFlow"]
