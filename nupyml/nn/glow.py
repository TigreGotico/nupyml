"""A normalizing flow with invertible LINEAR mixing (Kingma & Dhariwal, 2018)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


class Glow(Module):
    """A normalizing flow with invertible LINEAR mixing (Kingma & Dhariwal, 2018).

    RealNVP's coupling layers only ever transform half the dimensions at a time and
    permute by a fixed rule. Glow replaces the fixed permutation with a LEARNED
    invertible linear map (a "1x1 convolution"), so the dimensions get mixed
    optimally between couplings, and adds ``ActNorm`` (a data-dependent affine
    normalisation) for stable training. Each layer contributes an exact log-
    determinant, so the model has a tractable likelihood and both samples and
    scores densities. Implemented for vector data (the 1x1 conv is a dense
    invertible matrix).
    """

    def __init__(self, dim, n_flows=4, hidden=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim = dim
        self.flows = []
        for _ in range(n_flows):
            W = np.linalg.qr(r.randn(dim, dim))[0]        # start orthogonal (invertible)
            self.flows.append({
                "log_s": Parameter(np.zeros(dim)), "b": Parameter(np.zeros(dim)),
                "W": Parameter(W),
                "nn": [Linear(dim // 2, hidden, rng=r), Linear(hidden, dim - dim // 2, rng=r),
                       Linear(dim // 2, hidden, rng=r), Linear(hidden, dim - dim // 2, rng=r)],
            })

    def parameters(self):
        ps = []
        for f in self.flows:
            ps += [f["log_s"], f["b"], f["W"]]
            for m in f["nn"]:
                ps += list(m.parameters())
        return ps

    def forward(self, x):
        """Map data -> latent, accumulating the log-determinant of the Jacobian."""
        x = Tensor._wrap(x)
        logdet = Tensor(np.zeros(len(x.data)))
        d = self.dim // 2
        for f in self.flows:
            # actnorm
            x = x * f["log_s"].exp() + f["b"]
            logdet = logdet + f["log_s"].sum()
            # invertible linear (1x1 conv)
            x = x @ f["W"]
            sign, ld = np.linalg.slogdet(f["W"].data)
            logdet = logdet + ld
            # affine coupling
            xa, xb = x[:, :d], x[:, d:]
            scale = self.__nn(f["nn"][0], f["nn"][1], xa).tanh()
            shift = self.__nn(f["nn"][2], f["nn"][3], xa)
            xb = xb * scale.exp() + shift
            logdet = logdet + scale.sum(axis=1)
            x = Tensor.concatenate([xa, xb], axis=1)
        return x, logdet

    @staticmethod
    def __nn(l1, l2, x):
        return l2(l1(x).relu())

    def inverse(self, z):
        """Map latent -> data (exact inverse of ``forward``), for sampling."""
        z = np.asarray(Tensor._wrap(z).data, float)
        d = self.dim // 2
        for f in reversed(self.flows):
            za, zb = z[:, :d], z[:, d:]
            scale = np.tanh(self.__nn(f["nn"][0], f["nn"][1], Tensor(za)).data)
            shift = self.__nn(f["nn"][2], f["nn"][3], Tensor(za)).data
            zb = (zb - shift) * np.exp(-scale)            # invert coupling
            z = np.concatenate([za, zb], axis=1)
            z = z @ np.linalg.inv(f["W"].data)            # invert 1x1 conv
            z = (z - f["b"].data) * np.exp(-f["log_s"].data)   # invert actnorm
        return z

    def sample(self, n, rng=None):
        rng = check_random_state(rng)
        return self.inverse(rng.normal(size=(n, self.dim)))

    def log_prob(self, x):
        z, logdet = self.forward(x)
        # standard-normal base density
        logpz = -0.5 * ((z * z).sum(axis=1) + self.dim * np.log(2 * np.pi))
        return logpz + logdet

    def nll(self, x):
        return -self.log_prob(x).mean()


__all__ = ["Glow"]
