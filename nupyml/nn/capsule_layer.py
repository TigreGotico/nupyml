"""Vector neurons routed by AGREEMENT, not by pooling (Sabour et al., 2017)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter


def _squash(s, axis=-1, eps=1e-8):
    """Squash a capsule vector to length in [0, 1) keeping its direction."""
    sq = (s * s).sum(axis=axis, keepdims=True)
    scale = sq / (1.0 + sq)
    return s * (scale / (sq ** 0.5 + eps))


def _softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class CapsuleLayer(Module):
    """Vector neurons routed by AGREEMENT, not by pooling (Sabour et al., 2017).

    A scalar neuron loses all pose information -- max-pooling throws away WHERE a
    feature was. A capsule is a VECTOR whose length encodes the probability a
    feature is present and whose direction encodes its pose. Each lower capsule
    predicts each higher capsule via a learned transform, and DYNAMIC ROUTING
    iteratively raises the coupling to whichever higher capsule its prediction
    agrees with (measured by dot product) -- "routing by agreement" replaces
    pooling, so a part is assigned to the whole it actually votes for. Output
    lengths are class probabilities. Input ``(batch, n_in, d_in)``.
    """

    def __init__(self, n_in, d_in, n_out, d_out, routing_iters=3, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.n_in, self.n_out = n_in, n_out
        self.d_out = d_out
        self.routing_iters = routing_iters
        # transform matrix per (input capsule, output capsule)
        self.W = Parameter(r.randn(n_out, n_in, d_out, d_in) * 0.1)

    def parameters(self):
        return [self.W]

    def forward(self, x):
        x = Tensor._wrap(x)
        B = x.shape[0]
        # predictions u_hat[j][i] = W[j,i] @ u_i  -> list of (B, d_out) tensors
        u_hat = [[x[:, i, :] @ self.W[j, i].transpose() for i in range(self.n_in)]
                 for j in range(self.n_out)]
        b = np.zeros((B, self.n_out, self.n_in))       # routing logits (numpy)
        v = None
        for it in range(self.routing_iters):
            c = _softmax_np(b, axis=1)                  # couplings sum over outputs
            outs = []
            for j in range(self.n_out):
                s = None
                for i in range(self.n_in):
                    cij = Tensor(c[:, j, i][:, None])
                    term = u_hat[j][i] * cij
                    s = term if s is None else s + term
                outs.append(_squash(s).reshape(B, 1, self.d_out))
            v = Tensor.concatenate(outs, axis=1)        # (B, n_out, d_out)
            if it < self.routing_iters - 1:             # update agreement (detached)
                for j in range(self.n_out):
                    vj = v.data[:, j, :]
                    for i in range(self.n_in):
                        b[:, j, i] += (u_hat[j][i].data * vj).sum(axis=1)
        return v

    def lengths(self, x):
        v = self.forward(x)
        return ((v * v).sum(axis=-1) + 1e-9) ** 0.5     # capsule "probabilities"


__all__ = ["CapsuleLayer"]
