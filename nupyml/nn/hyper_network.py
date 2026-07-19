"""A network that GENERATES another network's weights (Ha et al., 2017)."""
from ..autograd import Tensor
from .module import Module, Parameter
from .layers import Linear, ReLU


class HyperNetwork(Module):
    """A network that GENERATES another network's weights (Ha et al., 2017).

    Instead of learning a target layer's weights directly, learn a small
    HYPERNETWORK that OUTPUTS them from a context/embedding. This shares parameters
    across many target layers (one hypernet generates all of them), enables
    conditioning (different context -> different weights, e.g. per-task or
    per-timestep), and can compress a large model into a small generator. Here the
    hypernet maps a context vector to the flattened weight matrix of a linear
    target layer, which is then applied to the input.
    """

    def __init__(self, context_dim, in_dim, out_dim, hidden=16, rng=None):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.gen = Linear(context_dim, in_dim * out_dim, rng=rng)
        self.ctx_hidden = Linear(context_dim, hidden, rng=rng)

    def parameters(self):
        return list(self.gen.parameters())

    def generate_weight(self, context):
        w = self.gen(Tensor._wrap(context))              # (batch, in*out)
        return w.reshape(-1, self.in_dim, self.out_dim)

    def forward(self, context, x):
        W = self.generate_weight(context)                # per-sample weight matrix
        x = Tensor._wrap(x)
        # apply each sample's generated weight to its input
        return (x.reshape(x.shape[0], 1, self.in_dim) @ W).reshape(x.shape[0], self.out_dim)


__all__ = ["HyperNetwork"]
