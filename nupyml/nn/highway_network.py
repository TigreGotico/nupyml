"""A layer that LEARNS how much to transform vs pass through (Srivastava, 2015)."""
from ..autograd import Tensor
from .module import Module, Parameter
from .layers import Linear, ReLU


class HighwayNetwork(Module):
    """A layer that LEARNS how much to transform vs pass through (Srivastava, 2015).

    Stacking many plain layers makes gradients vanish. A highway layer adds a
    learned GATE::

        y = T(x) * H(x) + (1 - T(x)) * x

    where ``H`` is a normal transform and ``T`` (a sigmoid "transform gate")
    decides, per unit, how much to transform versus carry the input straight
    through. When ``T=0`` the layer is the identity, so information (and gradient)
    flows unimpeded -- the idea that let networks go very deep, and the direct
    ancestor of the residual connection and the LSTM gate. Requires equal in/out
    dimension.
    """

    def __init__(self, dim, rng=None):
        super().__init__()
        self.H = Linear(dim, dim, rng=rng)
        self.T = Linear(dim, dim, rng=rng)
        # bias the gate toward CARRYING at init (negative gate bias) so deep stacks
        # start near the identity
        self.T.bias.data[...] = -4.0

    def parameters(self):
        return list(self.H.parameters()) + list(self.T.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        t = self.T(x).sigmoid()
        return t * self.H(x).relu() + (1.0 - t) * x


__all__ = ["HighwayNetwork"]
