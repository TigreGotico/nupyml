"""Channel attention: recalibrate features by their global importance"""
from ..autograd import Tensor
from .module import Module, Parameter
from .layers import Linear, ReLU


class SqueezeExcitation(Module):
    """Channel attention: recalibrate features by their global importance
    (Hu et al., 2018).

    A network treats every feature channel equally, but some are far more useful
    for a given input. Squeeze-and-excitation SQUEEZES the feature vector to a
    per-channel summary, learns EXCITATION weights from it through a tiny
    bottleneck MLP (sigmoid-gated), and rescales each channel by its weight -- a
    cheap self-attention over channels that consistently improves accuracy. Here it
    gates a feature vector; in a CNN the squeeze is a global average pool.
    """

    def __init__(self, channels, reduction=4, rng=None):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = Linear(channels, hidden, rng=rng)
        self.fc2 = Linear(hidden, channels, rng=rng)

    def parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        gate = self.fc2(self.fc1(x).relu()).sigmoid()   # per-channel importance
        return x * gate


__all__ = ["SqueezeExcitation"]
