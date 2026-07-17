"""Modern activation functions, as a progression of fixes.

WHAT AN ACTIVATION IS FOR
-------------------------
Stack linear layers and you get one linear layer -- composition of linear maps is
linear, so depth buys nothing. The non-linearity BETWEEN layers is the entire
reason a deep network can represent more than a shallow one. Which non-linearity
turns out to matter, and the history here is a sequence of specific problems and
their fixes.

THE STORY IN ONE PARAGRAPH
--------------------------
``ReLU`` (elsewhere in ``layers.py``) fixed the vanishing gradient of sigmoid/tanh
and made deep nets trainable -- but it has DEAD units (a neuron stuck at negative
input outputs zero forever, gradient zero, never recovers). ``LeakyReLU`` and
``ELU`` keep a little signal alive on the negative side to fix that. ``SELU``
makes the activation SELF-NORMALISING. ``SiLU``/``GELU`` smooth the kink, which
helps optimisation and turns out to work better in transformers. ``GLU`` and
``SwiGLU`` add a MULTIPLICATIVE gate, and SwiGLU is what most current large models
actually use. Each is a targeted repair of the last one's flaw.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd.functional import softmax
from .module import Module, Parameter


class SiLU(Module):
    """SiLU / Swish: ``x * sigmoid(x)``. A smooth ReLU that dips below zero.

    ReLU's hard corner at zero is non-differentiable and kills anything negative
    outright. SiLU is smooth everywhere and lets slightly-negative inputs through
    with a small negative output, so units do not die and the gradient never
    abruptly vanishes. Found by neural-architecture SEARCH (hence the original
    name Swish) and now standard in vision and language models -- a case where the
    machine-discovered function beat the hand-designed ones.
    """

    def forward(self, x):
        x = Tensor._wrap(x)
        return x * x.sigmoid()


class Mish(Module):
    """Mish: ``x * tanh(softplus(x))``. Smoother still than SiLU.

    Same idea as SiLU -- smooth, non-monotonic, self-gated -- with an even gentler
    curve and a slightly larger negative region. It sometimes edges out SiLU on
    vision tasks; the differences between the modern smooth activations are small
    and empirical, which is itself worth knowing: past ReLU, the choice is fine
    tuning, not a step change.
    """

    def forward(self, x):
        x = Tensor._wrap(x)
        softplus = ((x.exp() + 1.0).log())
        return x * softplus.tanh()


class ELU(Module):
    """Exponential Linear Unit: linear for positive x, ``alpha*(e^x - 1)`` below.

    ReLU's negative outputs are all exactly zero, so the mean activation is pushed
    positive -- a "bias shift" that slows learning. ELU saturates to ``-alpha``
    for very negative input, letting negatives contribute and pulling the mean
    back toward zero, which acts like a mild built-in normalisation. The smooth
    saturation also keeps the gradient alive where ReLU's is dead.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = Tensor._wrap(x)
        # x where positive, alpha*(e^x - 1) where negative -- selected on the tape
        neg = (x.exp() - 1.0) * self.alpha
        mask = (x.data > 0).astype(np.float64)
        return x * Tensor(mask) + neg * Tensor(1.0 - mask)


class SELU(Module):
    """Scaled ELU: an activation that keeps activations normalised BY ITSELF.

    THE SELF-NORMALISING PROPERTY
    -----------------------------
    The two constants are not tuned -- they are DERIVED so that, given
    zero-mean/unit-variance inputs, the outputs also have zero mean and unit
    variance, and this is a fixed point the network converges TO across layers. So
    a deep SELU network keeps its activations normalised with no BatchNorm at all
    -- the activation does the job that a normalisation layer usually does.

    The catch is that the guarantee depends on the specific constants, on inputs
    already being standardised, and on a matching ``AlphaDropout``. It is a
    beautiful idea that never displaced BatchNorm in practice, but it shows that
    normalisation can live in the activation itself.

    Klambauer et al. (2017).
    """

    _ALPHA = 1.6732632423543772
    _SCALE = 1.0507009873554805

    def forward(self, x):
        x = Tensor._wrap(x)
        neg = (x.exp() - 1.0) * self._ALPHA
        mask = (x.data > 0).astype(np.float64)
        return (x * Tensor(mask) + neg * Tensor(1.0 - mask)) * self._SCALE


class Softplus(Module):
    """``log(1 + e^x)``: a smooth ReLU whose output is strictly positive.

    The smooth approximation to ReLU, and useful precisely where you need a
    positive output with a gradient everywhere -- predicting a variance, a rate, a
    scale that must not go negative. It is the integral of the sigmoid, which is
    why its derivative IS the sigmoid.
    """

    def forward(self, x):
        x = Tensor._wrap(x)
        # via the stable identity max(x,0) + log(1 + e^-|x|) to avoid overflow
        return x.relu() + ((-x.abs()).exp() + 1.0).log()


class GLU(Module):
    """Gated Linear Unit: split the input, and let one half GATE the other.

    ``GLU(x) = a * sigmoid(b)`` where ``a, b`` are the two halves of the input.

    The multiplicative gate is the new idea: instead of a fixed non-linearity
    applied elementwise, the network LEARNS, per position, how much of each signal
    to let through -- ``sigmoid(b)`` is a data-dependent valve on ``a``. This
    gating is strictly more expressive than a pointwise activation, and it is why
    gated units keep reappearing (in LSTMs, in convolutional language models, and
    in the transformer feed-forward block).

    Dauphin et al. (2017).
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = Tensor._wrap(x)
        half = x.shape[self.dim] // 2
        # split into value and gate; the gate is squashed to (0,1) and multiplies
        a = x[..., :half] if self.dim == -1 else x
        b = x[..., half:] if self.dim == -1 else x
        return a * b.sigmoid()


class SwiGLU(Module):
    """GLU with a SiLU gate and its own weights -- the modern transformer FFN.

    THE UNIT MOST CURRENT LLMS USE
    ------------------------------
    Replace the GLU's sigmoid gate with SiLU, and give the block its own three
    weight matrices::

        SwiGLU(x) = (x W1) * SiLU(x W2),  then project by W3

    Two projections of the input, one gating the other through SiLU, then a
    down-projection. It consistently beats the plain ReLU feed-forward block that
    transformers originally used, by a margin large enough that PaLM, LLaMA and
    most recent models adopted it. The lesson of the whole file lands here: a
    learned multiplicative gate with a smooth activation is the current best answer
    for the non-linearity between attention layers.

    Shazeer (2020).
    """

    def __init__(self, dim, hidden_dim=None, rng=None):
        super().__init__()
        from .layers import Linear
        hidden_dim = hidden_dim or int(dim * 8 / 3)   # the LLaMA sizing rule
        self.w1 = Linear(dim, hidden_dim, bias=False, rng=rng)   # value
        self.w2 = Linear(dim, hidden_dim, bias=False, rng=rng)   # gate
        self.w3 = Linear(hidden_dim, dim, bias=False, rng=rng)   # down-projection
        self._silu = SiLU()

    def forward(self, x):
        x = Tensor._wrap(x)
        return self.w3(self.w1(x) * self._silu(self.w2(x)))


__all__ = ["SiLU", "Mish", "ELU", "SELU", "Softplus", "GLU", "SwiGLU"]
