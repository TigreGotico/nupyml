"""Weight initialization: why the starting point decides whether training works.

Initialise every weight to zero and every unit in a layer computes the same
thing, receives the same gradient, and stays identical forever -- the layer has
one effective unit no matter how wide it is. Symmetry must be broken randomly.

But the SCALE of that randomness decides everything. Each layer multiplies the
signal's variance by roughly ``fan_in * var(w)``. If that factor is below 1, the
signal shrinks geometrically with depth until it vanishes; above 1, it explodes.
Gradients suffer the same fate on the way back. This is why deep networks were
considered untrainable before the schemes below.

The fix is to choose ``var(w)`` so the factor is ~1:

* ``xavier_*`` (Glorot) targets ``2 / (fan_in + fan_out)`` -- a compromise
  keeping variance stable in both directions. Derived for symmetric activations
  like tanh.
* ``kaiming_*`` (He) targets ``2 / fan_in``. The extra factor of 2 accounts for
  ReLU zeroing half its inputs and therefore halving the variance. Use this with
  ReLU.

``fan_in`` is the number of inputs feeding a unit; ``fan_out`` the number it
feeds. For a conv layer both include the receptive field, since each output
draws from ``in_channels * kh * kw`` values.
"""
import numpy as np

from ..utils import check_random_state


def _fans(shape):
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


def xavier_uniform(shape, rng=None, gain=1.0):
    rng = check_random_state(rng)
    fan_in, fan_out = _fans(shape)
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape)


def xavier_normal(shape, rng=None, gain=1.0):
    rng = check_random_state(rng)
    fan_in, fan_out = _fans(shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=shape)


def kaiming_uniform(shape, rng=None):
    rng = check_random_state(rng)
    fan_in, _ = _fans(shape)
    bound = np.sqrt(6.0 / fan_in)
    return rng.uniform(-bound, bound, size=shape)


def kaiming_normal(shape, rng=None):
    rng = check_random_state(rng)
    fan_in, _ = _fans(shape)
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


__all__ = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
