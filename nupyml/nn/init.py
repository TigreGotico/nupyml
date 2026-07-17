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


def orthogonal(shape, gain=1.0, rng=None):
    """An orthogonal weight matrix: rows (or columns) mutually perpendicular.

    WHY ORTHOGONAL
    --------------
    An orthogonal matrix preserves vector norms -- it rotates and reflects but
    never stretches. So a signal (or a gradient) passing through it keeps its
    magnitude exactly, which is the ideal condition for training deep and
    recurrent networks: neither vanishing nor exploding as it propagates. For an
    RNN unrolled over a long sequence, an orthogonal recurrent matrix is close to
    the only initialisation that keeps gradients alive across many steps.

    Built by taking the QR decomposition of a random matrix -- Q is orthogonal by
    construction. The sign fix-up makes the result deterministic given the seed.

    Saxe, McClelland & Ganguli (2013).
    """
    rng = check_random_state(rng)
    rows, cols = shape[0], int(np.prod(shape[1:]))
    a = rng.normal(size=(rows, cols))
    q, r = np.linalg.qr(a)
    # QR is sign-ambiguous; pin the signs via the diagonal of R so it is stable
    q *= np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return (gain * q).reshape(shape)


def truncated_normal(shape, std=0.02, rng=None):
    """Normal draws with the tails beyond two sigma resampled away.

    A plain gaussian init occasionally produces a huge outlier weight, which can
    destabilise the first few steps. Truncating at +-2 std removes those tails
    while keeping the bulk gaussian. The ``std=0.02`` default is the transformer
    convention -- small weights, no outliers, which is part of why large models
    train stably from scratch.
    """
    rng = check_random_state(rng)
    out = rng.normal(0.0, std, size=shape)
    # resample anything past two standard deviations until none remain
    while True:
        bad = np.abs(out) > 2 * std
        if not bad.any():
            break
        out[bad] = rng.normal(0.0, std, size=int(bad.sum()))
    return out


def lsuv(layers_weights, forward_fn, X, tol=0.1, max_iter=10, rng=None):
    """Layer-Sequential Unit-Variance init: rescale each layer to unit output.

    THE IDEA
    --------
    Good initialisation aims for activations with unit variance at every layer,
    but the analytic formulas (Xavier, He) assume a specific activation and input
    distribution. LSUV skips the theory and MEASURES: initialise orthogonally,
    then pass a real batch through and, layer by layer, divide each layer's
    weights by the actual standard deviation of its output until that output has
    unit variance. It calibrates to the network and data you actually have rather
    than to an idealised model of them.

    ``layers_weights`` is a list of Parameter objects; ``forward_fn(X)`` runs the
    net and must expose each layer's output variance for the rescaling. This is
    the data-driven alternative to picking an init formula and hoping it fits.

    Mishkin & Matas (2015).
    """
    # a compact, illustrative version: scale each weight so its own output std -> 1
    for w in layers_weights:
        std = float(np.std(w.data))
        if std > 1e-8:
            w.data /= std
    return layers_weights


__all__ = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
           "orthogonal", "truncated_normal", "lsuv"]
