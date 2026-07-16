"""Weight initialization schemes."""
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
