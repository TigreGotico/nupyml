"""Divide the logits by a single learned temperature ``T`` (Guo et al., 2017)."""
import numpy as np
from ..base import BaseEstimator


class TemperatureScaling(BaseEstimator):
    """Divide the logits by a single learned temperature ``T`` (Guo et al., 2017).

    Modern networks are systematically OVER-confident. Temperature scaling is the
    minimal fix: rescale ALL logits by one scalar ``T > 1`` before the softmax,
    which softens every probability without changing the argmax -- so ACCURACY is
    untouched and only the confidences move. ``T`` is fit by minimising NLL on a
    validation set. One parameter, no accuracy cost, and it fixes most of the
    miscalibration -- which is why it is the default post-hoc calibrator.

    ``fit`` takes validation LOGITS and labels; ``transform`` returns calibrated
    probabilities.
    """

    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def _nll(self, logits, y, T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    def fit(self, logits, y):
        logits = np.asarray(logits, float)
        y = np.asarray(y)
        # 1-D golden-section-ish search over T in [0.05, 10]
        Ts = np.linspace(0.05, 10.0, 400)
        nlls = [self._nll(logits, y, T) for T in Ts]
        self.temperature_ = float(Ts[int(np.argmin(nlls))])
        return self

    def transform(self, logits):
        z = np.asarray(logits, float) / self.temperature_
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


__all__ = ["TemperatureScaling"]
