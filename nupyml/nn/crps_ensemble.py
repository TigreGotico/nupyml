"""A PROPER score for probabilistic forecasts (Gneiting & Raftery, 2007)."""
import numpy as np
from ..autograd import Tensor


def _abs(t):
    return t.relu() + (t * (-1.0)).relu()


def crps_ensemble(forecast_samples, y):
    """A PROPER score for probabilistic forecasts (Gneiting & Raftery, 2007).

    A point forecast cannot be rewarded for honest uncertainty. The Continuous
    Ranked Probability Score judges a whole predictive DISTRIBUTION against the
    single observed value, and it is PROPER: it is minimised in expectation only by
    the true distribution, so a model cannot game it by lying about its confidence.
    In the ensemble form used here it is
    ``mean|x_i - y| - 0.5 * mean|x_i - x_j|`` -- reward for being close to the
    outcome, penalty for being over-dispersed -- and it is fully differentiable, so
    it trains probabilistic (ensemble/sample) forecasters directly. ``forecast_samples``
    is ``(batch, n_samples)``; ``y`` is ``(batch,)``.
    """
    x = Tensor._wrap(forecast_samples)
    y = Tensor(np.asarray(y, float).reshape(-1, 1))
    n = x.shape[1]
    term1 = _abs(x - y).mean(axis=1)                      # E|X - y|
    xi = x.reshape(x.shape[0], n, 1)
    xj = x.reshape(x.shape[0], 1, n)
    term2 = _abs(xi - xj).mean(axis=2).mean(axis=1)       # E|X - X'|
    return (term1 - 0.5 * term2).mean()


__all__ = ["crps_ensemble"]
