"""The multivariate CRPS -- a proper score for VECTOR forecasts (Gneiting, 2008)."""
import numpy as np
from ..autograd import Tensor


def _norm(v, axis):
    return ((v * v).sum(axis=axis) + 1e-12) ** 0.5


def energy_score(forecast_samples, y):
    """The multivariate CRPS -- a proper score for VECTOR forecasts (Gneiting, 2008).

    CRPS scores a scalar predictive distribution; the energy score generalises it to
    vectors, so it can judge a probabilistic forecast of several correlated quantities
    at once (a wind field, a portfolio of returns). In its ensemble form it is
    ``mean||X_i - y|| - 0.5 mean||X_i - X_j||`` -- reward for the ensemble being close
    to the outcome, penalty for being spread out -- and it is PROPER, minimised in
    expectation only by the true joint distribution, so a model cannot cheat by
    misstating correlations. Fully differentiable. ``forecast_samples`` is
    ``(batch, n_samples, dim)``; ``y`` is ``(batch, dim)``.
    """
    X = Tensor._wrap(forecast_samples)
    B, n, d = X.shape
    y = Tensor(np.asarray(y, float)).reshape(B, 1, d)
    term1 = _norm(X - y, axis=2).mean(axis=1)             # E||X - y||
    xi = X.reshape(B, n, 1, d)
    xj = X.reshape(B, 1, n, d)
    term2 = _norm(xi - xj, axis=3).mean(axis=2).mean(axis=1)   # E||X - X'||
    return (term1 - 0.5 * term2).mean()


__all__ = ["energy_score"]
