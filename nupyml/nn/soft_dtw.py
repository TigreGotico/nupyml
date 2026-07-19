"""A DIFFERENTIABLE dynamic time warping distance (Cuturi & Blondel, 2017)."""
from ..autograd import Tensor


def _soft_min(a, b, c, gamma):
    # soft-min(v) = -gamma * log( sum_i exp(-v_i / gamma) )
    stacked = Tensor.stack([a.reshape(()), b.reshape(()), c.reshape(())])
    return -gamma * ((stacked * (-1.0 / gamma)).exp().sum().log())


def soft_dtw(x, y, gamma=1.0):
    """A DIFFERENTIABLE dynamic time warping distance (Cuturi & Blondel, 2017).

    DTW aligns two sequences by the best warping path, but the ``min`` in its
    recursion has no gradient, so you cannot train through it. Soft-DTW replaces
    that hard ``min`` with a SOFT one (``-gamma·logsumexp``), giving a smooth,
    differentiable measure of alignment cost -- so it can be a loss for forecasting
    or a metric for clustering time series of unequal length or phase. ``gamma``
    controls the softness (``->0`` recovers exact DTW). Returns a scalar Tensor,
    differentiable w.r.t. both sequences.
    """
    x = Tensor._wrap(x); y = Tensor._wrap(y)
    xd = x.data.ravel(); yd = y.data.ravel()
    n, m = len(xd), len(yd)
    INF = 1e10
    R = [[None] * (m + 1) for _ in range(n + 1)]
    R[0][0] = Tensor(0.0)
    for i in range(1, n + 1):
        R[i][0] = Tensor(INF)
    for j in range(1, m + 1):
        R[0][j] = Tensor(INF)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (x[i - 1:i] - y[j - 1:j])
            cost = (d * d).reshape(())
            R[i][j] = cost + _soft_min(R[i - 1][j], R[i][j - 1],
                                       R[i - 1][j - 1], gamma)
    return R[n][m]


__all__ = ["soft_dtw"]
