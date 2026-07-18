"""Time-series v2: intermittent demand, the Theta method, SSA, SAX, DBA, and
seasonal features.

These join ARIMA/Holt-Winters/STL/Kalman (forecasting/filtering) and the
``tsclass`` classifiers. Each targets a case the classics handle poorly:
intermittent demand, robust extrapolation, subspace decomposition, symbolic
representation, elastic averaging, and calendar seasonality.
"""
import numpy as np

from ..base import BaseEstimator


class Croston(BaseEstimator):
    """Forecasting for INTERMITTENT demand (Croston, 1972).

    Exponential smoothing applied to a mostly-zero series chases the occasional
    spike and decays wrongly between them. Croston's insight: smooth the demand
    SIZES and the INTERVALS between demands SEPARATELY, and forecast the rate as
    ``size / interval``. So a part sold in bursts of 4 every ~5 periods forecasts a
    steady 0.8/period, not a decaying echo of the last spike. The standard method
    for spare-parts and slow-moving inventory.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float)
        nz = np.where(y > 0)[0]
        if len(nz) == 0:
            self.rate_ = 0.0
            return self
        sizes = y[nz]
        intervals = np.diff(np.concatenate([[-1], nz]))   # gaps between demands
        z = sizes[0]; x = intervals[0]
        for s, iv in zip(sizes[1:], intervals[1:]):
            z += self.alpha * (s - z)                      # smooth demand size
            x += self.alpha * (iv - x)                     # smooth interval
        self.rate_ = z / x                                 # demand per period
        return self

    def predict(self, horizon):
        return np.full(horizon, self.rate_)

    forecast = predict


class Theta(BaseEstimator):
    """The Theta method -- robust extrapolation that won the M3 competition
    (Assimakopoulos & Nikolopoulos, 2000).

    Decompose the series into "theta lines" -- versions with the local curvature
    scaled by ``theta``. ``theta=0`` is the linear regression trend (long-term
    direction); ``theta=2`` doubles the curvature (short-term structure). Forecast
    each and combine (the classic recipe averages the ``theta=0`` line with a
    simple-exponential-smoothing forecast of the ``theta=2`` line). This simple
    blend of long-term trend and short-term smoothing was, astonishingly, the most
    accurate method in the M3 forecasting competition.
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        t = np.arange(n)
        # linear trend (the theta=0 line)
        A = np.column_stack([np.ones(n), t])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        # SES level of the theta=2 line (2*y - trend)
        theta2 = 2 * y - (A @ self.coef_)
        level = theta2[0]
        for v in theta2[1:]:
            level += self.alpha * (v - level)
        self.level_ = level
        self.n_ = n
        return self

    def predict(self, horizon):
        t = np.arange(self.n_, self.n_ + horizon)
        trend = self.coef_[0] + self.coef_[1] * t
        return 0.5 * trend + 0.5 * self.level_         # blend trend + SES level

    forecast = predict


class SSA(BaseEstimator):
    """Singular Spectrum Analysis: decompose a series by SVD of its trajectory
    matrix.

    SSA is model-free spectral decomposition. Slide a window over the series to
    build a TRAJECTORY MATRIX (each column a lagged snapshot); its SVD yields
    components that, grouped, separate TREND, oscillations, and noise -- without
    assuming a parametric model. Reconstruct a component by diagonal-averaging its
    rank-1 piece back into a series. Widely used to extract trend/seasonality and
    to denoise. ``n_components`` sets how many leading components to keep.
    """

    def __init__(self, window=None, n_components=2):
        self.window = window
        self.n_components = n_components

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        L = self.window or n // 3
        K = n - L + 1
        X = np.array([y[i:i + L] for i in range(K)]).T   # trajectory matrix (L x K)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = []
        for k in range(min(self.n_components, len(s))):
            Xk = s[k] * np.outer(U[:, k], Vt[k])
            self.components_.append(self._diag_avg(Xk, n))
        self.reconstruction_ = np.sum(self.components_, axis=0)
        return self

    @staticmethod
    def _diag_avg(X, n):
        """Hankelise: average each anti-diagonal back into a 1-D series."""
        L, K = X.shape
        out = np.zeros(n)
        cnt = np.zeros(n)
        for i in range(L):
            for j in range(K):
                out[i + j] += X[i, j]; cnt[i + j] += 1
        return out / cnt


def sax(y, n_segments=8, alphabet_size=4):
    """Symbolic Aggregate approXimation: turn a series into a short STRING
    (Lin et al., 2003).

    SAX compresses a series to symbols in two steps: PAA averages it into
    ``n_segments`` equal pieces (dimensionality reduction), then each piece's value
    is mapped to a letter by Gaussian BREAKPOINTS that split a standard normal into
    equal-probability bins (after z-normalising the series). The result is a tiny
    string that supports fast indexing, motif discovery, and a distance that lower-
    bounds the true one -- the representation behind much of symbolic time-series
    mining. Returns the symbol list.
    """
    from scipy.stats import norm
    y = np.asarray(y, float)
    y = (y - y.mean()) / (y.std() + 1e-12)             # z-normalise
    seg = np.array_split(y, n_segments)
    paa = np.array([s.mean() for s in seg])            # piecewise aggregate approx
    breaks = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    idx = np.searchsorted(breaks, paa)
    return [chr(ord("a") + i) for i in idx]


def dtw_barycenter_averaging(series, n_iter=10):
    """DBA: the average of a set of series under DYNAMIC TIME WARPING (Petitjean, 2011).

    Averaging time series pointwise blurs features that occur at slightly different
    TIMES (two heartbeats a few samples apart average into a smear). DBA instead
    averages under DTW: align every series to a candidate average by warping, then
    update each average point to the mean of the points aligned to it, and iterate.
    The result preserves the shared SHAPE rather than smearing it -- the correct
    centroid for DTW-based clustering (k-means with DTW). Returns the barycenter
    series.
    """
    from ..sequence import dtw_path
    series = [np.asarray(s, float) for s in series]
    avg = series[len(series) // 2].copy()              # init from a medoid-ish member
    for _ in range(n_iter):
        assoc = [[] for _ in range(len(avg))]
        for s in series:
            path, _ = dtw_path(avg, s)
            for i, j in path:
                assoc[i].append(s[j])                  # points aligned to avg[i]
        avg = np.array([np.mean(a) if a else avg[k] for k, a in enumerate(assoc)])
    return avg


def fourier_features(length, period, n_harmonics=3):
    """Sine/cosine SEASONAL features for a given period (a Fourier basis).

    Model smooth seasonality by regressing on a few Fourier harmonics of the
    seasonal period instead of one dummy per season: for period ``p`` and harmonic
    ``h``, the pair ``sin(2*pi*h*t/p), cos(2*pi*h*t/p)``. A handful of harmonics
    captures a smooth yearly/weekly cycle with far fewer parameters than seasonal
    dummies, and it extrapolates. Returns a ``(length, 2*n_harmonics)`` design
    matrix to feed any regressor.
    """
    t = np.arange(length)
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * t / period))
        cols.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(cols)


__all__ = ["Croston", "Theta", "SSA", "sax", "dtw_barycenter_averaging",
           "fourier_features"]
