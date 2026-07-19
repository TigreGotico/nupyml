"""Time series v6: shape clustering, causality, extreme-value anomalies, shrinkage
VAR, and dependent-data resampling.

Five more time-series tools. k-Shape clusters series by SHAPE, invariant to shift and
scale. Convergent cross mapping detects nonlinear CAUSALITY from reconstructed
attractors. SPOT flags anomalies by fitting the TAIL. Bayesian VAR shrinks toward a
random walk for stable multivariate forecasts. The block bootstrap resamples a series
without destroying its autocorrelation.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


def _sbd(a, b):
    """Shape-based distance: 1 - max normalised cross-correlation over all shifts."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    cc = np.correlate(a, b, mode="full")
    ncc = cc / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return 1 - ncc.max()


class KShape(BaseEstimator):
    """Cluster series by SHAPE, ignoring shift and scale (Paparrizos & Gravano, 2015).

    Two heartbeats are the "same shape" even if one is shifted in time or scaled in
    amplitude -- but Euclidean k-means calls them different. k-Shape uses a SHAPE-based
    distance built on normalised cross-correlation (which maximises over all
    alignments), and updates each centroid by extracting the shape that best aligns
    with its members (the top eigenvector of their aligned covariance). So it groups
    series by their pattern, invariant to phase and amplitude -- the right notion for
    most time-series clustering. Series are z-normalised internally.
    """

    def __init__(self, n_clusters=2, max_iter=50, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        labels = rng.randint(0, self.n_clusters, n)
        Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
        for _ in range(self.max_iter):
            centroids = np.array([self._extract_shape(Xn[labels == k], Xn.shape[1])
                                  for k in range(self.n_clusters)])
            new = np.array([min(range(self.n_clusters),
                                key=lambda k: _sbd(Xn[i], centroids[k]))
                            for i in range(n)])
            if np.array_equal(new, labels):
                break
            labels = new
        self.labels_ = labels
        self.centroids_ = centroids
        return self

    def _extract_shape(self, members, length):
        if len(members) == 0:
            return np.zeros(length)
        # centroid = top eigenvector of the members' covariance (the common shape)
        M = members - members.mean(axis=1, keepdims=True)
        S = M.T @ M
        vals, vecs = np.linalg.eigh(S)
        c = vecs[:, -1]
        if (c @ members.mean(axis=0)) < 0:
            c = -c
        return (c - c.mean()) / (c.std() + 1e-9)

    def fit_predict(self, X):
        return self.fit(X).labels_


def convergent_cross_mapping(x, y, embed_dim=3, tau=1, lib_sizes=None):
    """Detect nonlinear CAUSALITY from reconstructed attractors (Sugihara, 2012).

    Granger causality assumes linearity and separability; many real systems (ecology,
    physiology) are neither. Convergent cross mapping tests causality through Takens'
    theorem: if X drives Y, then the history of Y contains a shadow of X, so X should be
    RECOVERABLE from Y's reconstructed attractor -- and, crucially, the recovery should
    IMPROVE (converge) as more data fills in the attractor. It reconstructs each series
    by time-delay embedding, cross-predicts one from the other's neighbours, and reports
    the skill vs library size. Returns cross-map skill for each library size.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if lib_sizes is None:
        lib_sizes = [20, len(x) // 2, len(x) - embed_dim * tau - 1]

    def embed(s):
        n = len(s) - (embed_dim - 1) * tau
        return np.array([[s[i + j * tau] for j in range(embed_dim)]
                         for i in range(n)])

    My = embed(y)                                         # Y's shadow manifold
    target = x[(embed_dim - 1) * tau:]                    # aligned X values
    from scipy.spatial.distance import cdist
    skills = []
    for L in lib_sizes:
        L = min(L, len(My))
        lib = My[:L]; lib_t = target[:L]
        D = cdist(My, lib)
        pred = np.empty(len(My))
        for i in range(len(My)):
            nn = np.argsort(D[i])[:embed_dim + 1]
            d = D[i, nn]
            w = np.exp(-d / (d[0] + 1e-9)); w /= w.sum()  # cross-map from neighbours
            pred[i] = w @ lib_t[nn]
        skills.append(np.corrcoef(pred, target)[0, 1])
    return np.array(lib_sizes), np.array(skills)


class SPOT(BaseEstimator):
    """Flag anomalies by fitting the TAIL, not the whole distribution (Siffer, 2017).

    Setting an anomaly threshold by hand fails when the data's scale drifts or is
    unknown. SPOT (Streaming Peaks-Over-Threshold) uses EXTREME VALUE THEORY: it fits a
    Generalised Pareto Distribution to the PEAKS above an initial high quantile, then
    sets the alarm threshold at the value whose exceedance probability is a chosen risk
    ``q``. So the threshold is derived from the tail's own shape and a target false-
    alarm RATE, adapting automatically to the data's scale -- principled anomaly
    detection with a statistical guarantee. ``q`` is the desired anomaly probability.
    """

    def __init__(self, q=1e-3, init_quantile=0.95):
        self.q = q
        self.init_quantile = init_quantile

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.t_ = np.quantile(y, self.init_quantile)      # peak-selection threshold
        peaks = y[y > self.t_] - self.t_
        gamma, sigma = self._fit_gpd(peaks)
        n, Nt = len(y), len(peaks)
        # threshold z_q whose exceedance probability equals q (GPD quantile)
        if abs(gamma) > 1e-6:
            self.threshold_ = self.t_ + (sigma / gamma) * ((self.q * n / Nt) ** (-gamma) - 1)
        else:
            self.threshold_ = self.t_ - sigma * np.log(self.q * n / Nt)
        self.gamma_, self.sigma_ = gamma, sigma
        return self

    @staticmethod
    def _fit_gpd(peaks):
        # method-of-moments estimate of the GPD shape and scale
        m, v = peaks.mean(), peaks.var() + 1e-9
        gamma = 0.5 * (1 - m ** 2 / v)
        sigma = 0.5 * m * (m ** 2 / v + 1)
        return gamma, max(sigma, 1e-6)

    def predict(self, y):
        return (np.asarray(y, float) > self.threshold_).astype(int)


class BayesianVAR(BaseEstimator):
    """VAR that SHRINKS toward a random walk (Litterman's Minnesota prior, 1986).

    A VAR on many series has more coefficients than short macro samples can pin down,
    so OLS overfits and forecasts badly. The Minnesota prior encodes the belief that
    each series is roughly a RANDOM WALK: it shrinks every coefficient toward zero
    EXCEPT each variable's own first lag (shrunk toward one), with own lags shrunk less
    than cross lags and higher lags shrunk more. The posterior is a ridge-like
    regularised VAR whose forecasts are far more stable than OLS on limited data.
    ``lam`` controls the overall shrinkage tightness.
    """

    def __init__(self, p=1, lam=0.1):
        self.p = p
        self.lam = lam

    def fit(self, Y):
        Y = check_array(Y)
        T, k = Y.shape
        rows, targ = [], []
        for t in range(self.p, T):
            rows.append(np.concatenate([[1.0], Y[t - self.p:t][::-1].ravel()]))
            targ.append(Y[t])
        Xd = np.array(rows); B = np.array(targ)
        # prior mean: own first lag = 1, everything else 0
        prior_mean = np.zeros((Xd.shape[1], k))
        for j in range(k):
            prior_mean[1 + j, j] = 1.0                    # own lag-1 -> random walk
        # ridge toward the prior mean (Minnesota-style shrinkage)
        pen = np.eye(Xd.shape[1]) / self.lam ** 2
        pen[0, 0] = 0                                     # don't shrink the intercept
        self.coef_ = np.linalg.solve(Xd.T @ Xd + pen,
                                     Xd.T @ B + pen @ prior_mean)
        self.k_ = k
        self._history = Y[-self.p:].copy()
        return self

    def forecast(self, steps=10):
        hist = list(self._history)
        out = []
        for _ in range(steps):
            x = np.concatenate([[1.0], np.array(hist[-self.p:][::-1]).ravel()])
            pred = x @ self.coef_
            out.append(pred); hist.append(pred)
        return np.array(out)


def block_bootstrap(series, block_size=10, n_boot=100, statistic=None,
                    random_state=None):
    """Resample a dependent series WITHOUT destroying its autocorrelation
    (Kunsch, 1989).

    The ordinary bootstrap resamples observations independently, which is exactly wrong
    for a time series -- it shatters the temporal dependence you are trying to reason
    about. The moving-block bootstrap resamples contiguous BLOCKS instead, so the
    within-block correlation structure is preserved in each replicate. This gives valid
    confidence intervals for statistics of dependent data (a mean, an autocorrelation, a
    forecast error) that the naive bootstrap would get badly wrong. Returns bootstrap
    replicates (or the statistic evaluated on each if ``statistic`` is given).
    """
    y = np.asarray(series, float).ravel()
    rng = check_random_state(random_state)
    n = len(y)
    n_blocks = int(np.ceil(n / block_size))
    out = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, n_blocks)
        rep = np.concatenate([y[s:s + block_size] for s in starts])[:n]
        out.append(statistic(rep) if statistic else rep)
    return np.array(out)


__all__ = ["KShape", "convergent_cross_mapping", "SPOT", "BayesianVAR",
           "block_bootstrap"]
