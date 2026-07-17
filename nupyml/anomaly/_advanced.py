"""Anomaly detection expansion: projection, ensemble, streaming, and linear
detectors, plus automatic thresholding.

These join the ``detectors`` module (HBOS/ECOD/COPOD/KNN/CBLOF/ABOD) and follow
the same contract: ``fit(X)``, ``decision_function(X)`` (higher = more
anomalous), and ``predict(X)`` (1 inlier / -1 outlier at the ``contamination``
threshold).
"""
import numpy as np

from ..base import BaseEstimator, clone
from ..utils import check_array, check_random_state
from .detectors import _Detector


class LODA(_Detector):
    """Lightweight On-line Detector of Anomalies (Pevný, 2016).

    THE IDEA
    --------
    A single 1-D histogram is a weak density model, but an ENSEMBLE of histograms
    on many random 1-D PROJECTIONS is a strong one -- and dazzlingly cheap. Project
    the data onto ``n_projections`` random (sparse) directions, build a histogram
    per projection, and score a point by the AVERAGE negative log-density it gets
    across the projections. A point that lands in a low-density bin on many
    projections is anomalous.

    WHY SPARSE PROJECTIONS
    ----------------------
    Each projection vector is mostly zeros (a random subset of features with
    random weights), which makes the projections cheap AND diverse, and lets the
    per-projection contributions double as a crude feature-importance for WHY a
    point is anomalous. No distances, trivially streamable, embarrassingly cheap.
    """

    def __init__(self, n_projections=100, n_bins=10, contamination=0.1,
                 random_state=None):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        n_nonzero = max(1, int(np.sqrt(d)))          # sparse projections
        self.projections_, self.edges_, self.logdens_ = [], [], []
        for _ in range(self.n_projections):
            w = np.zeros(d)
            idx = rng.choice(d, n_nonzero, replace=False)
            w[idx] = rng.randn(n_nonzero)
            proj = X @ w
            hist, edges = np.histogram(proj, bins=self.n_bins, density=True)
            self.projections_.append(w)
            self.edges_.append(edges)
            self.logdens_.append(np.log(hist + 1e-12))
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        score = np.zeros(len(X))
        for w, edges, logd in zip(self.projections_, self.edges_, self.logdens_):
            proj = X @ w
            b = np.clip(np.searchsorted(edges, proj) - 1, 0, len(logd) - 1)
            score += -logd[b]                        # negative log-density
        return score / self.n_projections


class FeatureBaggingDetector(_Detector):
    """Ensemble a base detector over random FEATURE subsets (Lazarevic, 2005).

    In high dimensions the anomaly signal often lives in a few features and is
    drowned out by the rest (the curse of dimensionality for distance-based
    detectors). Feature bagging fits the base detector on many random feature
    SUBSPACES and averages the (rank-normalised) scores, so a subspace where the
    anomaly stands out can carry the vote even when the full-space detector is
    blind to it. The subspace analogue of bagging.
    """

    def __init__(self, base_estimator=None, n_estimators=10,
                 max_features=0.5, contamination=0.1, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        from .detectors import KNN
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        k = max(1, int(self.max_features * d))
        base = self.base_estimator if self.base_estimator is not None else KNN()
        self.subsets_, self.detectors_ = [], []
        for _ in range(self.n_estimators):
            cols = rng.choice(d, k, replace=False)
            det = clone(base).fit(X[:, cols])
            self.subsets_.append(cols)
            self.detectors_.append(det)
        self._set_threshold(self.decision_function(X))
        return self

    @staticmethod
    def _rank(s):
        order = s.argsort().argsort()                # 0..n-1 ranks
        return order / (len(s) - 1 + 1e-12)

    def decision_function(self, X):
        X = check_array(X)
        total = np.zeros(len(X))
        for cols, det in zip(self.subsets_, self.detectors_):
            total += self._rank(det.decision_function(X[:, cols]))
        return total / self.n_estimators


class HalfSpaceTrees(_Detector):
    """Half-space trees: isolation by random axis-aligned splits, scored by MASS.

    A half-space tree splits space by picking a random feature and a random
    threshold within its range, recursively, to a fixed depth -- with NO reference
    to the data (the structure is random). Each training point falls into a leaf;
    a leaf's MASS is how many training points share it. A test point landing in a
    low-mass leaf sits where little training data lived -- an anomaly. Averaging
    the (depth-weighted) mass over many random trees gives the score.

    Like isolation forest it needs no distances and scales linearly, and its
    data-independent structure is what makes the streaming version (updating leaf
    masses as data flows) trivial.

    Tan, Ting & Liu (2011).
    """

    def __init__(self, n_estimators=25, max_depth=8, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.contamination = contamination
        self.random_state = random_state

    def _build(self, lo, hi, depth, rng):
        if depth >= self.max_depth:
            return {"leaf": True, "mass": 0}
        j = rng.randint(len(lo))
        split = rng.uniform(lo[j], hi[j])
        left_hi = hi.copy(); left_hi[j] = split
        right_lo = lo.copy(); right_lo[j] = split
        return {"leaf": False, "feature": j, "split": split,
                "left": self._build(lo, left_hi, depth + 1, rng),
                "right": self._build(right_lo, hi, depth + 1, rng)}

    def _fill(self, node, X):
        if node["leaf"]:
            node["mass"] = len(X)
            return
        left = X[:, node["feature"]] <= node["split"]
        self._fill(node["left"], X[left])
        self._fill(node["right"], X[~left])

    def _leaf_mass(self, node, x):
        if node["leaf"]:
            return node["mass"]
        side = "left" if x[node["feature"]] <= node["split"] else "right"
        return self._leaf_mass(node[side], x)

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        # randomise the workspace a little beyond the data range (HST convention)
        span = X.max(axis=0) - X.min(axis=0) + 1e-9
        lo = X.min(axis=0) - 0.2 * span
        hi = X.max(axis=0) + 0.2 * span
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = self._build(lo, hi, 0, rng)
            self._fill(tree, X)
            self.trees_.append(tree)
        self._n_train = len(X)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        scores = np.empty(len(X))
        for i, x in enumerate(X):
            mass = np.mean([self._leaf_mass(t, x) for t in self.trees_])
            scores[i] = -mass                        # low mass -> anomalous
        return scores


class MahalanobisDetector(_Detector):
    """Distance to the centre in the metric of the data's covariance.

    The oldest multivariate outlier score: the Mahalanobis distance
    ``sqrt((x-mu)' S^{-1} (x-mu))`` measures how many "standard deviations" a point
    is from the mean AFTER accounting for feature correlations -- so it flags
    points that are unusual relative to the ellipse the data actually occupies,
    not just far in raw units. Exactly right for a single roughly-Gaussian blob;
    blind to structure a mixture would need.
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.precision_ = np.linalg.inv(cov)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        Xc = check_array(X) - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", Xc, self.precision_, Xc))


class PCAReconstructionDetector(_Detector):
    """Anomaly = how badly PCA (fit on the data) RECONSTRUCTS the point.

    Fit PCA keeping the top components; a normal point lives near that principal
    subspace and is reconstructed with tiny error, while an anomaly -- off the
    manifold the bulk of the data spans -- is reconstructed poorly. The
    reconstruction error IS the score. The linear cousin of an autoencoder anomaly
    detector, and a strong baseline when the inliers really do lie near a
    low-dimensional subspace.
    """

    def __init__(self, n_components=2, contamination=0.1):
        self.n_components = n_components
        self.contamination = contamination

    def fit(self, X):
        from ..decomposition import PCA
        X = check_array(X)
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        recon = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.sqrt(np.sum((X - recon) ** 2, axis=1))


# --- automatic thresholding -----------------------------------------------

def threshold_iqr(scores, k=1.5):
    """Tukey's rule: outliers are scores above Q3 + k*IQR."""
    scores = np.asarray(scores)
    q1, q3 = np.percentile(scores, [25, 75])
    return scores > q3 + k * (q3 - q1)


def threshold_mad(scores, k=3.0):
    """Robust z-score via the median absolute deviation: |s - median| / MAD > k."""
    scores = np.asarray(scores)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-12
    return (scores - med) / (1.4826 * mad) > k       # 1.4826 -> std-consistent


def threshold_gesd(scores, alpha=0.05, max_outliers=None):
    """Generalized ESD test: iteratively remove the most extreme score while a
    Grubbs-style statistic exceeds its critical value (a principled, parametric
    cut when scores are roughly normal). Returns a boolean outlier mask."""
    from scipy import stats
    scores = np.asarray(scores, float)
    n = len(scores)
    max_outliers = max_outliers or max(1, n // 10)
    mask = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    active = list(idx)
    for i in range(1, max_outliers + 1):
        s = scores[active]
        mean, std = s.mean(), s.std(ddof=1) + 1e-12
        R = np.abs(s - mean) / std
        j = int(np.argmax(R))
        n_i = len(active)
        p = 1 - alpha / (2 * n_i)
        t = stats.t.ppf(p, n_i - 2)
        crit = (n_i - 1) * t / np.sqrt((n_i - 2 + t ** 2) * n_i)
        if R[j] > crit:
            mask[active[j]] = True
            active.pop(j)
        else:
            break
    return mask


__all__ = ["LODA", "FeatureBaggingDetector", "HalfSpaceTrees",
           "MahalanobisDetector", "PCAReconstructionDetector",
           "threshold_iqr", "threshold_mad", "threshold_gesd"]
