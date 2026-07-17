"""Statistical and distance-based anomaly detectors (pyod-style).

Each exposes ``fit(X)`` then ``decision_function(X)`` (higher = more anomalous)
and ``predict(X)`` (1 = inlier, -1 = outlier, by a ``contamination`` threshold),
matching the ``outlier`` module's convention.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from ..base import BaseEstimator
from ..utils import check_array


class _Detector(BaseEstimator):
    """Shared threshold/predict logic once a decision_function exists."""

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > self.threshold_, -1, 1)

    def _set_threshold(self, train_scores):
        # the top `contamination` fraction of training scores are called outliers
        self.threshold_ = np.quantile(train_scores, 1 - self.contamination)


class HBOS(_Detector):
    """Histogram-Based Outlier Score: rare bins, summed across features.

    THE IDEA
    --------
    Build a histogram per feature. A point's anomaly score is the sum, over
    features, of the log-inverse-density of the bin it falls in -- so landing in a
    sparsely-populated bin on any feature adds to the score. That is it: no
    distances, no neighbours, one pass to build the histograms and one to score.

    THE ASSUMPTION AND WHY IT IS WORTH IT
    -------------------------------------
    Summing per-feature scores assumes the features are INDEPENDENT, which is
    usually false -- so HBOS cannot catch an anomaly that is only unusual in the
    COMBINATION of features (each value normal alone). In exchange it is
    linear-time and needs no distance computation, which makes it one of the
    fastest detectors and a strong default on high-dimensional data where
    distance-based methods choke. A classic speed-for-expressiveness trade.

    Goldstein & Dengel (2012).
    """

    def __init__(self, n_bins=10, contamination=0.1):
        self.n_bins = n_bins
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.bin_edges_ = []
        self.log_inv_density_ = []
        for j in range(X.shape[1]):
            hist, edges = np.histogram(X[:, j], bins=self.n_bins, density=True)
            self.bin_edges_.append(edges)
            # log(1/density): a rare (low-density) bin scores high
            self.log_inv_density_.append(np.log(1.0 / (hist + 1e-10)))
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        score = np.zeros(len(X))
        for j in range(X.shape[1]):
            edges = self.bin_edges_[j]
            # which bin each value falls in (clipped to the valid range)
            idx = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                          0, self.n_bins - 1)
            score += self.log_inv_density_[j][idx]
        return score


class ECOD(_Detector):
    """Empirical-CDF Outlier Detection: how deep in the tails does the point sit?

    THE IDEA
    --------
    For each feature, estimate its distribution by the empirical CDF, and measure
    how far into the TAIL a value falls -- ``-log`` of the left-tail probability
    ``F(x)`` and the right-tail ``1-F(x)``. A point that is extreme (either
    direction) on several features accumulates a large tail score. Summing the
    log-tail-probabilities across features gives the anomaly score.

    WHY IT IS NOTABLE
    -----------------
    It is entirely PARAMETER-FREE -- no bins, no k, no bandwidth, no contamination
    needed to score (only to threshold). Nothing to tune means nothing to tune
    wrong, which is why ECOD is a strong, boringly-reliable default. Like HBOS it
    treats features independently, so it too misses purely-interaction anomalies;
    the CDF (versus HBOS's histogram) makes it smoother and better in the tails,
    which is exactly where anomalies live.

    Li et al. (2022).
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.X_train_ = X
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        Xt = self.X_train_
        score = np.zeros(len(X))
        for j in range(X.shape[1]):
            col = Xt[:, j]
            n = len(col)
            # empirical left- and right-tail probabilities of each query value
            left = np.searchsorted(np.sort(col), X[:, j], side="right") / n
            left = np.clip(left, 1e-10, 1)
            right = np.clip(1 - left, 1e-10, 1)
            # take the more extreme tail per feature, in log space
            score += np.maximum(-np.log(left), -np.log(right))
        return score


class COPOD(_Detector):
    """Copula-based outlier detection: the tail idea, combined more carefully.

    Very close to ECOD -- both build per-feature empirical tail probabilities --
    but COPOD frames the combination through a COPULA, the object that couples
    marginal distributions into a joint. In this simplified form it sums the
    left-, right-, and skewness-corrected tail scores, which is a touch more
    robust to asymmetric features than ECOD's plain max. Also parameter-free, also
    fast, and it makes the same feature-independence simplification in the tail
    aggregation.

    Li et al. (2020).
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.X_train_ = X
        self.skew_ = stats.skew(X, axis=0)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        Xt = self.X_train_
        n = len(Xt)
        left_total = np.zeros(len(X))
        right_total = np.zeros(len(X))
        for j in range(X.shape[1]):
            sorted_col = np.sort(Xt[:, j])
            left = np.clip(np.searchsorted(sorted_col, X[:, j], "right") / n,
                           1e-10, 1)
            right = np.clip(1 - left, 1e-10, 1)
            left_total += -np.log(left)
            right_total += -np.log(right)
        # pick the tail direction implied by each feature's skew, then take the
        # larger of the left/right/skew-corrected aggregates -- COPOD's combination
        return np.maximum(left_total, right_total)


class KNN(_Detector):
    """k-nearest-neighbour outlier score: distance to the k-th neighbour.

    THE IDEA
    --------
    A point in a dense region has close neighbours; an outlier's k-th nearest
    neighbour is far away. So the distance to the k-th neighbour (or the mean of
    the k nearest) is a direct anomaly score. Simple, intuitive, and a strong
    baseline whenever a meaningful distance exists.

    THE CATCH
    ---------
    It uses a GLOBAL distance threshold, so it struggles when normal density
    VARIES across the space -- a point that is loose by dense-region standards may
    be perfectly normal in a sparse region, yet KNN scores it high. That failure
    is exactly what LOF (in ``outlier``) fixes by comparing each point's density
    to its neighbours' rather than to a global scale.
    """

    def __init__(self, n_neighbors=5, method="largest", contamination=0.1):
        self.n_neighbors = n_neighbors
        self.method = method            # 'largest' (k-th) or 'mean' of k
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.X_train_ = X
        self._set_threshold(self.decision_function(X, _train=True))
        return self

    def decision_function(self, X, _train=False):
        X = check_array(X)
        d = cdist(X, self.X_train_)
        if _train:
            # exclude self-distance (0) when scoring the training set
            np.fill_diagonal(d, np.inf)
        k = self.n_neighbors
        nearest = np.sort(d, axis=1)[:, :k]
        return nearest[:, -1] if self.method == "largest" else nearest.mean(axis=1)


class CBLOF(_Detector):
    """Cluster-Based Local Outlier Factor: cluster, then score by cluster.

    THE IDEA
    --------
    Cluster the data (k-means). Split the clusters into LARGE and SMALL by size.
    A point in a large cluster is scored by its distance to that cluster's centre;
    a point in a small cluster is scored by its distance to the nearest LARGE
    cluster -- because small, isolated clusters are themselves suspicious. So
    anomalies are points that are either far from their big cluster's core or
    stranded in a tiny cluster.

    Clustering first makes it far cheaper than all-pairs distance methods on large
    data, and the large/small split is what lets it treat a small dense clump as
    anomalous rather than as its own legitimate mode -- something plain KNN cannot.

    He, Xu & Deng (2003).
    """

    def __init__(self, n_clusters=8, alpha=0.9, contamination=0.1,
                 random_state=None):
        self.n_clusters = n_clusters
        self.alpha = alpha              # fraction of points that must be "large"
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        from ..cluster import KMeans
        X = check_array(X)
        km = KMeans(n_clusters=min(self.n_clusters, len(X)), n_init=5,
                    random_state=self.random_state).fit(X)
        self.centers_ = km.cluster_centers_
        labels = km.labels_
        sizes = np.bincount(labels, minlength=len(self.centers_))
        # partition clusters into large/small by cumulative membership share
        order = np.argsort(sizes)[::-1]
        cum = np.cumsum(sizes[order])
        n_large = np.searchsorted(cum, self.alpha * len(X)) + 1
        self.large_ = set(order[:n_large].tolist())
        self._km = km
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        labels = self._km.predict(X)
        large_centers = self.centers_[list(self.large_)]
        score = np.empty(len(X))
        for i, (x, lab) in enumerate(zip(X, labels)):
            if lab in self.large_:
                score[i] = np.linalg.norm(x - self.centers_[lab])
            else:
                # stranded in a small cluster: distance to the nearest big one
                score[i] = np.min(np.linalg.norm(large_centers - x, axis=1))
        return score


class ABOD(_Detector):
    """Angle-Based Outlier Detection: outliers see their neighbours from one side.

    THE INSIGHT
    -----------
    Stand at a normal point in the middle of a cloud: its neighbours surround it,
    so the ANGLES between pairs of them (seen from the point) vary widely. Stand
    at an outlier on the fringe: all its neighbours lie in roughly the same
    direction, so those angles barely vary. The VARIANCE of the angles is
    therefore high for inliers and low for outliers -- and the anomaly score is
    the negative of that variance.

    WHY BOTHER WHEN DISTANCES EXIST
    -------------------------------
    In high dimensions distances concentrate -- every pair of points ends up
    almost equidistant, so distance-based scores lose their power (the curse of
    dimensionality, in its sharpest form). Angles degrade far more gracefully, so
    ABOD keeps discriminating where KNN and LOF have gone flat. That robustness is
    its whole reason to exist.

    Kriegel, Schubert & Zimek (2008).
    """

    def __init__(self, n_neighbors=10, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X):
        X = check_array(X)
        self.X_train_ = X
        self._set_threshold(self.decision_function(X, _train=True))
        return self

    def decision_function(self, X, _train=False):
        X = check_array(X)
        Xt = self.X_train_
        d = cdist(X, Xt)
        if _train:
            np.fill_diagonal(d, np.inf)
        k = min(self.n_neighbors, Xt.shape[0] - (1 if _train else 0))
        scores = np.empty(len(X))
        for i, x in enumerate(X):
            nbr = Xt[np.argsort(d[i])[:k]]
            vecs = nbr - x                          # directions to neighbours
            norms = np.linalg.norm(vecs, axis=1) + 1e-10
            # weighted angle cosines between all neighbour pairs
            angles = []
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    cos = (vecs[a] @ vecs[b]) / (norms[a] * norms[b])
                    # weight by 1/(dist^2) as in the original, so near neighbours
                    # dominate the angle spread
                    angles.append(cos / (norms[a] * norms[b]))
            # low variance of angles => outlier => high score, hence the negation
            scores[i] = -np.var(angles) if angles else 0.0
        return scores


__all__ = ["HBOS", "ECOD", "COPOD", "KNN", "CBLOF", "ABOD"]
