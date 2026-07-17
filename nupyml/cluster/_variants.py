"""Clustering variants for the cases plain k-means cannot handle.

K-means assumes three things that are often false: that a cluster centre is the
MEAN of its members (fails for non-numeric data), that clusters are round blobs
of similar size, and that every point belongs fully to exactly one cluster. Each
class here relaxes one of those.
"""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClusterMixin, check_is_fitted
from ..utils import check_array, check_random_state


class KMedoids(BaseEstimator, ClusterMixin):
    """Like k-means, but each centre is an ACTUAL data point (a medoid).

    WHY IT MATTERS
    --------------
    K-means represents a cluster by its mean, which requires the data to live in a
    space where averaging makes sense -- and to be Euclidean. K-medoids drops
    both requirements: the centre is the member that minimises total distance to
    the others, so it works with ANY distance matrix (edit distance between
    strings, a precomputed similarity, a non-metric dissimilarity) and never
    invents an impossible "average" point.

    The second payoff is ROBUSTNESS. A mean is dragged by outliers; a medoid,
    being a real central point, is not -- one wild value cannot move it. K-medoids
    is to k-means as the median is to the mean.

    THE COST
    --------
    Finding the best medoid means checking distances among cluster members, which
    is more expensive than averaging -- so k-medoids is slower and used when its
    generality or robustness is worth it, not by default. This is the PAM
    (Partitioning Around Medoids) update: reassign, then pick each cluster's
    minimum-total-distance member.

    Kaufman & Rousseeuw (1987).
    """

    def __init__(self, n_clusters=8, metric="euclidean", max_iter=300,
                 random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        D = cdist(X, X, metric=self.metric)         # any metric, precomputed once
        medoids = rng.choice(len(X), self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for c in range(self.n_clusters):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                # the medoid is the member with the least total distance to the
                # rest of its cluster -- a real point, not a computed centre
                costs = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[c] = members[np.argmin(costs)]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = np.argmin(D[:, medoids], axis=1)
        self.inertia_ = float(D[np.arange(len(X)), medoids[self.labels_]].sum())
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        D = cdist(check_array(X), self.cluster_centers_, metric=self.metric)
        return np.argmin(D, axis=1)


class KModes(BaseEstimator, ClusterMixin):
    """K-means for CATEGORICAL data: modes, not means, and matching, not distance.

    WHY K-MEANS CANNOT DO THIS
    --------------------------
    Categorical features -- colour, country, blood type -- have no meaningful
    average ("the mean of red and blue"?) and no meaningful Euclidean distance
    (one-hot encoding then k-means invents both, badly). K-modes replaces the two
    Euclidean assumptions with categorical ones:

    * DISSIMILARITY is the count of features that DIFFER (the Hamming distance) --
      how many attributes two points disagree on.
    * a cluster's CENTRE is its MODE -- the most frequent value of each feature
      among its members, which is a real, sensible categorical prototype.

    Everything else is k-means unchanged: assign to the nearest mode, recompute
    modes, repeat. It is the right tool the moment the data is labels rather than
    numbers.

    Huang (1998).
    """

    def __init__(self, n_clusters=8, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _dissim(self, X, centers):
        # mismatch count: for each point-centre pair, how many features differ
        return np.array([[np.sum(x != c) for c in centers] for x in X])

    def fit(self, X, y=None):
        X = np.asarray(X)
        rng = check_random_state(self.random_state)
        centers = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()

        for _ in range(self.max_iter):
            labels = np.argmin(self._dissim(X, centers), axis=1)
            new_centers = centers.copy()
            for c in range(self.n_clusters):
                members = X[labels == c]
                if len(members) == 0:
                    continue
                # the mode of each feature: the most common category among members
                for f in range(X.shape[1]):
                    vals, counts = np.unique(members[:, f], return_counts=True)
                    new_centers[c, f] = vals[np.argmax(counts)]
            if np.array_equal(new_centers, centers):
                break
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = np.argmin(self._dissim(X, centers), axis=1)
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return np.argmin(self._dissim(np.asarray(X), self.cluster_centers_), axis=1)


class FuzzyCMeans(BaseEstimator, ClusterMixin):
    """Soft clustering: each point belongs to every cluster, by degree.

    THE RELAXATION
    --------------
    K-means makes a HARD assignment -- each point is wholly in one cluster. But a
    point sitting between two clusters is genuinely ambiguous, and forcing it into
    one throws that information away. Fuzzy c-means gives each point a MEMBERSHIP
    in [0, 1] to every cluster, summing to 1 -- "70% cluster A, 30% cluster B".

    The centres become WEIGHTED means (by membership), and memberships update from
    inverse distances, alternating like k-means. A point near a centre gets high
    membership there; a point equidistant from two splits evenly. The hard
    assignment falls out as the special case where one membership goes to 1.

    THE FUZZINESS KNOB
    ------------------
    ``m > 1`` controls how soft the clustering is. As ``m -> 1`` memberships
    sharpen toward hard 0/1 (recovering k-means); larger ``m`` makes them fuzzier,
    until at large ``m`` every point belongs equally to everything. ``m = 2`` is
    the standard. This softness is exactly what makes it useful for data with no
    crisp boundaries -- gene expression, image segmentation -- where a hard label
    would be a fiction.

    Dunn (1973); Bezdek (1981).
    """

    def __init__(self, n_clusters=8, m=2.0, max_iter=150, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        # random memberships, normalised so each point's row sums to 1
        U = rng.uniform(size=(n, self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)

        for _ in range(self.max_iter):
            Um = U ** self.m
            # centres are membership-weighted means -- soft version of k-means'
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            D = cdist(X, centers) + 1e-12
            # update memberships from inverse distances; the exponent is the
            # fuzzifier m, and it is what tunes hard-vs-soft
            power = 2.0 / (self.m - 1)
            U_new = 1.0 / np.array([[np.sum((D[i, c] / D[i, :]) ** power)
                                     for c in range(self.n_clusters)]
                                    for i in range(n)])
            if np.max(np.abs(U_new - U)) < self.tol:
                U = U_new
                break
            U = U_new

        self.cluster_centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)         # the hard label, if wanted
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        D = cdist(check_array(X), self.cluster_centers_)
        return np.argmin(D, axis=1)


__all__ = ["KMedoids", "KModes", "FuzzyCMeans"]
