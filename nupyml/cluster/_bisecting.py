"""Bisecting k-means and feature agglomeration: two more clustering shapes."""
import numpy as np

from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class BisectingKMeans(BaseEstimator, ClusterMixin):
    """Build clusters top-down: repeatedly SPLIT the worst cluster in two.

    THE DIFFERENCE FROM K-MEANS
    ---------------------------
    Plain k-means places all ``k`` centres at once and is at the mercy of their
    initialisation. Bisecting k-means instead starts with everything in one
    cluster and, ``k-1`` times, picks the cluster with the largest inertia (the
    loosest one) and splits it into two with a 2-means run. It is a divisive
    hierarchical method wearing a k-means engine.

    WHY IT IS OFTEN BETTER
    ----------------------
    Each split is a small, easy 2-means problem, far less sensitive to
    initialisation than a full k-way one -- so bisecting k-means tends to produce
    more balanced, more stable clusters, especially at large k, and it is much
    faster than agglomerative clustering while giving a similar top-down hierarchy.
    Always splitting the WORST cluster is what drives the total inertia down
    greedily, one bisection at a time.
    """

    def __init__(self, n_clusters=8, random_state=None, n_init=3):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def _two_means(self, X, rng):
        from . import KMeans
        km = KMeans(n_clusters=2, n_init=self.n_init,
                    random_state=rng.randint(1 << 30)).fit(X)
        return km.labels_, km.cluster_centers_

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        # each cluster is (member indices); start with one holding everything
        clusters = [np.arange(len(X))]

        while len(clusters) < self.n_clusters:
            # split the cluster with the largest within-cluster inertia
            inertias = []
            for idx in clusters:
                if len(idx) < 2:
                    inertias.append(-1)
                    continue
                centre = X[idx].mean(axis=0)
                inertias.append(np.sum((X[idx] - centre) ** 2))
            worst = int(np.argmax(inertias))
            if inertias[worst] <= 0:
                break
            idx = clusters.pop(worst)
            sub_labels, _ = self._two_means(X[idx], rng)
            clusters.append(idx[sub_labels == 0])
            clusters.append(idx[sub_labels == 1])

        self.labels_ = np.empty(len(X), dtype=int)
        self.cluster_centers_ = np.zeros((len(clusters), X.shape[1]))
        for c, idx in enumerate(clusters):
            self.labels_[idx] = c
            self.cluster_centers_[c] = X[idx].mean(axis=0)
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        from scipy.spatial.distance import cdist
        return np.argmin(cdist(check_array(X), self.cluster_centers_), axis=1)


class FeatureAgglomeration(BaseEstimator, TransformerMixin):
    """Hierarchical clustering applied to the FEATURES, not the samples.

    THE ROLE REVERSAL
    -----------------
    Agglomerative clustering groups similar SAMPLES. Transpose the data and it
    groups similar FEATURES -- and merging a group of correlated features into
    their mean is a form of dimensionality reduction. Where PCA builds new features
    as linear combinations of ALL originals (dense, hard to interpret), feature
    agglomeration builds each new feature from a DISJOINT group of originals, so
    every reduced feature is "the average of these specific columns" -- readable,
    and structure-preserving.

    It shines when features come in natural correlated blocks (adjacent pixels,
    neighbouring sensors, co-expressed genes): it collapses each block to one
    feature, cutting dimensionality while keeping the reduced features
    interpretable, which the PCA route cannot.
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        from . import AgglomerativeClustering
        X = check_array(X)
        # cluster the COLUMNS by transposing the data
        agg = AgglomerativeClustering(n_clusters=self.n_clusters,
                                      linkage=self.linkage).fit(X.T)
        self.labels_ = agg.labels_
        self.n_features_out_ = self.n_clusters
        return self

    def transform(self, X):
        check_is_fitted(self, "labels_")
        X = check_array(X)
        # each output feature is the MEAN of the originals in its group
        out = np.zeros((len(X), self.n_clusters))
        for c in range(self.n_clusters):
            cols = self.labels_ == c
            if cols.any():
                out[:, c] = X[:, cols].mean(axis=1)
        return out


__all__ = ["BisectingKMeans", "FeatureAgglomeration"]
