"""Clustering: finding structure without labels.

There is no single right answer to "what are the clusters", because there is no
ground truth to check against -- only whatever notion of similarity the
algorithm assumes. The methods here disagree about that assumption, and the
disagreement IS the taxonomy:

============================ ==========================================
assumption                   algorithms
============================ ==========================================
clusters are round and       KMeans, MiniBatchKMeans
of similar size
clusters are dense regions   DBSCAN, OPTICS, HDBSCAN
of any shape
clusters form a hierarchy    AgglomerativeClustering, Birch
clusters are modes of a      MeanShift
density
clusters are connected in a  SpectralClustering
graph, not compact in space
every point is an exemplar   AffinityPropagation
candidate
============================ ==========================================

Choosing wrongly does not produce an error; it produces confident nonsense.
KMeans will happily cut a pair of concentric rings into two half-moons, because
its assumption -- clusters are blobs around a centre -- is false there.
SpectralClustering solves that case exactly, by clustering connectivity rather
than position.

The other axis is whether the number of clusters must be known. KMeans demands
k up front. DBSCAN and MeanShift discover it, but demand a scale instead
(``eps``, ``bandwidth``) -- the question does not disappear, it changes form.
"""
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _kmeans_plusplus(X, k, rng):
    """Seed centres by D^2 sampling (Arthur & Vassilvitskii, 2007).

    Picking k centres uniformly at random tends to drop several into the same
    dense region and leave others empty, and Lloyd's algorithm only finds a
    local optimum, so a bad start stays bad. D^2 sampling instead picks each
    new centre with probability proportional to its squared distance from the
    nearest centre already chosen: points in unserved regions are the most
    likely to be picked, which spreads the seeds out. This is what earns
    k-means++ its O(log k) approximation guarantee.
    """
    n = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.randint(n)]
    closest_sq = cdist(X, centers[:1]).ravel() ** 2
    for i in range(1, k):
        total = closest_sq.sum()
        if total > 0:
            probs = closest_sq / total
        else:
            # every point already sits exactly on a chosen centre (duplicated
            # rows, or fewer distinct points than k). D^2 sampling has no
            # signal left to use, so fall back to uniform rather than divide
            # zero by zero
            probs = np.full(n, 1.0 / n)
        centers[i] = X[rng.choice(n, p=probs)]
        d = cdist(X, centers[i:i + 1]).ravel() ** 2
        closest_sq = np.minimum(closest_sq, d)
    return centers


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Partition points into k clusters by minimising within-cluster variance.

    THE OBJECTIVE
    -------------
    Minimise the total squared distance from each point to its cluster's
    centre (the "inertia")::

        sum_i ||x_i - centre(assignment_i)||^2

    Finding the true optimum is NP-hard. Lloyd's algorithm settles for a local
    one by alternating two steps, each of which is trivial given the other:

    * **E-step**: assign every point to its nearest centre (centres fixed).
    * **M-step**: move each centre to the mean of its members (assignments fixed).

    Neither step can ever increase the objective, and there are finitely many
    assignments, so it must terminate. It terminates at a LOCAL minimum, which
    is why ``n_init`` runs the whole thing several times from different seeds
    and keeps the best -- the single most important defence against a bad
    outcome here.

    WHY THE MEAN, SPECIFICALLY
    --------------------------
    The mean is the point minimising SQUARED distance to a set. That is not a
    convention, it is why the M-step is "take the mean" -- change the objective
    to absolute distance and the optimal centre becomes the median (k-medians).
    Squared distance is also what makes KMeans sensitive to outliers: a single
    far-away point can drag a centre across the space.

    THE ASSUMPTIONS, MADE EXPLICIT
    ------------------------------
    Because a point goes to the nearest centre, the boundaries are perpendicular
    bisectors -- straight lines. KMeans can therefore only produce convex,
    roughly spherical, roughly equal-sized clusters. Elongated, nested or
    varying-density structure is out of reach no matter how good the
    optimisation is. That is a property of the objective, not a bug.

    It also uses raw Euclidean distance, so features must be scaled first, or
    whichever feature happens to have the largest units will define the
    clusters by itself.
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _single_run(self, X, rng, w):
        k = self.n_clusters
        if self.init == "k-means++":
            centers = _kmeans_plusplus(X, k, rng)
        elif self.init == "random":
            centers = X[rng.choice(len(X), size=k, replace=False)]
        else:
            centers = np.asarray(self.init, dtype=np.float64).copy()
        for _ in range(self.max_iter):
            # E-step: assign every point to its closest centre
            dist = cdist(X, centers)
            labels = dist.argmin(axis=1)
            # M-step: recompute each centre as the weighted mean of its members.
            # Masking per cluster (``X[labels == c]``) would walk the whole
            # label array k times and allocate a fresh block each time. A
            # scatter-add visits each sample once instead: bincount sums the
            # weights landing in each cluster, and one more bincount per
            # feature sums the weighted coordinates. O(n*d) with no per-cluster
            # pass, and no temporaries.
            weight_per_cluster = np.bincount(labels, weights=w, minlength=k)
            new_centers = np.stack(
                [np.bincount(labels, weights=w * X[:, j], minlength=k)
                 for j in range(X.shape[1])], axis=1)
            alive = weight_per_cluster > 0
            new_centers[alive] /= weight_per_cluster[alive, None]
            if not alive.all():
                # a centre that captured nothing is wasted: restart it at the
                # point currently worst served, which is where a centre helps most
                new_centers[~alive] = X[dist.min(axis=1).argmax()]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break
        dist = cdist(X, centers)
        labels = dist.argmin(axis=1)
        inertia = float((w * dist[np.arange(len(X)), labels] ** 2).sum())
        return centers, labels, inertia

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=True)
        if sp.issparse(X):
            # centroids are dense anyway, so densify once rather than in the
            # inner loop
            X = np.asarray(X.todense())
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        rng = check_random_state(self.random_state)
        runs = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(runs):
            centers, labels, inertia = self._single_run(X, rng, w)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def _densify(self, X):
        X = check_array(X, accept_sparse=True)
        return np.asarray(X.todense()) if sp.issparse(X) else X

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(self._densify(X), self.cluster_centers_).argmin(axis=1)

    def transform(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(self._densify(X), self.cluster_centers_)


class MiniBatchKMeans(BaseEstimator, ClusterMixin):
    """KMeans on random mini-batches: much faster, slightly worse.

    Lloyd's E-step touches every point every iteration. Mini-batch KMeans
    updates from a random subset instead, moving each centre a little way
    toward its members in the batch::

        centre <- (1 - eta) * centre + eta * batch_mean,   eta = 1/count

    The learning rate falls as ``1/count`` because that makes the centre the
    running MEAN of everything it has ever been assigned -- an online average.
    Early batches move a centre a lot, later ones refine it, and the updates
    settle rather than oscillate.

    The result is typically a slightly worse inertia for a large constant-factor
    speedup, and it supports ``partial_fit`` for data that never fits in memory.
    """

    def __init__(self, n_clusters=8, batch_size=256, max_iter=100,
                 random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        centers = _kmeans_plusplus(X, self.n_clusters, rng)
        counts = np.zeros(self.n_clusters)
        n = len(X)
        for _ in range(self.max_iter):
            batch = X[rng.randint(0, n, size=min(self.batch_size, n))]
            labels = cdist(batch, centers).argmin(axis=1)
            for c in np.unique(labels):
                pts = batch[labels == c]
                counts[c] += len(pts)
                eta = len(pts) / counts[c]
                centers[c] = (1 - eta) * centers[c] + eta * pts.mean(axis=0)
        self.cluster_centers_ = centers
        self.labels_ = cdist(X, centers).argmin(axis=1)
        return self

    def partial_fit(self, X, y=None):
        """Update centers from one mini-batch of data."""
        X = check_array(X)
        if not hasattr(self, "cluster_centers_"):
            rng = check_random_state(self.random_state)
            k = min(self.n_clusters, len(X))
            self.cluster_centers_ = _kmeans_plusplus(X, k, rng)
            self._counts = np.zeros(len(self.cluster_centers_))
        labels = cdist(X, self.cluster_centers_).argmin(axis=1)
        for c in np.unique(labels):
            pts = X[labels == c]
            self._counts[c] += len(pts)
            eta = len(pts) / self._counts[c]
            self.cluster_centers_[c] = ((1 - eta) * self.cluster_centers_[c]
                                        + eta * pts.mean(axis=0))
        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_).argmin(axis=1)


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-Based Spatial Clustering of Applications with Noise.

    THE IDEA
    --------
    A cluster is a region where points are packed closely together, separated
    from other such regions by sparseness. No centres, no shapes assumed -- just
    density. Two parameters define "dense": a radius ``eps`` and a count
    ``min_samples``.

    Every point is then one of three things:

    * **core**: has at least ``min_samples`` neighbours within ``eps``. It sits
      in the interior of a dense region.
    * **border**: within ``eps`` of a core point, but not dense itself. The
      fringe of a cluster.
    * **noise**: neither. Labelled ``-1`` and belonging to NO cluster.

    Clusters grow by chaining core points: if two core points are neighbours,
    they are in the same cluster, transitively. That chaining is what lets
    DBSCAN trace an arbitrarily long, curved, or nested shape -- something no
    centroid method can do.

    WHAT IT BUYS AND COSTS
    ----------------------
    * k is discovered, not specified.
    * Outliers are identified rather than forced into a cluster. Almost alone
      among clustering methods, DBSCAN is allowed to say "this point is not in
      any cluster".
    * But density is assumed UNIFORM: one global ``eps`` must fit every
      cluster. Given a tight cluster and a diffuse one, no single eps works --
      the tight one absorbs its neighbours or the diffuse one dissolves into
      noise. ``HDBSCAN`` exists precisely to remove this constraint, by
      considering all eps at once.
    * Border points are assigned to whichever core reached them first, so their
      labels can depend on iteration order.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        tree = cKDTree(X)
        neighbors = tree.query_ball_point(X, self.eps)
        core = np.array([len(nb) >= self.min_samples for nb in neighbors])
        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not core[i]:
                continue
            # BFS expand cluster
            labels[i] = cluster
            queue = list(neighbors[i])
            while queue:
                j = queue.pop()
                if labels[j] == -1:
                    labels[j] = cluster
                    if core[j]:
                        queue.extend(q for q in neighbors[j] if labels[q] == -1)
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where(core)[0]
        return self


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    """Bottom-up hierarchical clustering: start with n clusters, merge to k.

    Begin with every point its own cluster, then repeatedly merge the two
    closest ones. The result is a whole tree of nestings (a dendrogram), not a
    single partition -- ``n_clusters`` just says where to cut it. That is the
    real appeal: the structure at every scale is available at once.

    "Closest" needs defining for SETS of points, and the choice changes the
    outcome more than anything else here:

    * ``ward`` -- merge whichever pair increases within-cluster variance least.
      Tends toward compact, equal-sized clusters (KMeans-like preferences,
      hierarchically arranged).
    * ``complete`` -- distance between the two FURTHEST members. Compact,
      outlier-sensitive.
    * ``average`` -- mean pairwise distance. A middle ground.
    * ``single`` -- distance between the two NEAREST members. Can follow long
      chains, so it traces non-convex shapes but suffers "chaining": a thin
      bridge of points fuses two real clusters.

    Cost is at least quadratic in n, which is what limits this to modest data --
    and what ``Birch`` addresses, by summarising the data first.
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        X = check_array(X)
        Z = sch.linkage(X, method=self.linkage)
        self.labels_ = sch.fcluster(Z, t=self.n_clusters, criterion="maxclust") - 1
        self.linkage_matrix_ = Z
        return self


class MeanShift(BaseEstimator, ClusterMixin):
    """Move every point uphill to a mode of the density; modes become clusters.

    Treat the data as samples from some density. Each point repeatedly hops to
    the mean of its neighbours within ``bandwidth`` -- a step that always points
    uphill, so it is gradient ascent on the density without ever computing a
    gradient. Points converging to the same peak form a cluster.

    So k is not chosen; it emerges as the number of modes. But ``bandwidth``
    controls that number completely: wide smooths everything into one mode,
    narrow finds a mode per point. The question "how many clusters?" has become
    "at what scale?", which is sometimes easier to answer and never disappears.
    """

    def __init__(self, bandwidth=None, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X = check_array(X)
        bw = self.bandwidth or float(np.mean(cdist(X, X).std(axis=1)))
        tree = cKDTree(X)
        points = X.copy()
        for _ in range(self.max_iter):
            moved = 0.0
            for i in range(len(points)):
                nb = tree.query_ball_point(points[i], bw)
                if not nb:
                    continue
                new = X[nb].mean(axis=0)
                moved = max(moved, np.linalg.norm(new - points[i]))
                points[i] = new
            if moved < self.tol:
                break
        # merge modes closer than bandwidth/2
        centers = []
        labels = np.full(len(X), -1)
        for i, p in enumerate(points):
            for ci, c in enumerate(centers):
                if np.linalg.norm(p - c) < bw / 2:
                    labels[i] = ci
                    break
            else:
                centers.append(p)
                labels[i] = len(centers) - 1
        self.cluster_centers_ = np.array(centers)
        self.labels_ = labels
        return self


class SpectralClustering(BaseEstimator, ClusterMixin):
    """Cluster a graph, not a cloud: connectivity instead of compactness.

    THE IDEA
    --------
    Build a graph whose edges say how similar points are, then look for a way
    to cut it into pieces with few edges crossing. Points that are far apart in
    space but joined by a chain of neighbours stay together -- which is why this
    separates concentric circles, and KMeans cannot.

    Cutting a graph optimally is NP-hard. The spectral relaxation gets around it
    with a striking fact about the graph Laplacian ``L = D - W``:

    * ``x' L x = sum over edges of w_ij*(x_i - x_j)^2``, so the Laplacian
      MEASURES how much a labelling disagrees across edges.
    * Its smallest eigenvalue is always 0. The number of eigenvalues equal to 0
      is exactly the number of connected components -- so the bottom of the
      spectrum encodes the component structure exactly.
    * For a graph that is nearly disconnected, the smallest nonzero eigenvectors
      are nearly constant within each near-component. Embedding points using
      those eigenvectors places each near-component at its own spot.

    So: embed with the bottom eigenvectors, then run KMeans in that space, where
    the clusters ARE now compact blobs. The hard geometry has been moved into a
    space where the easy algorithm is correct.

    The affinity choice matters most. ``nearest_neighbors`` builds a sparse graph
    that follows the data's shape; ``rbf`` connects everything with decaying
    weight, and its ``gamma`` decides what "near" means.
    """

    def __init__(self, n_clusters=8, affinity="rbf", gamma=1.0, n_neighbors=10,
                 random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        if self.affinity == "rbf":
            W = np.exp(-self.gamma * cdist(X, X) ** 2)
        elif self.affinity == "nearest_neighbors":
            tree = cKDTree(X)
            _, idx = tree.query(X, k=self.n_neighbors + 1)
            W = np.zeros((n, n))
            rows = np.repeat(np.arange(n), self.n_neighbors)
            W[rows, idx[:, 1:].ravel()] = 1.0
            W = np.maximum(W, W.T)
        else:
            raise ValueError(f"Unknown affinity: {self.affinity!r}")
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        # smallest eigenvectors of the normalized laplacian
        k = self.n_clusters
        if n <= 200:
            vals, vecs = np.linalg.eigh(L_sym)
            U = vecs[:, :k]
        else:
            vals, U = sp.linalg.eigsh(sp.csr_matrix(L_sym), k=k, which="SM")
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        km = KMeans(n_clusters=k, random_state=self.random_state).fit(U)
        self.labels_ = km.labels_
        self.affinity_matrix_ = W
        return self


from .birch import Birch
from .optics import OPTICS
from .affinity_propagation import AffinityPropagation
from .hdbscan import HDBSCAN
from ._variants import KMedoids, KModes, FuzzyCMeans  # noqa: E402

__all__ = ["KMeans", "MiniBatchKMeans", "DBSCAN", "AgglomerativeClustering",
           "MeanShift", "SpectralClustering", "Birch", "OPTICS",
           "AffinityPropagation", "HDBSCAN",
           "KMedoids", "KModes", "FuzzyCMeans",
           "BisectingKMeans", "FeatureAgglomeration",
           "SpectralCoclustering", "DensityPeakClustering", "ConsensusClustering",
           "DPMeans", "PossibilisticCMeans", "SparseSubspaceClustering"]
from ._bisecting import BisectingKMeans, FeatureAgglomeration  # noqa: E402
from ._biclustering import (SpectralCoclustering, DensityPeakClustering,  # noqa: E402
                            ConsensusClustering)
from .dp_means import DPMeans
from .possibilistic_c_means import PossibilisticCMeans
from .sparse_subspace_clustering import SparseSubspaceClustering
