"""Biclustering and two clustering methods the core package lacks.

* co-clustering / biclustering -- cluster ROWS and COLUMNS together;
* density-peak clustering -- centres as high-density points far from other peaks;
* consensus clustering -- combine many runs into one stable partition.
"""
import numpy as np

from ..base import BaseEstimator, ClusterMixin, check_is_fitted
from ..utils import check_array, check_random_state


class SpectralCoclustering(BaseEstimator, ClusterMixin):
    """Cluster rows and columns of a matrix JOINTLY (Dhillon, 2001).

    THE IDEA
    --------
    A data matrix (documents × words, users × items, genes × conditions) often has
    block structure: a group of rows that behaves alike ON a group of columns.
    Clustering rows alone, or columns alone, misses it. Co-clustering treats the
    matrix as a bipartite graph (rows on one side, columns on the other, entries
    as edge weights) and finds a joint partition by a spectral cut of that graph.

    HOW
    ---
    Normalise the matrix as ``A_n = D_r^{-1/2} A D_c^{-1/2}`` (row/column degree
    scaling), take the top ``log2(k)`` singular vectors of ``A_n``, stack the
    scaled left vectors (rows) and right vectors (columns) into one embedding, and
    k-means it. Rows and columns that belong to the same block land near each
    other in that shared spectral space -- so ONE k-means labels both. Assumes the
    matrix is non-negative (it is a co-occurrence / contingency table).

    ``row_labels_`` and ``column_labels_`` give the two partitions.
    """

    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        from . import KMeans
        A = check_array(X).astype(float)
        A = np.maximum(A, 0)
        row_deg = np.sqrt(A.sum(axis=1) + 1e-12)
        col_deg = np.sqrt(A.sum(axis=0) + 1e-12)
        An = A / row_deg[:, None] / col_deg[None, :]
        U, s, Vt = np.linalg.svd(An, full_matrices=False)
        n_vec = max(1, int(np.ceil(np.log2(self.n_clusters))))
        # drop the first singular vector (it encodes the trivial all-connected cut)
        Z_row = (U[:, 1:n_vec + 1]) / row_deg[:, None]
        Z_col = (Vt[1:n_vec + 1].T) / col_deg[:, None]
        stacked = np.vstack([Z_row, Z_col])
        km = KMeans(n_clusters=self.n_clusters,
                    random_state=self.random_state).fit(stacked)
        n_rows = A.shape[0]
        self.row_labels_ = km.labels_[:n_rows]
        self.column_labels_ = km.labels_[n_rows:]
        self.labels_ = self.row_labels_             # convention: row labels
        return self


class DensityPeakClustering(BaseEstimator, ClusterMixin):
    """Density-peak clustering (Rodriguez & Laio, 2014).

    THE ELEGANT CRITERION
    ---------------------
    A cluster centre should be (a) DENSE -- many points nearby -- and (b) FAR from
    any other point that is even denser. Compute for each point its local density
    ``rho`` and the distance ``delta`` to the nearest point of higher density.
    Centres are the points with both large ``rho`` and large ``delta`` (the
    top-right of a "``rho`` vs ``delta``" plot); everything else is assigned to the
    same cluster as its nearest higher-density neighbour, in one descending-density
    pass.

    WHY IT IS NICE
    --------------
    No iteration, no assumption of convex clusters, and it finds the number of
    centres from the data (they stand out as high-``rho``-high-``delta`` outliers).
    Here the ``n_clusters`` points with the largest ``rho*delta`` score are taken
    as centres.
    """

    def __init__(self, n_clusters=3, percent=2.0):
        self.n_clusters = n_clusters
        self.percent = percent

    def fit(self, X, y=None):
        from scipy.spatial.distance import cdist
        X = check_array(X)
        D = cdist(X, X)
        n = len(X)
        # cutoff distance dc: the given percentile of all pairwise distances
        dc = np.percentile(D[np.triu_indices(n, 1)], self.percent)
        rho = (D < dc).sum(axis=1) - 1              # local density (neighbour count)

        order = np.argsort(-rho)                    # densest first
        delta = np.full(n, np.inf)
        nearest_denser = np.full(n, -1, dtype=int)
        for ii in range(1, n):
            i = order[ii]
            higher = order[:ii]                     # all strictly-denser points
            d = D[i, higher]
            k = int(np.argmin(d))
            delta[i] = d[k]
            nearest_denser[i] = higher[k]
        delta[order[0]] = D[order[0]].max()         # densest point: farthest

        centres = np.argsort(-(rho * delta))[:self.n_clusters]
        labels = np.full(n, -1, dtype=int)
        for c, idx in enumerate(centres):
            labels[idx] = c
        # assign the rest in descending density to their nearest denser neighbour
        for i in order:
            if labels[i] == -1:
                labels[i] = labels[nearest_denser[i]]
        self.labels_ = labels
        self.centres_ = centres
        return self


class ConsensusClustering(BaseEstimator, ClusterMixin):
    """Combine many clusterings into one stable partition (Monti et al., 2003).

    THE PROBLEM
    -----------
    A single k-means run depends on its random init and on the sampled data; its
    partition can be unstable. Consensus clustering runs the base clusterer many
    times on bootstrap subsamples, builds a CO-ASSOCIATION matrix (how often each
    pair of points landed in the same cluster), and clusters THAT. Two points that
    are grouped together across most runs end up together in the consensus; a pair
    the base method keeps flip-flopping on is separated.

    The co-association frequency doubles as a stability measure -- a crisp,
    near-0/1 matrix means the structure is robust, a grey one means it is not.
    Here the consensus is obtained by agglomerative clustering on ``1 -
    co-association`` (a distance).
    """

    def __init__(self, n_clusters=3, base_estimator=None, n_runs=25,
                 sample_fraction=0.8, random_state=None):
        self.n_clusters = n_clusters
        self.base_estimator = base_estimator
        self.n_runs = n_runs
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def fit(self, X, y=None):
        from . import KMeans
        from ..base import clone
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        base = (self.base_estimator if self.base_estimator is not None
                else KMeans(n_clusters=self.n_clusters))
        co = np.zeros((n, n))                        # co-association counts
        seen = np.zeros((n, n))                      # co-sampled counts
        m = max(2, int(self.sample_fraction * n))
        for _ in range(self.n_runs):
            idx = rng.choice(n, m, replace=False)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=rng.randint(2 ** 31 - 1))
            labels = est.fit(X[idx]).labels_
            for c in np.unique(labels):
                members = idx[labels == c]
                co[np.ix_(members, members)] += 1
            seen[np.ix_(idx, idx)] += 1
        with np.errstate(invalid="ignore"):
            frac = np.where(seen > 0, co / seen, 0.0)   # co-association frequency
        self.coassociation_ = frac
        # spectral clustering ON the co-association similarity: normalised-Laplacian
        # embedding + k-means (done here since the base estimators take no
        # precomputed affinity)
        deg = np.sqrt(frac.sum(axis=1) + 1e-12)
        L = frac / deg[:, None] / deg[None, :]
        vals, vecs = np.linalg.eigh(L)
        embed = vecs[:, -self.n_clusters:]          # top eigenvectors
        norm = np.linalg.norm(embed, axis=1, keepdims=True)
        embed = embed / np.where(norm > 0, norm, 1.0)
        self.labels_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state).fit(embed).labels_
        return self


__all__ = ["SpectralCoclustering", "DensityPeakClustering", "ConsensusClustering"]
