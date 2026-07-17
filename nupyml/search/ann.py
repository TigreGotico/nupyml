"""Approximate nearest neighbours: finding close points without scanning all.

THE PROBLEM
-----------
Exact nearest-neighbour search compares the query to every point -- O(n) per
query, hopeless at billions of vectors. But you rarely need the EXACT nearest;
the second- or third-nearest is usually just as good. Trading a small chance of
missing the true nearest for sublinear search is the bargain the whole field is
built on.

THREE STRATEGIES, THREE INTUITIONS
----------------------------------
* **Hashing** (``RandomProjectionLSH``, ``MinHashLSH``) -- design a hash where
  NEARBY points collide. Then a query only compares against its own bucket. The
  art is a hash whose collision probability rises with similarity.
* **Partitioning** (``IVFIndex``) -- cluster the data once; a query searches only
  the few nearest clusters. Coarse but effective, and the basis of billion-scale
  vector databases.
* **Compression** (``ProductQuantizer``) -- shrink each vector to a few bytes so
  distances can be computed in cache, approximately, at enormous throughput.

They compose in practice (IVF + PQ is the standard recipe); kept separate here so
each idea is legible on its own.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from .sketch import MinHash


class RandomProjectionLSH(BaseEstimator):
    """LSH for Euclidean/cosine space: random hyperplanes as the hash.

    THE HASH
    --------
    Project each point onto ``n_bits`` random hyperplanes and record which side of
    each it falls on -- a bit string. Two points close in angle are separated by
    few random hyperplanes, so they share most bits and land in the same bucket;
    distant points are split by many and rarely collide. The probability of a
    collision on one hyperplane is exactly ``1 - angle/pi`` -- the hash is
    LOCALITY-SENSITIVE by construction, not by luck.

    THE TABLE TRADE-OFF
    -------------------
    More bits per table => fewer collisions => higher precision but more misses
    (RECALL drops). To recover recall, use several INDEPENDENT tables and take the
    union of their buckets: a true neighbour missed by one table is likely caught
    by another. ``n_bits`` and ``n_tables`` are the two dials -- bits for
    precision, tables for recall -- and tuning them is the whole craft of LSH.

    Indyk & Motwani (1998).
    """

    def __init__(self, n_bits=12, n_tables=8, random_state=None):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self.X_ = X
        d = X.shape[1]
        # one independent set of random hyperplanes per table
        self.planes_ = [rng.normal(size=(self.n_bits, d))
                        for _ in range(self.n_tables)]
        self.tables_ = []
        for planes in self.planes_:
            codes = (X @ planes.T > 0)          # the sign bits, per point
            table = {}
            for i, code in enumerate(codes):
                table.setdefault(code.tobytes(), []).append(i)
            self.tables_.append(table)
        return self

    def query(self, x, k=5):
        """Candidate neighbours from the query's buckets, re-ranked exactly.

        LSH narrows billions to a handful; the exact distance is then computed on
        just those candidates. So the approximation is only in WHICH points are
        considered, never in how they are ranked -- the returned order is exact
        among the candidates.
        """
        x = np.asarray(x, float).ravel()
        candidates = set()
        for planes, table in zip(self.planes_, self.tables_):
            code = (planes @ x > 0).tobytes()
            candidates.update(table.get(code, []))
        if not candidates:
            return np.array([], dtype=int)
        cand = np.array(sorted(candidates))
        dists = np.linalg.norm(self.X_[cand] - x, axis=1)
        return cand[np.argsort(dists)[:k]]


class MinHashLSH(BaseEstimator):
    """LSH for SETS: banded MinHash signatures for near-duplicate detection.

    THE BANDING TRICK
    -----------------
    Take each set's MinHash signature (which estimates Jaccard similarity; see
    ``sketch.MinHash``) and split it into ``n_bands`` bands of ``rows_per_band``
    rows. Two sets are candidates if they match on ANY band exactly.

    The banding turns a smooth similarity into a sharp THRESHOLD. The probability
    two sets become candidates is ``1 - (1 - s^r)^b`` in their Jaccard ``s`` -- an
    S-curve whose steepness and location are set by the band shape. Choose ``r``
    and ``b`` to put the steep part right at your similarity threshold, and you
    get near-perfect recall above it and near-zero false positives below. This is
    how web-scale near-duplicate document detection is done.

    Leskovec, Rajaraman & Ullman, MMDS.
    """

    def __init__(self, n_bands=16, rows_per_band=8, random_state=None):
        self.n_bands = n_bands
        self.rows_per_band = rows_per_band
        self.random_state = random_state

    def fit(self, sets):
        self._minhash = MinHash(n_perm=self.n_bands * self.rows_per_band)
        self.signatures_ = [self._minhash.signature(s) for s in sets]
        self.sets_ = [set(s) for s in sets]
        # one hash table per band; a band-slice of the signature is the key
        self.bands_ = [{} for _ in range(self.n_bands)]
        for idx, sig in enumerate(self.signatures_):
            for b in range(self.n_bands):
                lo = b * self.rows_per_band
                key = sig[lo:lo + self.rows_per_band].tobytes()
                self.bands_[b].setdefault(key, []).append(idx)
        return self

    def query(self, items, threshold=0.5):
        """Sets sharing at least one band, filtered to a true Jaccard threshold."""
        sig = self._minhash.signature(items)
        query_set = set(items)
        candidates = set()
        for b in range(self.n_bands):
            lo = b * self.rows_per_band
            key = sig[lo:lo + self.rows_per_band].tobytes()
            candidates.update(self.bands_[b].get(key, []))
        # verify candidates with the exact Jaccard -- banding proposes, exact
        # similarity disposes
        out = []
        for idx in candidates:
            inter = len(query_set & self.sets_[idx])
            union = len(query_set | self.sets_[idx])
            if union and inter / union >= threshold:
                out.append(idx)
        return sorted(out)


class IVFIndex(BaseEstimator):
    """Inverted file index: cluster once, then search only nearby clusters.

    THE IDEA
    --------
    Run k-means to carve the space into ``n_clusters`` cells and file each point
    under its nearest centroid. A query compares against the ``n_probe`` NEAREST
    centroids only, then scans just those cells -- a tiny fraction of the data.

    ``n_probe`` is the accuracy dial with an honest meaning: probe 1 cell and you
    are fast but miss any true neighbour sitting just across a cell boundary;
    probe more cells and recall climbs toward exact search as ``n_probe`` reaches
    ``n_clusters``. The boundary problem -- the nearest point may be in the
    adjacent cell -- is inherent to partitioning, and probing multiple cells is
    the direct, tunable remedy. This is the coarse quantizer at the heart of
    FAISS-style billion-scale indexes.
    """

    def __init__(self, n_clusters=100, n_probe=8, random_state=None):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.random_state = random_state

    def fit(self, X):
        from ..cluster import KMeans
        X = check_array(X)
        self.X_ = X
        n_clusters = min(self.n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                    n_init=3).fit(X)
        self.centroids_ = km.cluster_centers_
        self.assignments_ = km.labels_
        # the inverted lists: for each cell, the points filed under it
        self.lists_ = [np.where(self.assignments_ == c)[0]
                       for c in range(len(self.centroids_))]
        return self

    def query(self, x, k=5):
        x = np.asarray(x, float).ravel()
        # find the n_probe nearest cells, then scan only their contents
        cdists = np.linalg.norm(self.centroids_ - x, axis=1)
        probe = np.argsort(cdists)[:self.n_probe]
        candidates = np.concatenate([self.lists_[c] for c in probe]) \
            if len(probe) else np.array([], dtype=int)
        if len(candidates) == 0:
            return np.array([], dtype=int)
        dists = np.linalg.norm(self.X_[candidates] - x, axis=1)
        return candidates[np.argsort(dists)[:k]]


class ProductQuantizer(BaseEstimator):
    """Compress vectors to a few bytes, so distances fit in cache.

    THE IDEA
    --------
    A 128-dim float vector is 512 bytes -- too big to keep billions in memory.
    Split it into ``n_subvectors`` chunks and quantize each chunk to its nearest
    entry in a small per-chunk codebook (learned by k-means). A vector becomes a
    handful of BYTE codes, a ~64x compression.

    Why "product": the effective codebook is the CARTESIAN PRODUCT of the
    sub-codebooks, so ``m`` chunks of ``256`` codes each represent ``256^m``
    distinct vectors while storing only ``m * 256`` centroids. Exponential
    representational power from linear storage -- that factorisation is the whole
    trick.

    ASYMMETRIC DISTANCE COMPUTATION
    -------------------------------
    At query time the query stays UNCOMPRESSED. Precompute its distance to every
    sub-codebook entry once, into a small table; then each database vector's
    approximate distance is just ``m`` table lookups and adds -- no arithmetic on
    the vectors themselves. Millions of distances become millions of cache-friendly
    additions, which is what makes PQ fast rather than merely small.

    Jegou, Douze & Schmid (2011).
    """

    def __init__(self, n_subvectors=8, n_codes=256, random_state=None):
        self.n_subvectors = n_subvectors
        self.n_codes = n_codes
        self.random_state = random_state

    def fit(self, X):
        from ..cluster import KMeans
        X = check_array(X)
        d = X.shape[1]
        if d % self.n_subvectors != 0:
            raise ValueError(
                f"n_features ({d}) must be divisible by n_subvectors "
                f"({self.n_subvectors})")
        self.sub_dim_ = d // self.n_subvectors
        # one k-means codebook per chunk of the vector
        self.codebooks_ = []
        for m in range(self.n_subvectors):
            sub = X[:, m * self.sub_dim_:(m + 1) * self.sub_dim_]
            n_codes = min(self.n_codes, len(np.unique(sub, axis=0)))
            km = KMeans(n_clusters=n_codes, random_state=self.random_state,
                        n_init=3).fit(sub)
            self.codebooks_.append(km.cluster_centers_)
        self.codes_ = self.encode(X)
        self.X_ = X
        return self

    def encode(self, X):
        """Each vector as ``n_subvectors`` byte codes."""
        X = check_array(X)
        codes = np.empty((len(X), self.n_subvectors), dtype=np.int32)
        for m in range(self.n_subvectors):
            sub = X[:, m * self.sub_dim_:(m + 1) * self.sub_dim_]
            dists = np.linalg.norm(sub[:, None] - self.codebooks_[m][None], axis=2)
            codes[:, m] = np.argmin(dists, axis=1)
        return codes

    def query(self, x, k=5):
        """Approximate nearest neighbours via the asymmetric distance table."""
        x = np.asarray(x, float).ravel()
        # precompute the query's distance to every sub-codebook entry, ONCE
        tables = []
        for m in range(self.n_subvectors):
            xs = x[m * self.sub_dim_:(m + 1) * self.sub_dim_]
            tables.append(np.linalg.norm(self.codebooks_[m] - xs, axis=1) ** 2)
        # each database vector's distance is now m table lookups summed -- no
        # arithmetic on the vectors themselves
        approx = np.zeros(len(self.codes_))
        for m in range(self.n_subvectors):
            approx += tables[m][self.codes_[:, m]]
        return np.argsort(approx)[:k]


class LSHForest(RandomProjectionLSH):
    """A convenience alias: an LSH index over random projections.

    The name follows sklearn's retired ``LSHForest``; the mechanism is
    ``RandomProjectionLSH`` with multiple tables, which is the "forest".
    """


__all__ = ["LSHForest", "MinHashLSH", "RandomProjectionLSH", "IVFIndex",
           "ProductQuantizer"]
