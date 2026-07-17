"""Probabilistic sketches: answering questions about a stream in tiny memory.

THE SETTING
-----------
A stream too large to store -- every click, every packet, every word. You want a
statistic (how many distinct users? how often this IP? have I seen this before?)
but cannot keep the data. A sketch keeps a small, FIXED-SIZE summary that never
grows with the stream, and answers approximately.

THE TRADE, MADE PRECISE
-----------------------
Each sketch swaps exactness for space, and the error is not vague -- it is bounded
and tunable. Add memory, shrink the error. That controllability is what separates
a sketch from a mere heuristic: you choose the accuracy and pay the corresponding
space, and the guarantee holds.

The engine in every case is HASHING. Hashing scatters items uniformly, and the
mathematics of collisions and extreme values under uniform hashing is what turns
a tiny array into a calibrated estimator.
"""
import numpy as np


def _hash(item, seed):
    """A seeded, stable hash of any hashable item.

    Python's ``hash`` is salted per process (for security), so it is unstable
    across runs -- useless for a data structure that must be reproducible.
    Combining a fixed content hash with the seed gives a deterministic family of
    hash functions, which is what every sketch below needs.
    """
    h = hash((seed, repr(item)))
    return h & 0x7FFFFFFF


class BloomFilter:
    """Set membership with NO false negatives, in a bit array.

    THE GUARANTEE, AND ITS ONE-SIDEDNESS
    ------------------------------------
    "Have I seen ``x`` before?" answered from a fixed bit array. Add an item by
    setting ``k`` hash-chosen bits; query by checking whether all ``k`` are set.

    The error is ONE-SIDED, which is the whole point:

    * "not present" is ALWAYS correct -- if any of the bits is 0, the item was
      never added (setting bits is the only thing add does, so a 0 is proof).
    * "present" MIGHT be wrong -- the ``k`` bits could have been set by other
      items colliding. A false positive, never a false negative.

    That asymmetry is exactly right for the classic uses: skip a slow disk lookup
    when the filter says "definitely not there" (safe), and tolerate the rare
    unnecessary lookup when it says "maybe there". You can never wrongly skip a
    real item.

    ``m`` bits and ``k`` hashes trade space for the false-positive rate; the
    optimal ``k`` for a target capacity is derived below. Items cannot be removed
    -- clearing bits would create false negatives, breaking the one guarantee the
    structure makes.
    """

    def __init__(self, capacity=1000, error_rate=0.01):
        # the textbook optimal sizing for a target false-positive rate
        self.m = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        self.k = max(1, int(self.m / capacity * np.log(2)))
        self.bits = np.zeros(self.m, dtype=bool)
        self.capacity = capacity
        self.n_added = 0

    def add(self, item):
        for i in range(self.k):
            self.bits[_hash(item, i) % self.m] = True
        self.n_added += 1

    def __contains__(self, item):
        # all k bits set => "maybe"; any bit clear => "definitely not"
        return all(self.bits[_hash(item, i) % self.m] for i in range(self.k))

    def estimated_false_positive_rate(self):
        """Rises as the filter fills -- worth watching, since a full Bloom filter
        silently degrades to "everything is present"."""
        frac_set = self.bits.mean()
        return frac_set ** self.k


class CountMinSketch:
    """Approximate frequency counts, always an OVER-estimate.

    THE IDEA
    --------
    A 2D table of counters, ``depth`` rows each with its own hash. To count an
    item, increment one cell per row; to query, return the MINIMUM over its rows.

    Collisions can only ADD to a counter, never subtract, so every row's count is
    an over-estimate -- and the minimum is the tightest of several over-estimates,
    which is why ``min`` and not ``mean``. The error is one-sided (never
    under-counts) and bounded by the table width. Heavy hitters, whose true count
    dwarfs the collision noise, are estimated almost exactly; that is the regime
    it is built for.

    The dual of a Bloom filter: that answers presence, this answers frequency,
    both from a hashed table with a one-sided error.

    Cormode & Muthukrishnan (2005).
    """

    def __init__(self, width=1000, depth=5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)

    def add(self, item, count=1):
        for row in range(self.depth):
            self.table[row, _hash(item, row) % self.width] += count

    def estimate(self, item):
        # the minimum row is the least-contaminated over-estimate
        return int(min(self.table[row, _hash(item, row) % self.width]
                       for row in range(self.depth)))


class HyperLogLog:
    """Count DISTINCT items in kilobytes, however many billions there are.

    THE IDEA, WHICH IS GENUINELY SURPRISING
    ---------------------------------------
    Estimate the number of distinct items without storing any of them, from a
    single statistic: the longest run of leading zeros seen in the items' hashes.

    Why it works: a random hash has a leading zero with probability 1/2, two with
    1/4, ``r`` with ``2^-r``. So seeing a hash that begins with ``r`` zeros is
    evidence you have hashed about ``2^r`` distinct items -- a rare event that
    only shows up once the stream is large. The maximum run length is thus a
    (noisy) logarithm of the cardinality.

    To tame the noise, split items into ``2^p`` buckets by their first ``p`` bits,
    track the max run in each, and combine with a HARMONIC mean (which resists the
    upward pull of any one lucky bucket). The result estimates cardinalities into
    the billions using a few kilobytes, with a relative error around
    ``1.04 / sqrt(2^p)``.

    Flajolet, Fusy, Gandouet & Meunier (2007).
    """

    def __init__(self, p=14):
        self.p = p
        self.m = 1 << p                      # number of buckets
        self.registers = np.zeros(self.m, dtype=np.int8)
        # the bias-correction constant for the harmonic-mean estimator
        self.alpha = 0.7213 / (1 + 1.079 / self.m) if self.m >= 128 else 0.673

    def add(self, item):
        h = _hash(item, 0)
        bucket = h & (self.m - 1)            # first p bits pick the bucket
        rest = h >> self.p
        # position of the leftmost 1 in the remaining bits = leading-zeros + 1
        rank = 1
        while rest & 1 == 0 and rank <= 32:
            rank += 1
            rest >>= 1
        # keep the MAX run seen in this bucket -- a distinct item can only raise it
        self.registers[bucket] = max(self.registers[bucket], rank)

    def count(self):
        # harmonic mean of 2^register across buckets, then bias-corrected
        harmonic = 1.0 / np.sum(2.0 ** -self.registers.astype(float))
        estimate = self.alpha * self.m ** 2 * harmonic
        # small-range correction: when many buckets are empty, linear counting is
        # more accurate than the raw estimator
        if estimate <= 2.5 * self.m:
            zeros = np.sum(self.registers == 0)
            if zeros > 0:
                return self.m * np.log(self.m / zeros)
        return estimate


class ReservoirSampler:
    """A uniform sample of ``k`` items from a stream of unknown length.

    THE PROBLEM
    -----------
    You want ``k`` items chosen uniformly at random from a stream, but you do not
    know how long the stream is (so you cannot pick indices in advance) and cannot
    store it (so you cannot sample at the end).

    THE TRICK
    ---------
    Keep the first ``k``. For the ``i``-th item after that, keep it with
    probability ``k/i``, evicting a random current member if you do. A one-line
    induction shows every item seen so far has exactly probability ``k/n`` of
    being in the reservoir at any moment -- so it stays uniform after every step,
    no matter when the stream ends. One pass, ``O(k)`` memory, no length needed.

    Vitter (1985).
    """

    def __init__(self, k, random_state=None):
        self.k = k
        self.reservoir = []
        self.n_seen = 0
        self._rng = np.random.RandomState(random_state)

    def add(self, item):
        self.n_seen += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)          # fill the reservoir first
        else:
            # keep with probability k/n, replacing a random incumbent -- this is
            # exactly what keeps every seen item at probability k/n
            j = self._rng.randint(self.n_seen)
            if j < self.k:
                self.reservoir[j] = item

    def sample(self):
        return list(self.reservoir)


class TDigest:
    """Streaming quantiles, accurate at the TAILS where it matters most.

    THE PROBLEM WITH NAIVE STREAMING QUANTILES
    ------------------------------------------
    You want the 99th percentile of a latency stream (the number SLAs are written
    against) without storing every measurement. A fixed histogram is accurate only
    where you happened to place its bins; the extreme tail, which is the whole
    point, is where bins are sparsest and error is worst.

    THE IDEA
    --------
    Keep a set of weighted CENTROIDS summarising clusters of values, but vary
    their size by position: centroids near the median may be large (coarse
    resolution is fine there), while those near 0 and 1 are kept SMALL (fine
    resolution where accuracy matters). A scale function enforces this, so the
    99.9th percentile is estimated far more accurately than a uniform summary of
    the same size could manage.

    This simplified version merges the nearest centroids when it exceeds its size
    budget -- the essential mechanism, if not the full mergeable machinery.

    Dunning & Ertl (2019).
    """

    def __init__(self, compression=100):
        self.compression = compression
        self.centroids = []      # list of [mean, weight], kept sorted by mean

    def add(self, x, weight=1.0):
        self.centroids.append([float(x), float(weight)])
        if len(self.centroids) > self.compression * 2:
            self._compress()

    def _compress(self):
        """Merge adjacent centroids, allowing bigger ones toward the middle.

        The size limit per centroid scales with ``q*(1-q)`` -- near zero at the
        tails, largest at the median -- which is what preserves tail accuracy
        while still bounding the total number of centroids.
        """
        self.centroids.sort(key=lambda c: c[0])
        total = sum(c[1] for c in self.centroids)
        merged = [self.centroids[0][:]]
        cum = self.centroids[0][1]
        for mean, w in self.centroids[1:]:
            q = cum / total                       # position of this centroid
            # the tail-preserving size bound, largest at q=0.5
            limit = 4 * total * q * (1 - q) / self.compression
            if merged[-1][1] + w <= max(limit, 1.0):
                mw = merged[-1][1]
                merged[-1][0] = (merged[-1][0] * mw + mean * w) / (mw + w)
                merged[-1][1] = mw + w
            else:
                merged.append([mean, w])
            cum += w
        self.centroids = merged

    def quantile(self, q):
        """Estimate the value at quantile ``q`` by interpolating between centroids."""
        if not self.centroids:
            return np.nan
        self.centroids.sort(key=lambda c: c[0])
        total = sum(c[1] for c in self.centroids)
        target = q * total
        cum = 0.0
        for mean, w in self.centroids:
            if cum + w / 2 >= target:
                return mean
            cum += w
        return self.centroids[-1][0]


class MinHash:
    """Estimate the Jaccard similarity of two SETS from tiny signatures.

    THE IDEA
    --------
    Hash a set's elements ``n_perm`` different ways; the signature is the MINIMUM
    hash under each. The magic: for one hash, the probability that two sets share
    a minimum equals their Jaccard similarity exactly -- because the element
    achieving the global minimum is equally likely to be any element of the union,
    and it lands in the intersection with probability |intersection| / |union|.

    So the FRACTION of the ``n_perm`` signature slots where two sets agree is an
    unbiased estimate of their Jaccard similarity. Two enormous sets are compared
    by two short integer vectors, and the error shrinks as ``1/sqrt(n_perm)``.
    This is the engine behind ``MinHashLSH`` for near-duplicate detection.

    Broder (1997).
    """

    def __init__(self, n_perm=128):
        self.n_perm = n_perm

    def signature(self, items):
        """The vector of per-permutation minimum hashes for a set."""
        sig = np.full(self.n_perm, np.iinfo(np.int64).max, dtype=np.int64)
        for item in items:
            for i in range(self.n_perm):
                sig[i] = min(sig[i], _hash(item, i))
        return sig

    @staticmethod
    def jaccard(sig_a, sig_b):
        """Estimated Jaccard = fraction of signature slots that agree."""
        return float(np.mean(sig_a == sig_b))


__all__ = ["BloomFilter", "CountMinSketch", "HyperLogLog", "ReservoirSampler",
           "TDigest", "MinHash"]
