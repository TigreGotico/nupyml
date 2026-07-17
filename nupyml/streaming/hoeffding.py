"""Hoeffding trees: growing a decision tree from a stream, one example at a time.

THE PROBLEM
-----------
A normal decision tree needs the whole dataset in memory to pick each split -- it
sorts every feature at every node. On an unbounded stream that is impossible: you
cannot store the data, and you cannot wait for it to end.

THE KEY INSIGHT
---------------
You do not need all the data to pick a split -- only enough to be CONFIDENT which
feature is best. The Hoeffding bound says: after ``n`` examples, the observed best
split's advantage over the runner-up is within ``epsilon`` of the true advantage,
where ``epsilon`` shrinks as ``sqrt(ln(1/delta) / 2n)``. So split a node the
moment the observed gap between the top two candidate splits EXCEEDS ``epsilon`` --
at that point more data cannot change the winner, with probability ``1 - delta``.

The consequence is remarkable: a Hoeffding tree provably converges to the SAME
tree a batch learner would build on the full data, while looking at each example
ONCE and keeping only per-node counts. Statistics stand in for storage.

THE TIE CASE
------------
When two splits are genuinely almost equal, the gap never exceeds ``epsilon`` and
the node would wait forever. A ``tie_threshold`` forces a split once ``epsilon``
itself is tiny -- if the two candidates are within a hair, it does not matter
which you pick, so pick one and move on. Without it the tree stalls on ties.

Domingos & Hulten (2000).
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..utils import check_array


class _HoeffdingNode:
    """A leaf that accumulates per-feature, per-value, per-class counts until it
    is confident enough to split, then becomes an internal node."""

    __slots__ = ("class_counts", "feature_stats", "n_seen", "feature",
                 "threshold", "left", "right", "n_bins")

    def __init__(self, n_classes, n_features, n_bins):
        self.class_counts = np.zeros(n_classes)
        # counts[feature][bin][class] -- the sufficient statistics for a split,
        # which is all a leaf needs to keep instead of the data itself
        self.feature_stats = np.zeros((n_features, n_bins, n_classes))
        self.n_seen = 0
        self.feature = -1
        self.threshold = None
        self.left = self.right = None
        self.n_bins = n_bins

    def is_leaf(self):
        return self.left is None


class HoeffdingTree(BaseEstimator, ClassifierMixin):
    """A streaming decision tree (a.k.a. VFDT -- Very Fast Decision Tree).

    ``feature_ranges`` bins each numeric feature into ``n_bins`` so that split
    candidates can be tallied with fixed memory; without known ranges it adapts
    them from the first examples seen.
    """

    def __init__(self, n_classes=2, delta=1e-6, tie_threshold=0.05,
                 grace_period=50, n_bins=10, feature_ranges=None):
        self.n_classes = n_classes
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.grace_period = grace_period       # min examples between split checks
        self.n_bins = n_bins
        self.feature_ranges = feature_ranges

    def _bin(self, x):
        lo, hi = self._ranges[:, 0], self._ranges[:, 1]
        span = np.where(hi > lo, hi - lo, 1.0)
        b = ((x - lo) / span * self.n_bins).astype(int)
        return np.clip(b, 0, self.n_bins - 1)

    def _entropy(self, counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts[counts > 0] / total
        return -(p * np.log2(p)).sum()

    def _best_two_splits(self, node):
        """Best BINARY split per feature; return the top two features' gains.

        For each feature, scan every bin boundary as a candidate threshold and
        keep the one whose two-way split gains most information -- a cumulative
        sweep exactly like the batch tree's, done over bin counts rather than
        sorted values. The Hoeffding test then compares the BEST feature against
        the SECOND best: if the leader's margin is real, more data will not
        overturn it, so only these two matter.

        Records the winning bin per feature so the node can split at the RIGHT
        threshold, not the range midpoint.
        """
        parent_entropy = self._entropy(node.class_counts)
        n_features = node.feature_stats.shape[0]
        gains = np.zeros(n_features)
        self._best_bin = np.zeros(n_features, dtype=int)

        for f in range(n_features):
            stats = node.feature_stats[f]        # (bins, classes)
            total = stats.sum()
            if total == 0:
                continue
            # cumulative class counts up to each boundary = the "left" side of a
            # binary split; the right side is the parent total minus it
            cum = np.cumsum(stats, axis=0)
            best_gain, best_b = 0.0, 0
            for b in range(self.n_bins - 1):
                left = cum[b]
                right = node.class_counts - left
                nl, nr = left.sum(), right.sum()
                if nl == 0 or nr == 0:
                    continue
                child = (nl * self._entropy(left) + nr * self._entropy(right)) / total
                gain = parent_entropy - child
                if gain > best_gain:
                    best_gain, best_b = gain, b
            gains[f] = best_gain
            self._best_bin[f] = best_b

        order = np.argsort(gains)[::-1]
        best = order[0]
        second_gain = gains[order[1]] if len(order) > 1 else 0.0
        return best, gains[best], second_gain

    def _hoeffding_bound(self, n):
        """epsilon: how close the observed gain gap is to the true one after n
        examples. Shrinks as 1/sqrt(n), so more data means a tighter bound."""
        R = np.log2(self.n_classes)              # range of the information gain
        return np.sqrt(R ** 2 * np.log(1 / self.delta) / (2 * n))

    def partial_fit(self, x, y):
        x = np.asarray(x, float).ravel()
        if not hasattr(self, "root_"):
            n_features = len(x)
            if self.feature_ranges is not None:
                self._ranges = np.asarray(self.feature_ranges, float)
            else:
                # seed ranges from the first example; widened as more arrive
                self._ranges = np.column_stack([x - 1, x + 1])
            self.root_ = _HoeffdingNode(self.n_classes, n_features, self.n_bins)
            self.classes_ = np.arange(self.n_classes)

        # adapt ranges to keep binning meaningful as the stream reveals its spread
        self._ranges[:, 0] = np.minimum(self._ranges[:, 0], x)
        self._ranges[:, 1] = np.maximum(self._ranges[:, 1], x)

        node = self._leaf_for(x)
        bins = self._bin(x)
        node.class_counts[y] += 1
        node.n_seen += 1
        for f in range(len(x)):
            node.feature_stats[f, bins[f], y] += 1

        # only test for a split every grace_period examples -- checking every
        # example would waste time recomputing gains that barely moved
        if node.n_seen % self.grace_period == 0:
            self._attempt_split(node)
        return self

    def _leaf_for(self, x):
        node = self.root_
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _attempt_split(self, node):
        best, best_gain, second_gain = self._best_two_splits(node)
        eps = self._hoeffding_bound(node.n_seen)
        # split when the leader's margin is provably real, OR when the two are so
        # close (eps below the tie threshold) that it does not matter which wins
        if best_gain > 0 and (best_gain - second_gain > eps or eps < self.tie_threshold):
            node.feature = best
            # threshold at the winning BIN boundary, not the range midpoint --
            # bins are uniform, so boundary b sits at this fraction of the range
            lo, hi = self._ranges[best]
            node.threshold = lo + (hi - lo) * (self._best_bin[best] + 1) / self.n_bins
            nf, nb = node.feature_stats.shape[0], self.n_bins
            node.left = _HoeffdingNode(self.n_classes, nf, nb)
            node.right = _HoeffdingNode(self.n_classes, nf, nb)

    def fit(self, X, y):
        X, y = check_array(X), np.asarray(y, int)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def predict(self, X):
        check_is_fitted(self, "root_")
        X = check_array(X)
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            leaf = self._leaf_for(x)
            # majority class at the reached leaf; the counts ARE the model
            out[i] = int(np.argmax(leaf.class_counts)) \
                if leaf.class_counts.sum() > 0 else 0
        return out

    def n_nodes(self):
        def count(node):
            return 1 if node.is_leaf() else 1 + count(node.left) + count(node.right)
        return count(self.root_)


__all__ = ["HoeffdingTree"]
