"""Histogram-based gradient boosting (LightGBM-style, second order)."""
import heapq
import itertools

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state, softmax, sigmoid


class _BinMapper:
    """Quantile binning. NaN is mapped to the last bin, so a split can isolate
    missing values instead of requiring them to be imputed first."""

    def __init__(self, max_bins=256, categorical_features=None):
        self.max_bins = max_bins
        self.categorical_features = categorical_features

    def fit(self, X):
        self.bin_edges_ = []
        self.categories_ = []
        cat = set(self.categorical_features or [])
        self.categorical_ = cat
        for j in range(X.shape[1]):
            col = X[:, j]
            obs = col[~np.isnan(col)]
            if j in cat:
                cats = np.unique(obs)
                if len(cats) > self.max_bins:
                    raise ValueError(
                        f"categorical feature {j} has {len(cats)} categories, "
                        f"more than max_bins={self.max_bins}")
                self.categories_.append(cats)
                self.bin_edges_.append(None)
            else:
                self.categories_.append(None)
                if len(obs) == 0:
                    self.bin_edges_.append(np.array([0.0]))
                    continue
                qs = np.unique(np.percentile(
                    obs, np.linspace(0, 100, self.max_bins + 1)[1:-1]))
                self.bin_edges_.append(qs)
        return self

    @property
    def missing_bin_(self):
        return self.max_bins

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.bin_edges_):
            col = X[:, j]
            nan = np.isnan(col)
            if self.categories_[j] is not None:
                idx = np.searchsorted(self.categories_[j],
                                      np.where(nan, self.categories_[j][0], col))
                out[:, j] = np.clip(idx, 0, self.max_bins - 1)
            else:
                out[:, j] = np.searchsorted(edges, np.where(nan, 0.0, col),
                                            side="right")
            out[nan, j] = self.missing_bin_
        return out


class _HistNode:
    __slots__ = ("value", "feature", "bin_threshold", "left", "right")

    def __init__(self, value):
        self.value = value
        self.feature = -1
        self.bin_threshold = 0
        self.left = None
        self.right = None


class _HistTree:
    """A single regression tree grown on pre-binned data.

    WHY BINNING CHANGES THE GAME
    ----------------------------
    An exact tree must consider every distinct feature value as a candidate
    threshold, which means sorting each feature at each node: O(n log n) per
    feature per node. A histogram tree first buckets each feature into at most
    ``max_bins`` bins (done once, up front) and then only considers bin
    boundaries. Finding the best split becomes a single O(n) pass to fill the
    bins plus an O(bins) scan -- and ``bins`` is a small constant like 256.

    The whole design follows from one observation: to score every threshold on
    a feature you only need, for each bin, the SUM of the gradients, the SUM of
    the hessians, and the COUNT of samples. A running total over bins then
    gives you both sides of every candidate split at once.

    THE TWO OPTIMIZATIONS THAT MATTER
    ---------------------------------
    1. Building histograms with ``np.bincount`` instead of ``np.add.at``
       (see ``_build_histograms``).
    2. The histogram subtraction trick (see ``_split_histograms``).

    Both are documented at their definitions.
    """

    def __init__(self, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20,
                 l2=1.0, n_bins=256):
        self.max_depth = max_depth if max_depth is not None else 1 << 30
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.n_bins = n_bins

    # -- histogram construction ------------------------------------------

    def _build_histograms(self, Xb, idx, g, h):
        """Per-feature, per-bin sums of gradient, hessian and count.

        Returns three ``(n_features, n_bins)`` arrays. This is where a
        histogram tree spends nearly all of its time, so the details pay off:

        ``np.bincount`` rather than ``np.add.at``
            Both perform a scatter-add ("add this value into that bin"), and
            the obvious spelling is ``np.add.at(hist, bins, values)``. But
            ``ufunc.at`` is numpy's *unbuffered* generic path: it exists to
            handle duplicate indices correctly for arbitrary ufuncs, and it
            pays for that generality per element. ``np.bincount`` is a
            specialised C loop doing exactly this one job. On the access
            pattern here -- a strided ``uint8`` index column taken from a
            fancy-indexed block -- the specialised loop measures about 7x
            faster, and histogram building dominates training, so this single
            substitution is most of the tree's speed.

        Gathering ``Xb[idx]`` once
            The alternative, indexing ``Xb[idx, j]`` inside the feature loop,
            re-walks the index array once per feature. One gather up front
            keeps the inner loop reading contiguous memory.
        """
        n_feat = Xb.shape[1]
        hist_g = np.empty((n_feat, self.n_bins))
        hist_h = np.empty((n_feat, self.n_bins))
        hist_c = np.empty((n_feat, self.n_bins))
        Xsub = Xb[idx]
        g_sub, h_sub = g[idx], h[idx]
        for j in range(n_feat):
            col = Xsub[:, j]
            hist_g[j] = np.bincount(col, weights=g_sub,
                                    minlength=self.n_bins)[:self.n_bins]
            hist_h[j] = np.bincount(col, weights=h_sub,
                                    minlength=self.n_bins)[:self.n_bins]
            hist_c[j] = np.bincount(col, minlength=self.n_bins)[:self.n_bins]
        return hist_g, hist_h, hist_c

    def _split_histograms(self, Xb, g, h, parent_hist, left_idx, right_idx):
        """Histograms for both children, building only one of them.

        THE SUBTRACTION TRICK
        ---------------------
        Every sample at a node goes to exactly one child, so for any bin::

            parent_count = left_count + right_count

        and the same holds for the gradient and hessian sums, because sums are
        additive over a partition. So once one child's histogram is built, the
        sibling's is free::

            sibling = parent - child

        That is one array subtraction -- O(n_features * n_bins), independent of
        how many samples the node holds -- instead of another O(n_samples *
        n_features) pass.

        Building the SMALLER child is what makes this pay. The cost of a
        histogram is proportional to the number of samples in it, so building
        the small side and subtracting for the big side means the work per
        tree level is bounded by the smaller half. Summed over a level, no
        matter how the tree is shaped, that is at most half the samples --
        halving the work at every level of the tree.
        """
        if len(left_idx) <= len(right_idx):
            small_idx, small_is_left = left_idx, True
        else:
            small_idx, small_is_left = right_idx, False
        small = self._build_histograms(Xb, small_idx, g, h)
        other = tuple(p - s for p, s in zip(parent_hist, small))
        return (small, other) if small_is_left else (other, small)

    # -- split search ----------------------------------------------------

    def _leaf_value(self, G, H):
        """The value minimising the regularised second-order loss at a leaf.

        Approximating the loss to second order, a leaf holding gradient sum G
        and hessian sum H has loss ``G*w + 0.5*(H + l2)*w**2``, which a little
        calculus minimises at ``w = -G / (H + l2)``. The ``l2`` term is what
        keeps a leaf holding very few samples (tiny H) from taking an
        enormous value.
        """
        return -G / (H + self.l2)

    def _best_split(self, hist_g, hist_h, hist_c, G, H, n_samples):
        """Best (feature, bin) split, scanning every candidate at once.

        A cumulative sum along the bin axis turns "sum of every bin up to b"
        into a single vectorised operation, giving the left side of every
        candidate threshold. The right side is then the parent total minus the
        left -- the same additivity the subtraction trick uses. Because both
        arrays are ``(n_features, n_bins)``, every feature and every threshold
        is scored in one shot, with no Python loop over features.

        The gain of a split is the drop in the regularised objective::

            gain = G_left^2/(H_left+l2) + G_right^2/(H_right+l2)
                   - G_parent^2/(H_parent+l2)

        Returns ``(gain, feature, bin)`` or ``None`` if no split is admissible.
        """
        # cumulative sums along bins: the left side of every threshold
        G_left = np.cumsum(hist_g, axis=1)[:, :-1]
        H_left = np.cumsum(hist_h, axis=1)[:, :-1]
        C_left = np.cumsum(hist_c, axis=1)[:, :-1]
        G_right, H_right = G - G_left, H - H_left
        C_right = n_samples - C_left

        admissible = ((C_left >= self.min_samples_leaf)
                      & (C_right >= self.min_samples_leaf))
        if not admissible.any():
            return None
        parent_score = G * G / (H + self.l2)
        gain = np.where(
            admissible,
            G_left ** 2 / (H_left + self.l2)
            + G_right ** 2 / (H_right + self.l2) - parent_score,
            -np.inf)
        flat = int(gain.argmax())
        j, b = np.unravel_index(flat, gain.shape)
        if not np.isfinite(gain[j, b]) or gain[j, b] <= 1e-9:
            return None
        return float(gain[j, b]), int(j), int(b)

    # -- growth ----------------------------------------------------------

    def fit(self, X_binned, g, h):
        """Grow the tree best-first: always split whichever node gains most.

        Best-first (rather than depth-first) growth is what makes
        ``max_leaf_nodes`` a meaningful budget -- each leaf spent goes to the
        split that helps most, wherever it sits in the tree.
        """
        n, self.n_features = X_binned.shape
        root_idx = np.arange(n)
        root_hist = self._build_histograms(X_binned, root_idx, g, h)
        self.root = _HistNode(0.0)
        # a counter breaks gain ties so heapq never compares the payloads
        tie = itertools.count()
        frontier = []
        self._consider(X_binned, g, h, self.root, root_idx, 0, root_hist,
                       frontier, tie)
        n_leaves = 1
        while frontier and n_leaves < self.max_leaf_nodes:
            neg_gain, _, node, j, b, idx, depth, hist = heapq.heappop(frontier)
            goes_left = X_binned[idx, j] <= b
            left_idx, right_idx = idx[goes_left], idx[~goes_left]
            node.feature, node.bin_threshold = j, b
            node.left = _HistNode(0.0)
            node.right = _HistNode(0.0)
            left_hist, right_hist = self._split_histograms(
                X_binned, g, h, hist, left_idx, right_idx)
            n_leaves += 1
            self._consider(X_binned, g, h, node.left, left_idx, depth + 1,
                           left_hist, frontier, tie)
            self._consider(X_binned, g, h, node.right, right_idx, depth + 1,
                           right_hist, frontier, tie)
        return self

    def _consider(self, Xb, g, h, node, idx, depth, hist, frontier, tie):
        """Set this node's leaf value, and queue its best split if it has one."""
        G = float(hist[0][0].sum())
        H = float(hist[1][0].sum())
        node.value = self._leaf_value(G, H)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return
        found = self._best_split(hist[0], hist[1], hist[2], G, H, len(idx))
        if found is None:
            return
        gain, j, b = found
        # negated: heapq is a min-heap, and we always want the largest gain
        heapq.heappush(frontier,
                       (-gain, next(tie), node, j, b, idx, depth, hist))

    def predict(self, X_binned):
        out = np.empty(len(X_binned))
        stack = [(self.root, np.arange(len(X_binned)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.left is None:
                out[idx] = node.value
                continue
            goes_left = X_binned[idx, node.feature] <= node.bin_threshold
            stack.append((node.left, idx[goes_left]))
            stack.append((node.right, idx[~goes_left]))
        return out


class _BaseHistGB(BaseEstimator):
    def __init__(self, max_iter=100, learning_rate=0.1, max_depth=None,
                 max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                 max_bins=255, categorical_features=None, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
                 random_state=None):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    def _maybe_split(self, X, y, w):
        """Hold out a validation slice when early_stopping is enabled."""
        if not self.early_stopping:
            return X, y, w, None, None, None
        rng = check_random_state(self.random_state)
        n_val = max(1, int(self.validation_fraction * len(X)))
        perm = rng.permutation(len(X))
        val, tr = perm[:n_val], perm[n_val:]
        return X[tr], y[tr], w[tr], X[val], y[val], w[val]

    def _new_tree(self):
        # +1 for the dedicated missing-value bin
        return _HistTree(max_depth=self.max_depth,
                         max_leaf_nodes=self.max_leaf_nodes,
                         min_samples_leaf=self.min_samples_leaf,
                         l2=self.l2_regularization, n_bins=self.max_bins + 1)


class HistGradientBoostingRegressor(_BaseHistGB, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite="allow-nan")
        w = np.ones(len(y)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        X, y, w, X_val, y_val, w_val = self._maybe_split(X, y, w)
        self._mapper = _BinMapper(
            self.max_bins, getattr(self, "categorical_features", None)).fit(X)
        Xb = self._mapper.transform(X)
        self.init_ = float(np.average(y, weights=w))
        pred = np.full(len(y), self.init_)
        self.trees_ = []
        self.validation_score_ = []
        best, stall = np.inf, 0
        for _ in range(self.max_iter):
            g = w * (pred - y)                # weighted gradient of 0.5*(pred-y)^2
            h = w.copy()
            tree = self._new_tree().fit(Xb, g, h)
            pred += self.learning_rate * tree.predict(Xb)
            self.trees_.append(tree)
            if X_val is not None:
                mse = float(np.average((self.predict(X_val) - y_val) ** 2,
                                       weights=w_val))
                self.validation_score_.append(mse)
                if mse < best - self.tol:
                    best, stall = mse, 0
                else:
                    stall += 1
                    if stall >= self.n_iter_no_change:
                        break
        self.n_iter_ = len(self.trees_)
        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        Xb = self._mapper.transform(
            check_array(X, force_all_finite="allow-nan"))
        pred = np.full(len(Xb), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(Xb)
        return pred


class HistGradientBoostingClassifier(_BaseHistGB, ClassifierMixin):
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, force_all_finite="allow-nan")
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        w = np.ones(len(y)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        k = len(self.classes_)
        self._mapper = _BinMapper(
            self.max_bins, getattr(self, "categorical_features", None)).fit(X)
        Xb = self._mapper.transform(X)
        n = len(y_idx)
        if k == 2:
            p0 = np.clip(np.average(y_idx, weights=w), 1e-12, 1 - 1e-12)
            self.init_ = np.array([np.log(p0 / (1 - p0))])
            F = np.full((n, 1), self.init_[0])
        else:
            prior = np.clip(np.bincount(y_idx, minlength=k) / n, 1e-12, None)
            self.init_ = np.log(prior)
            F = np.tile(self.init_, (n, 1))
        self.trees_ = []
        for _ in range(self.max_iter):
            round_trees = []
            if k == 2:
                p = sigmoid(F[:, 0])
                g = w * (p - y_idx)
                h = w * np.maximum(p * (1 - p), 1e-6)
                tree = self._new_tree().fit(Xb, g, h)
                F[:, 0] += self.learning_rate * tree.predict(Xb)
                round_trees.append(tree)
            else:
                P = softmax(F, axis=1)
                for c in range(k):
                    yc = (y_idx == c).astype(float)
                    g = w * (P[:, c] - yc)
                    h = w * np.maximum(P[:, c] * (1 - P[:, c]), 1e-6)
                    tree = self._new_tree().fit(Xb, g, h)
                    F[:, c] += self.learning_rate * tree.predict(Xb)
                    round_trees.append(tree)
            self.trees_.append(round_trees)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "trees_")
        Xb = self._mapper.transform(
            check_array(X, force_all_finite="allow-nan"))
        k = len(self.classes_)
        n_out = 1 if k == 2 else k
        F = np.tile(self.init_, (len(Xb), 1)) if n_out > 1 else \
            np.full((len(Xb), 1), self.init_[0])
        for round_trees in self.trees_:
            for c, tree in enumerate(round_trees):
                F[:, c] += self.learning_rate * tree.predict(Xb)
        return F.ravel() if n_out == 1 else F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if F.ndim == 1:
            p = sigmoid(F)
            return np.column_stack([1 - p, p])
        return softmax(F, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


__all__ = ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor"]
