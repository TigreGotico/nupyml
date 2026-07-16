"""Histogram-based gradient boosting (LightGBM-style, second order)."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state, softmax, sigmoid


class _BinMapper:
    def __init__(self, max_bins=256):
        self.max_bins = max_bins

    def fit(self, X):
        self.bin_edges_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            qs = np.unique(np.percentile(col, np.linspace(0, 100, self.max_bins + 1)[1:-1]))
            self.bin_edges_.append(qs)
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.bin_edges_):
            out[:, j] = np.searchsorted(edges, X[:, j], side="right")
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
    """Single tree grown on binned data with gradient/hessian histograms."""

    def __init__(self, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20,
                 l2=1.0, n_bins=256):
        self.max_depth = max_depth if max_depth is not None else 1 << 30
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.n_bins = n_bins

    def fit(self, X_binned, g, h):
        self.n_features = X_binned.shape[1]
        root_idx = np.arange(len(g))
        self.root = self._leaf(g, root_idx)
        # best-first growth
        candidates = []
        self._try_split(X_binned, g, h, self.root, root_idx, 0, candidates)
        n_leaves = 1
        while candidates and n_leaves < self.max_leaf_nodes:
            candidates.sort(key=lambda c: c[0])
            gain, node, j, b, idx, depth = candidates.pop()
            mask = X_binned[idx, j] <= b
            li, ri = idx[mask], idx[~mask]
            node.feature = j
            node.bin_threshold = b
            node.left = self._leaf(g, li)
            node.right = self._leaf(g, ri)
            n_leaves += 1
            self._try_split(X_binned, g, h, node.left, li, depth + 1, candidates)
            self._try_split(X_binned, g, h, node.right, ri, depth + 1, candidates)
        return self

    def _leaf(self, g, idx):
        return _HistNode(0.0)

    def _try_split(self, Xb, g, h, node, idx, depth, candidates):
        G = g[idx].sum()
        H = h[idx].sum()
        node.value = -G / (H + self.l2)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return
        parent_score = G * G / (H + self.l2)
        best = None
        Xsub = Xb[idx]
        gsub, hsub = g[idx], h[idx]
        for j in range(self.n_features):
            bins = Xsub[:, j]
            gh = np.zeros(self.n_bins)
            hh = np.zeros(self.n_bins)
            ch = np.zeros(self.n_bins)
            np.add.at(gh, bins, gsub)
            np.add.at(hh, bins, hsub)
            np.add.at(ch, bins, 1)
            Gl = np.cumsum(gh)[:-1]
            Hl = np.cumsum(hh)[:-1]
            Cl = np.cumsum(ch)[:-1]
            Gr = G - Gl
            Hr = H - Hl
            Cr = len(idx) - Cl
            valid = (Cl >= self.min_samples_leaf) & (Cr >= self.min_samples_leaf)
            if not valid.any():
                continue
            gain = np.where(
                valid,
                Gl ** 2 / (Hl + self.l2) + Gr ** 2 / (Hr + self.l2) - parent_score,
                -np.inf)
            b = int(np.argmax(gain))
            if gain[b] > 1e-9 and (best is None or gain[b] > best[0]):
                best = (gain[b], node, j, b, idx, depth)
        if best is not None:
            candidates.append(best)

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
            mask = X_binned[idx, node.feature] <= node.bin_threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out


class _BaseHistGB(BaseEstimator):
    def __init__(self, max_iter=100, learning_rate=0.1, max_depth=None,
                 max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                 max_bins=255, early_stopping=False, tol=1e-7, random_state=None):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.early_stopping = early_stopping
        self.tol = tol
        self.random_state = random_state

    def _new_tree(self):
        return _HistTree(max_depth=self.max_depth,
                         max_leaf_nodes=self.max_leaf_nodes,
                         min_samples_leaf=self.min_samples_leaf,
                         l2=self.l2_regularization, n_bins=self.max_bins + 1)


class HistGradientBoostingRegressor(_BaseHistGB, RegressorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self._mapper = _BinMapper(self.max_bins).fit(X)
        Xb = self._mapper.transform(X)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees_ = []
        for _ in range(self.max_iter):
            g = pred - y                      # gradient of 0.5*(pred-y)^2
            h = np.ones(len(y))
            tree = self._new_tree().fit(Xb, g, h)
            pred += self.learning_rate * tree.predict(Xb)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        Xb = self._mapper.transform(check_array(X))
        pred = np.full(len(Xb), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(Xb)
        return pred


class HistGradientBoostingClassifier(_BaseHistGB, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        self._mapper = _BinMapper(self.max_bins).fit(X)
        Xb = self._mapper.transform(X)
        n = len(y_idx)
        if k == 2:
            p0 = np.clip(y_idx.mean(), 1e-12, 1 - 1e-12)
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
                g = p - y_idx
                h = np.maximum(p * (1 - p), 1e-6)
                tree = self._new_tree().fit(Xb, g, h)
                F[:, 0] += self.learning_rate * tree.predict(Xb)
                round_trees.append(tree)
            else:
                P = softmax(F, axis=1)
                for c in range(k):
                    yc = (y_idx == c).astype(float)
                    g = P[:, c] - yc
                    h = np.maximum(P[:, c] * (1 - P[:, c]), 1e-6)
                    tree = self._new_tree().fit(Xb, g, h)
                    F[:, c] += self.learning_rate * tree.predict(Xb)
                    round_trees.append(tree)
            self.trees_.append(round_trees)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "trees_")
        Xb = self._mapper.transform(check_array(X))
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
