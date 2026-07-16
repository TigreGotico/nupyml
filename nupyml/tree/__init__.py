"""Decision trees (CART) with vectorized split search."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "impurity",
                 "n_samples")

    def __init__(self, value, impurity, n_samples):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.left is None


def _best_split_classification(X, y_onehot, feature_indices, criterion, min_leaf):
    """Vectorized best split: for each feature, sort once and score every
    threshold via cumulative class counts. Returns (feature, threshold, gain)."""
    n, k = y_onehot.shape
    total = y_onehot.sum(axis=0)

    def impurity(counts, sizes):
        # counts: (m, k), sizes: (m,)
        p = counts / sizes[:, None]
        if criterion == "gini":
            return 1.0 - (p ** 2).sum(axis=1)
        logp = np.log2(np.where(p > 0, p, 1.0))
        return -(p * logp).sum(axis=1)

    parent_imp = impurity(total[None, :], np.array([n]))[0]
    best = (-1, 0.0, 0.0)
    for j in feature_indices:
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        ys = y_onehot[order]
        cum = np.cumsum(ys, axis=0)         # (n, k)
        # candidate split positions: between distinct consecutive values
        distinct = np.nonzero(np.diff(xs))[0]
        if len(distinct) == 0:
            continue
        pos = distinct[(distinct + 1 >= min_leaf) & (n - distinct - 1 >= min_leaf)]
        if len(pos) == 0:
            continue
        left_counts = cum[pos]
        left_sizes = (pos + 1).astype(float)
        right_counts = total - left_counts
        right_sizes = n - left_sizes
        imp = (left_sizes * impurity(left_counts, left_sizes)
               + right_sizes * impurity(right_counts, right_sizes)) / n
        i = np.argmin(imp)
        gain = parent_imp - imp[i]
        if gain > best[2] + 1e-12:
            thr = 0.5 * (xs[pos[i]] + xs[pos[i] + 1])
            best = (j, thr, gain)
    return best


def _best_split_regression(X, y, feature_indices, min_leaf):
    n = len(y)
    total_sum = y.sum()
    total_sq = (y ** 2).sum()
    parent_imp = total_sq / n - (total_sum / n) ** 2
    best = (-1, 0.0, 0.0)
    for j in feature_indices:
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        ys = y[order]
        cum_sum = np.cumsum(ys)
        cum_sq = np.cumsum(ys ** 2)
        distinct = np.nonzero(np.diff(xs))[0]
        if len(distinct) == 0:
            continue
        pos = distinct[(distinct + 1 >= min_leaf) & (n - distinct - 1 >= min_leaf)]
        if len(pos) == 0:
            continue
        nl = (pos + 1).astype(float)
        nr = n - nl
        sl, sr = cum_sum[pos], total_sum - cum_sum[pos]
        ql, qr = cum_sq[pos], total_sq - cum_sq[pos]
        var_l = ql / nl - (sl / nl) ** 2
        var_r = qr / nr - (sr / nr) ** 2
        imp = (nl * var_l + nr * var_r) / n
        i = np.argmin(imp)
        gain = parent_imp - imp[i]
        if gain > best[2] + 1e-12:
            thr = 0.5 * (xs[pos[i]] + xs[pos[i] + 1])
            best = (j, thr, gain)
    return best


class _BaseDecisionTree(BaseEstimator):
    def __init__(self, criterion, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0,
                 random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(int(mf), d)

    def _grow(self, X, y_enc, depth, rng, is_classification):
        n, d = X.shape
        if is_classification:
            counts = y_enc.sum(axis=0)
            value = counts
            p = counts / n
            if self.criterion == "entropy":
                imp = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            else:
                imp = float(1.0 - (p ** 2).sum())
        else:
            value = float(y_enc.mean())
            imp = float(y_enc.var())
        node = _Node(value, imp, n)
        if (n < self.min_samples_split or imp <= 1e-12
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        k = self._n_features_to_try(d)
        features = rng.choice(d, size=k, replace=False) if k < d else np.arange(d)
        if is_classification:
            j, thr, gain = _best_split_classification(
                X, y_enc, features, self.criterion, self.min_samples_leaf)
        else:
            j, thr, gain = _best_split_regression(
                X, y_enc, features, self.min_samples_leaf)
        if j < 0 or gain * n / self._n_total < self.min_impurity_decrease + 1e-15:
            return node
        mask = X[:, j] <= thr
        node.feature = j
        node.threshold = thr
        node.left = self._grow(X[mask], y_enc[mask], depth + 1, rng, is_classification)
        node.right = self._grow(X[~mask], y_enc[~mask], depth + 1, rng, is_classification)
        return node

    def _predict_values(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X)
        out = np.empty((len(X),) + np.shape(self.tree_.value), dtype=np.float64)
        # iterative traversal over batches per node
        stack = [(self.tree_, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] = node.value
                continue
            mask = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    def get_depth(self):
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.tree_)

    def get_n_leaves(self):
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return leaves(self.tree_)


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None,
                 min_impurity_decrease=0.0, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, min_impurity_decrease,
                         random_state)

    def fit(self, X, y, sample_indices=None):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_onehot = np.eye(len(self.classes_))[self._le.transform(y)]
        self._n_total = len(X)
        self.tree_ = self._grow(X, y_onehot, 0, rng, True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        counts = self._predict_values(X)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._predict_values(X), axis=1)]


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def __init__(self, criterion="squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 min_impurity_decrease=0.0, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, min_impurity_decrease,
                         random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        self._n_total = len(X)
        self.tree_ = self._grow(X, y, 0, rng, False)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._predict_values(X)


__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]
