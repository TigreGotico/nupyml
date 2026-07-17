"""Decision trees: CART (Classification And Regression Trees).

THE MODEL
---------
A tree asks a sequence of yes/no questions about single features
(``is x[3] <= 2.7?``) and reads a constant off the leaf you land in. That is
all a tree is: a piecewise-constant function whose pieces are axis-aligned
boxes.

Its properties follow directly from that shape:

* No scaling needed. Only the ORDER of a feature's values matters, so any
  monotone transform of a feature -- log, standardise, anything -- leaves the
  tree unchanged. Almost no other model can say that.
* Interactions come free. A split inside a split is a conjunction, so trees
  express "if A and B" without being told to look for it.
* Diagonal boundaries are awkward. A 45-degree line has to be approximated by
  a staircase of axis-aligned cuts.
* Left alone, a tree fits the training data perfectly and generalises poorly:
  keep splitting and every leaf ends up holding one sample. Everything in
  ``max_depth``, ``min_samples_leaf`` and ``ccp_alpha`` exists to stop that,
  and the ensembles in ``nupyml.ensemble`` exist because averaging many trees
  works even better.

HOW IT IS BUILT
---------------
Finding the optimal tree is NP-complete, so CART is greedy: at each node take
the single best split available now, and never reconsider. That is a real
compromise -- a split that looks poor alone may be excellent in combination --
but it is what makes fitting fast, and in practice it works.

"Best" means the split that most reduces IMPURITY -- a measure of how mixed a
node's labels are:

* Gini ``1 - sum(p_k^2)``: the chance of misclassifying a sample labelled by
  drawing from the node's own class distribution.
* Entropy ``-sum(p_k log p_k)``: expected bits needed to encode a label.
* For regression, variance.

Gini and entropy rarely disagree; Gini avoids a logarithm.

WHY THE SPLIT SEARCH IS FAST
----------------------------
Naively, scoring one threshold means partitioning the node and summing each
side: O(n) per threshold, O(n^2) per feature. The trick is that thresholds are
NESTED -- sort by the feature, and moving the threshold right by one position
just moves one sample from right to left. So a single cumulative sum over the
sorted labels gives the class counts of the left side at EVERY threshold at
once, and the right side is the total minus the left. One sort, one cumsum,
and all n-1 candidate splits are scored: O(n log n) per feature, dominated by
the sort.
"""
import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                    check_is_fitted)
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state


class _Node:
    """One node. Leaves have ``left is None`` and carry a prediction in ``value``.

    ``impurity``, ``n_samples`` and ``weight`` are kept after fitting because
    pruning and feature importances need to re-examine the tree without the
    training data.
    """

    __slots__ = ("feature", "threshold", "left", "right", "value", "impurity",
                 "n_samples", "weight", "missing_left")

    def __init__(self, value, impurity, n_samples, weight):
        self.feature = -1
        self.threshold = 0.0
        self.missing_left = True
        self.left = None
        self.right = None
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples
        self.weight = weight

    @property
    def is_leaf(self):
        return self.left is None


def _best_split_classification(X, y_onehot, w, feature_indices, criterion,
                               min_leaf, monotonic_cst=None):
    """Find the split that most reduces impurity.

    THE CUMSUM TRICK
    ----------------
    Labels are one-hot, so a cumulative sum down the sorted rows gives, at row
    i, the class counts of everything at or left of i -- that is, the left
    child's class distribution for the threshold sitting there. Every
    threshold, in one vectorised pass. The right child needs no work at all:
    it is the node total minus the left.

    Only positions where the feature VALUE changes are real candidates: you
    cannot separate two samples that agree on the feature, so ``np.diff``
    filters the rest out. The threshold is placed midway between the two
    values, which is the standard convention -- any point between them
    separates the same samples, and the midpoint is the most defensible choice
    for unseen data.

    MISSING VALUES
    --------------
    NaNs are not imputed. Both destinations are scored and the better one is
    kept, so if missingness is informative the tree exploits it, and if it is
    not, the NaNs simply follow the bulk. That decision is stored on the node
    so prediction routes new NaNs the same way.

    Returns ``(feature, threshold, gain, send_missing_left)``; feature is -1
    when no admissible split exists.
    """
    n, k = y_onehot.shape
    wy = y_onehot * w[:, None]
    total = wy.sum(axis=0)
    total_w = w.sum()

    def impurity(counts, sizes):
        p = counts / sizes[:, None]
        if criterion == "gini":
            return 1.0 - (p ** 2).sum(axis=1)
        logp = np.log2(np.where(p > 0, p, 1.0))
        return -(p * logp).sum(axis=1)

    parent_imp = impurity(total[None, :], np.array([total_w]))[0]
    best = (-1, 0.0, 0.0, True)
    for j in feature_indices:
        col = X[:, j]
        nan_mask = np.isnan(col)
        obs = ~nan_mask
        if not obs.any():
            continue
        nan_wy = wy[nan_mask].sum(axis=0)
        nan_w = w[nan_mask].sum()
        Xo, wyo, wo = col[obs], wy[obs], w[obs]
        order = np.argsort(Xo, kind="stable")
        xs = Xo[order]
        cum = np.cumsum(wyo[order], axis=0)
        cum_w = np.cumsum(wo[order])
        obs_total = cum[-1]
        obs_total_w = cum_w[-1]
        distinct = np.nonzero(np.diff(xs))[0]
        if len(distinct) == 0:
            continue
        n_obs = len(xs)
        for missing_left in ((True,) if not nan_mask.any() else (True, False)):
            add_counts = nan_wy if missing_left else 0.0
            add_w = nan_w if missing_left else 0.0
            left_counts = cum[distinct] + add_counts
            left_sizes = cum_w[distinct] + add_w
            right_counts = total - left_counts
            right_sizes = total_w - left_sizes
            n_left = distinct + 1 + (nan_mask.sum() if missing_left else 0)
            n_right = n - n_left
            ok = ((left_sizes > 0) & (right_sizes > 0)
                  & (n_left >= min_leaf) & (n_right >= min_leaf))
            if not ok.any():
                continue
            imp = (left_sizes[ok] * impurity(left_counts[ok], left_sizes[ok])
                   + right_sizes[ok] * impurity(right_counts[ok],
                                                right_sizes[ok])) / total_w
            valid_pos = distinct[ok]
            i = int(np.argmin(imp))
            gain = parent_imp - imp[i]
            if gain <= best[2] + 1e-12:
                continue
            p = valid_pos[i]
            thr = 0.5 * (xs[p] + xs[p + 1])
            if monotonic_cst is not None and monotonic_cst[j] != 0:
                # class-1 probability must move with the feature in the
                # constrained direction, else this split is not admissible
                lc, rc = left_counts[ok][i], right_counts[ok][i]
                if len(lc) == 2:
                    p_left = lc[1] / max(lc.sum(), 1e-12)
                    p_right = rc[1] / max(rc.sum(), 1e-12)
                    if monotonic_cst[j] == 1 and p_left > p_right:
                        continue
                    if monotonic_cst[j] == -1 and p_left < p_right:
                        continue
            best = (j, thr, gain, missing_left)
    return best


def _best_split_regression(X, y, w, feature_indices, min_leaf,
                           monotonic_cst=None):
    """The same scan, with variance as the impurity.

    Variance needs the mean of each side at every threshold, which would seem
    to need a second pass. It does not: use the identity::

        var = E[y^2] - E[y]^2

    Running sums of ``y`` and ``y^2`` are both cumsums, so both moments are
    available at every threshold from a single pass -- the same structure as
    the classification case, with two accumulators instead of k.

    (This identity is numerically delicate in general: when the mean is huge
    relative to the spread, it subtracts two nearly-equal numbers. Within a
    node of a tree the range is bounded and it is not a problem in practice.)
    """
    n = len(y)
    total_w = w.sum()
    total_sum = (w * y).sum()
    total_sq = (w * y ** 2).sum()
    parent_imp = total_sq / total_w - (total_sum / total_w) ** 2
    best = (-1, 0.0, 0.0, True)
    for j in feature_indices:
        col = X[:, j]
        nan_mask = np.isnan(col)
        obs = ~nan_mask
        if not obs.any():
            continue
        nan_w = w[nan_mask].sum()
        nan_sum = (w[nan_mask] * y[nan_mask]).sum()
        nan_sq = (w[nan_mask] * y[nan_mask] ** 2).sum()
        Xo, yo, wo = col[obs], y[obs], w[obs]
        order = np.argsort(Xo, kind="stable")
        xs, ys, ws = Xo[order], yo[order], wo[order]
        cum_w = np.cumsum(ws)
        cum_sum = np.cumsum(ws * ys)
        cum_sq = np.cumsum(ws * ys ** 2)
        distinct = np.nonzero(np.diff(xs))[0]
        if len(distinct) == 0:
            continue
        for missing_left in ((True,) if not nan_mask.any() else (True, False)):
            aw = nan_w if missing_left else 0.0
            asum = nan_sum if missing_left else 0.0
            asq = nan_sq if missing_left else 0.0
            nl = cum_w[distinct] + aw
            nr = total_w - nl
            n_left = distinct + 1 + (nan_mask.sum() if missing_left else 0)
            n_right = n - n_left
            ok = ((nl > 0) & (nr > 0) & (n_left >= min_leaf)
                  & (n_right >= min_leaf))
            if not ok.any():
                continue
            sl = cum_sum[distinct] + asum
            sr = total_sum - sl
            ql = cum_sq[distinct] + asq
            qr = total_sq - ql
            nlk, nrk = nl[ok], nr[ok]
            slk, srk, qlk, qrk = sl[ok], sr[ok], ql[ok], qr[ok]
            var_l = qlk / nlk - (slk / nlk) ** 2
            var_r = qrk / nrk - (srk / nrk) ** 2
            imp = (nlk * var_l + nrk * var_r) / total_w
            i = int(np.argmin(imp))
            gain = parent_imp - imp[i]
            if gain <= best[2] + 1e-12:
                continue
            valid_pos = distinct[ok]
            p = valid_pos[i]
            thr = 0.5 * (xs[p] + xs[p + 1])
            if monotonic_cst is not None and monotonic_cst[j] != 0:
                # leaf means must move with the feature in the required
                # direction, else the split would break monotonicity
                mean_l, mean_r = slk[i] / nlk[i], srk[i] / nrk[i]
                if monotonic_cst[j] == 1 and mean_l > mean_r:
                    continue
                if monotonic_cst[j] == -1 and mean_l < mean_r:
                    continue
            best = (j, thr, gain, missing_left)
    return best


class _BaseDecisionTree(BaseEstimator):
    def __init__(self, criterion, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0,
                 ccp_alpha=0.0, monotonic_cst=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
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

    @staticmethod
    def _node_score(value, is_classification):
        """The quantity a monotonic constraint orders: the leaf mean for
        regression, the class-1 probability for binary classification."""
        if not is_classification:
            return value
        total = value.sum()
        return value[1] / total if total > 0 else 0.5

    def _grow(self, X, y_enc, w, depth, rng, is_classification,
              bounds=(-np.inf, np.inf)):
        """Recursively split, returning the subtree rooted here.

        ``bounds`` carries monotonic constraints down the tree. Checking a
        split's own two children is not enough: a descendant could still
        double back and break monotonicity globally. So a constrained split
        divides the allowed value range at the midpoint of its children, and
        each child inherits a narrower interval it can never escape. Leaf
        values are clipped into it -- which is what makes the guarantee hold
        for the whole tree rather than one split at a time.
        """
        n, d = X.shape
        wsum = w.sum()
        lower, upper = bounds
        if is_classification:
            counts = (y_enc * w[:, None]).sum(axis=0)
            value = counts
            p = counts / wsum
            if self.criterion == "entropy":
                imp = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            else:
                imp = float(1.0 - (p ** 2).sum())
        else:
            mean = (w * y_enc).sum() / wsum
            value = float(mean)
            imp = float((w * (y_enc - mean) ** 2).sum() / wsum)
        # a monotonic constraint is only honoured globally if every leaf value
        # is clipped into the bounds inherited from its ancestors
        if self._monotonic_cst is not None and np.isfinite([lower, upper]).any():
            if is_classification:
                total = value.sum()
                p1 = value[1] / total if total > 0 else 0.5
                p1 = float(np.clip(p1, lower, upper))
                value = np.array([(1 - p1) * total, p1 * total])
            else:
                value = float(np.clip(value, lower, upper))
        node = _Node(value, imp, n, wsum)
        if (n < self.min_samples_split or imp <= 1e-12
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        k = self._n_features_to_try(d)
        features = rng.choice(d, size=k, replace=False) if k < d else np.arange(d)
        cst = self._monotonic_cst
        if is_classification:
            j, thr, gain, missing_left = _best_split_classification(
                X, y_enc, w, features, self.criterion, self.min_samples_leaf,
                cst)
        else:
            j, thr, gain, missing_left = _best_split_regression(
                X, y_enc, w, features, self.min_samples_leaf, cst)
        if j < 0 or gain * wsum / self._w_total < self.min_impurity_decrease + 1e-15:
            return node
        col = X[:, j]
        nan_mask = np.isnan(col)
        mask = np.where(nan_mask, missing_left, col <= thr)
        if not mask.any() or mask.all():
            return node
        node.feature = j
        node.threshold = thr
        node.missing_left = missing_left
        left_bounds = right_bounds = (lower, upper)
        if cst is not None and cst[j] != 0:
            # split the value interval at the midpoint of the children so no
            # descendant can cross back over its sibling
            def side_score(sub_y, sub_w):
                if is_classification:
                    counts = (sub_y * sub_w[:, None]).sum(axis=0)
                    t = counts.sum()
                    return counts[1] / t if t > 0 else 0.5
                return (sub_w * sub_y).sum() / sub_w.sum()

            m_l = side_score(y_enc[mask], w[mask])
            m_r = side_score(y_enc[~mask], w[~mask])
            middle = (m_l + m_r) / 2
            if cst[j] == 1:
                left_bounds = (lower, min(upper, middle))
                right_bounds = (max(lower, middle), upper)
            else:
                left_bounds = (max(lower, middle), upper)
                right_bounds = (lower, min(upper, middle))
        node.left = self._grow(X[mask], y_enc[mask], w[mask], depth + 1, rng,
                               is_classification, left_bounds)
        node.right = self._grow(X[~mask], y_enc[~mask], w[~mask], depth + 1,
                                rng, is_classification, right_bounds)
        return node

    def _predict_values(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, force_all_finite="allow-nan")
        out = np.empty((len(X),) + np.shape(self.tree_.value), dtype=np.float64)
        # iterative traversal over batches per node
        stack = [(self.tree_, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] = node.value
                continue
            col = X[idx, node.feature]
            mask = np.where(np.isnan(col), node.missing_left,
                            col <= node.threshold)
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    @property
    def feature_importances_(self):
        """How much each feature reduced impurity, normalised to sum to 1.

        Each split is credited with its impurity drop, weighted by how many
        samples reached it -- a split near the root affects everything and
        counts accordingly.

        This measure is biased and worth distrusting: it favours
        high-cardinality features, because a feature with many distinct values
        offers more thresholds and so more chances to fit noise. It also
        arbitrarily splits credit between correlated features. It is computed
        from the training data alone, so it says what the tree USED, not what
        actually predicts. ``permutation_importance`` measures the latter.
        """
        check_is_fitted(self, "tree_")
        imp = np.zeros(self.n_features_in_)

        def walk(node):
            if node.is_leaf:
                return
            decrease = (node.weight * node.impurity
                        - node.left.weight * node.left.impurity
                        - node.right.weight * node.right.impurity)
            imp[node.feature] += decrease
            walk(node.left)
            walk(node.right)

        walk(self.tree_)
        total = imp.sum()
        return imp / total if total > 0 else imp

    def _prune_ccp(self, ccp_alpha):
        """Cost-complexity pruning: grow greedily, then cut back.

        WHY PRUNE INSTEAD OF STOPPING EARLY
        -----------------------------------
        Stopping when a split looks unhelpful is short-sighted: a weak split
        can enable an excellent one beneath it, and early stopping never finds
        out. So CART grows the tree out fully and then removes what did not
        earn its keep -- a decision made with the whole subtree visible.

        THE CRITERION
        -------------
        Score a tree by error plus a charge per leaf::

            R_alpha(T) = R(T) + alpha * |leaves(T)|

        ``alpha`` is the price of a leaf. At alpha = 0 the full tree wins. As
        alpha rises, subtrees that bought little accuracy stop being worth
        their leaves, and collapse.

        For each internal node, the alpha at which its subtree stops paying for
        itself is::

            alpha_eff = (error if collapsed - error of subtree) / (leaves - 1)

        The "weakest link" is the node with the smallest such value. Collapse
        it, repeat, and stop once every remaining alpha_eff exceeds
        ``ccp_alpha``. This yields a nested sequence of trees, which is what
        makes ``cost_complexity_pruning_path`` a well-defined object to
        cross-validate over.
        """
        def subtree_stats(node):
            """(total weighted impurity of leaves, leaf count) below node."""
            if node.is_leaf:
                return node.weight * node.impurity, 1
            lr, ln = subtree_stats(node.left)
            rr, rn = subtree_stats(node.right)
            return lr + rr, ln + rn

        while True:
            weakest, weakest_alpha = None, np.inf

            def visit(node):
                nonlocal weakest, weakest_alpha
                if node.is_leaf:
                    return
                r_sub, n_leaves = subtree_stats(node)
                if n_leaves > 1:
                    alpha = ((node.weight * node.impurity - r_sub)
                             / (self._w_total * (n_leaves - 1)))
                    if alpha < weakest_alpha:
                        weakest, weakest_alpha = node, alpha
                visit(node.left)
                visit(node.right)

            visit(self.tree_)
            if weakest is None or weakest_alpha > ccp_alpha:
                break
            weakest.left = weakest.right = None
            weakest.feature = -1

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """The ccp_alpha values at which the tree structure changes."""
        from types import SimpleNamespace
        full = clone(self).set_params(ccp_alpha=0.0).fit(X, y)
        alphas, impurities = [0.0], []

        def total_impurity(node):
            if node.is_leaf:
                return node.weight * node.impurity
            return total_impurity(node.left) + total_impurity(node.right)

        impurities.append(total_impurity(full.tree_) / full._w_total)
        while not full.tree_.is_leaf:
            weakest, weakest_alpha = None, np.inf

            def subtree_stats(node):
                if node.is_leaf:
                    return node.weight * node.impurity, 1
                lr, ln = subtree_stats(node.left)
                rr, rn = subtree_stats(node.right)
                return lr + rr, ln + rn

            def visit(node):
                nonlocal weakest, weakest_alpha
                if node.is_leaf:
                    return
                r_sub, n_leaves = subtree_stats(node)
                if n_leaves > 1:
                    a = ((node.weight * node.impurity - r_sub)
                         / (full._w_total * (n_leaves - 1)))
                    if a < weakest_alpha:
                        weakest, weakest_alpha = node, a
                visit(node.left)
                visit(node.right)

            visit(full.tree_)
            if weakest is None:
                break
            weakest.left = weakest.right = None
            weakest.feature = -1
            alphas.append(weakest_alpha)
            impurities.append(total_impurity(full.tree_) / full._w_total)
        return SimpleNamespace(ccp_alphas=np.array(alphas),
                               impurities=np.array(impurities))

    def get_depth(self):
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.tree_)

    def get_n_leaves(self):
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return leaves(self.tree_)


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    """A classification tree. Leaves store class counts; ``predict_proba``
    normalises them into frequencies.

    Those probabilities are honest only in the crudest sense: a pure leaf
    reports 1.0 regardless of whether it holds two samples or two hundred.
    Trees are famously badly calibrated for this reason -- see
    ``nupyml.calibration``.
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None,
                 random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, min_impurity_decrease,
                         ccp_alpha, monotonic_cst, random_state)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, force_all_finite="allow-nan")
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        if self.monotonic_cst is not None and len(self.classes_) > 2:
            raise ValueError(
                "monotonic_cst is only defined for binary classification")
        y_onehot = np.eye(len(self.classes_))[self._le.transform(y)]
        self._w_total = w.sum()
        self._monotonic_cst = (None if self.monotonic_cst is None
                               else np.asarray(self.monotonic_cst, dtype=int))
        self.tree_ = self._grow(X, y_onehot, w, 0, rng, True)
        self.n_features_in_ = X.shape[1]
        if self.ccp_alpha > 0:
            self._prune_ccp(self.ccp_alpha)
        return self

    def predict_proba(self, X):
        counts = self._predict_values(X)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._predict_values(X), axis=1)]


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    """A regression tree. Leaves store the mean of their samples.

    Since the output is piecewise constant, a regression tree cannot
    extrapolate: beyond the range of the training data every prediction is the
    value of the nearest boundary leaf, flat forever. A linear model
    extrapolates (perhaps wrongly, but it moves); a tree simply stops.
    """

    def __init__(self, criterion="squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None,
                 random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         min_samples_leaf, max_features, min_impurity_decrease,
                         ccp_alpha, monotonic_cst, random_state)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite="allow-nan")
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        rng = check_random_state(self.random_state)
        self._w_total = w.sum()
        self._monotonic_cst = (None if self.monotonic_cst is None
                               else np.asarray(self.monotonic_cst, dtype=int))
        self.tree_ = self._grow(X, y, w, 0, rng, False)
        self.n_features_in_ = X.shape[1]
        if self.ccp_alpha > 0:
            self._prune_ccp(self.ccp_alpha)
        return self

    def predict(self, X):
        return self._predict_values(X)


__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]
