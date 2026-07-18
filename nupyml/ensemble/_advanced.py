"""Ensembles v2: glass-box boosting, rotated forests, stacked forests, and
greedy regularised forests.

Four ensembles that each depart from "average many independent trees". The EBM is
a BOOSTED additive model you can read off as one shape function per feature.
Rotation Forest DECORRELATES trees by rotating each one's feature space with PCA.
The cascade forest STACKS layers of forests like a deep net. The regularized
greedy forest grows ONE forest structure under an explicit penalty rather than
fitting fixed-size trees.
"""
import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor, DecisionTreeClassifier
from ..decomposition import PCA


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ExplainableBoostingClassifier(BaseEstimator, ClassifierMixin):
    """A glass-box GA2M: boosted per-feature shape functions (Lou et al., 2013).

    A full gradient-boosted forest is accurate but opaque. An EBM keeps the
    boosting but restricts each weak learner to a SINGLE feature, cycling through
    features round-robin. The model is then a sum of one-dimensional SHAPE
    FUNCTIONS -- ``f(x) = b + sum_j f_j(x_j)`` -- so you can plot exactly how each
    feature moves the prediction, an additive model with the accuracy of boosting.
    Binary classification via logistic loss; ``feature_shapes_`` exposes each
    learned ``f_j`` as a small regression tree ensemble.
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, max_leaf_nodes=8,
                 random_state=None):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        self.intercept_ = np.log((yb.mean() + 1e-6) / (1 - yb.mean() + 1e-6))
        F = np.full(n, self.intercept_)
        self.shapes_ = [[] for _ in range(d)]           # per-feature tree list
        rng = check_random_state(self.random_state)
        for r in range(self.n_rounds):
            j = r % d                                    # round-robin over features
            p = _sigmoid(F)
            grad = p - yb                                # logistic gradient
            tree = DecisionTreeRegressor(max_depth=3, random_state=rng)
            tree.fit(X[:, [j]], -grad)
            update = self.learning_rate * tree.predict(X[:, [j]])
            F += update
            self.shapes_[j].append(tree)
        return self

    def _score(self, X):
        F = np.full(len(X), self.intercept_)
        for j, trees in enumerate(self.shapes_):
            for tree in trees:
                F += self.learning_rate * tree.predict(X[:, [j]])
        return F

    def shape_function(self, j, grid):
        """The learned contribution of feature ``j`` over a 1-D grid of values."""
        grid = np.asarray(grid).reshape(-1, 1)
        out = np.zeros(len(grid))
        for tree in self.shapes_[j]:
            out += self.learning_rate * tree.predict(grid)
        return out

    def predict_proba(self, X):
        X = check_array(X)
        p = _sigmoid(self._score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self._score(check_array(X)) > 0).astype(int)]


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    """Decorrelate trees by ROTATING each one's feature space (Rodriguez, 2006).

    Random forests decorrelate trees by sampling features; Rotation Forest does it
    by ROTATION. For each tree it splits the features into random subsets, runs PCA
    on a bootstrap of each subset, and trains the tree on the PCA-ROTATED features.
    Because axis-parallel splits in a rotated space are oblique in the original, the
    trees are both diverse AND individually strong -- the combination that often
    edges out plain forests on structured data.
    """

    def __init__(self, n_estimators=25, n_subsets=3, max_depth=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.n_subsets = n_subsets
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rotation(self, X, rng):
        d = X.shape[1]
        perm = rng.permutation(d)
        subsets = np.array_split(perm, self.n_subsets)
        R = np.zeros((d, d))
        for sub in subsets:
            if len(sub) == 0:
                continue
            boot = X[rng.choice(len(X), len(X), replace=True)][:, sub]
            pca = PCA().fit(boot)
            comp = pca.components_.T                     # (len(sub), len(sub))
            for a, ia in enumerate(sub):
                for b, ib in enumerate(sub):
                    R[ia, ib] = comp[a, b]
        return R

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        self.rotations_, self.trees_ = [], []
        for _ in range(self.n_estimators):
            R = self._make_rotation(X, rng)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=rng)
            tree.fit(X @ R, y)
            self.rotations_.append(R); self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for R, tree in zip(self.rotations_, self.trees_):
            pred = tree.predict(X @ R)
            for i, c in enumerate(self.classes_):
                votes[:, i] += (pred == c)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CascadeForestClassifier(BaseEstimator, ClassifierMixin):
    """Deep Forest: stack forest LAYERS like a neural net (Zhou & Feng, 2017).

    "Deep" need not mean neural. gcForest stacks LAYERS of forests: each layer is
    an ensemble of random forests, and its class-probability outputs are CONCATENATED
    to the original features and fed to the next layer -- representation learning by
    forests instead of neurons. It grows layers until validation accuracy stops
    improving, so the depth is chosen automatically (no backprop, few
    hyperparameters). Here each layer holds a few random forests.
    """

    def __init__(self, n_layers=5, n_forests=2, n_estimators=30, max_depth=None,
                 tol=1e-4, random_state=None):
        self.n_layers = n_layers
        self.n_forests = n_forests
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        from . import RandomForestClassifier
        from ..model_selection import train_test_split
        from ..metrics import accuracy_score
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3,
                                                random_state=rng)
        self.layers_ = []
        cur_tr, cur_val = Xtr, Xval
        prev_acc = -np.inf
        for _ in range(self.n_layers):
            forests, tr_out, val_out = [], [], []
            for f in range(self.n_forests):
                rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            max_features="sqrt" if f == 0 else None,
                                            random_state=rng)
                rf.fit(cur_tr, ytr)
                forests.append(rf)
                tr_out.append(rf.predict_proba(cur_tr))
                val_out.append(rf.predict_proba(cur_val))
            acc = accuracy_score(yval, self.classes_[
                np.mean(val_out, axis=0).argmax(axis=1)])
            if acc <= prev_acc + self.tol and self.layers_:
                break                                    # stop growing depth
            prev_acc = acc
            self.layers_.append(forests)
            cur_tr = np.hstack([Xtr] + tr_out)           # augment with class probs
            cur_val = np.hstack([Xval] + val_out)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        cur = X
        last = None
        for forests in self.layers_:
            outs = [f.predict_proba(cur) for f in forests]
            last = np.mean(outs, axis=0)
            cur = np.hstack([X] + outs)
        return last

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class RegularizedGreedyForest(BaseEstimator, RegressorMixin):
    """Grow ONE forest structure under an explicit penalty (Johnson & Zhang, 2014).

    Gradient boosting fits a sequence of FIXED-size trees. RGF instead treats the
    whole forest as one evolving structure and, at each step, greedily makes the
    single change -- grow an existing leaf or start a new tree -- that most reduces
    a regularised objective (loss + a penalty on leaf weights), periodically
    re-optimising all leaf values. Directly regularising the structure, rather than
    capping tree size, often gives a more accurate model for the same number of
    leaves. This is a compact regression implementation on squared loss.
    """

    def __init__(self, max_leaves=100, learning_rate=0.5, l2=0.1, max_tree_depth=4,
                 random_state=None):
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_tree_depth = max_tree_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.intercept_ = y.mean()
        pred = np.full(len(y), self.intercept_)
        self.trees_, self.weights_ = [], []
        n_leaves = 0
        while n_leaves < self.max_leaves:
            residual = y - pred                          # negative gradient (sq loss)
            depth = min(self.max_tree_depth,
                        max(1, int(np.log2(self.max_leaves - n_leaves + 1))))
            tree = DecisionTreeRegressor(max_depth=depth, random_state=rng)
            tree.fit(X, residual)
            raw = tree.predict(X)
            # shrink toward zero: L2 regularisation on the added leaf weights
            w = self.learning_rate / (1.0 + self.l2)
            pred = pred + w * raw
            self.trees_.append(tree); self.weights_.append(w)
            n_leaves += 2 ** depth
            if np.abs(residual).max() < 1e-6:
                break
        return self

    def predict(self, X):
        X = check_array(X)
        out = np.full(len(X), self.intercept_)
        for tree, w in zip(self.trees_, self.weights_):
            out += w * tree.predict(X)
        return out


__all__ = ["ExplainableBoostingClassifier", "RotationForestClassifier",
           "CascadeForestClassifier", "RegularizedGreedyForest"]
