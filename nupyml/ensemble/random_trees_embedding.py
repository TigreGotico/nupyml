"""Use a forest of totally random trees as a feature map."""
import numpy as np
from ..base import (BaseEstimator, RegressorMixin, TransformerMixin, clone,
                    check_is_fitted)
from ..tree import DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state


class RandomTreesEmbedding(BaseEstimator, TransformerMixin):
    """Use a forest of totally random trees as a feature map.

    THE IDEA
    --------
    Fit trees that split on RANDOM features at RANDOM thresholds, ignoring ``y``
    entirely -- this is unsupervised. Then encode each sample by WHICH LEAF it
    lands in, one-hot, in every tree.

    The output is a high-dimensional sparse binary code. Its meaning: two samples
    share a 1 exactly when some random partition of the space put them in the same
    cell. Agreeing across many random partitions means being genuinely close, so
    the code is a learned NEIGHBOURHOOD encoding.

    WHY RANDOM SPLITS ARE NOT A COMPROMISE
    --------------------------------------
    There is no ``y``, so there is no criterion to optimize -- randomness is not
    a shortcut here, it is the only option. And it is enough: what matters is that
    the partitions are DIVERSE, not that any single one is good.

    WHAT IT IS FOR
    --------------
    The code is a non-linear transformation into a space where a LINEAR model can
    work. Trees carve axis-aligned regions; one-hot encoding which region a point
    is in gives a linear model the power to fit a different constant per region.
    It is the same manoeuvre as the kernel trick -- go somewhere high-dimensional
    where the problem is linear -- but the map is learned from the data's actual
    structure rather than fixed in advance.

    The classic use is trees feeding a linear model, which was how ad click
    prediction was done at scale for years: the forest finds the interactions,
    the linear model stays cheap to update.
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_leaf=1,
                 sparse_output=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sparse_output = sparse_output
        self.random_state = random_state

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)

        # the trees are fit against NOISE, which is the trick that makes them
        # "totally random": with a meaningless target, the split criterion has
        # nothing to prefer, so every split is effectively arbitrary
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, max_features=1,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(np.iinfo(np.int32).max))
            tree.fit(X, rng.uniform(size=n))
            self.estimators_.append(tree)

        self._leaf_maps = [self._number_leaves(tree) for tree in self.estimators_]
        self.n_features_out_ = sum(len(m) for m in self._leaf_maps)
        return self.transform(X)

    @staticmethod
    def _number_leaves(tree):
        """Give every leaf a stable index, by depth-first left-to-right order.

        The obvious shortcut -- keying leaves by ``id(node)`` -- silently makes
        the output depend on MEMORY ADDRESSES. The embedding still separates the
        data, so it looks fine, but the columns come out in whatever order the
        allocator happened to produce: two runs with the same seed disagree, and
        column 7 means nothing in particular. Traversal order is a property of
        the tree, which is what the column index should reflect.
        """
        leaves = {}
        stack = [tree.tree_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves[id(node)] = len(leaves)
            else:
                # right first, so left is popped first and indices ascend
                # left-to-right
                stack.append(node.right)
                stack.append(node.left)
        return leaves

    def _apply(self, tree, X):
        """Which leaf each sample reaches, as that leaf's stable index."""
        leaf_map = self._number_leaves(tree)
        out = np.empty(len(X), dtype=np.int64)
        for i, row in enumerate(X):
            node = tree.tree_
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = leaf_map[id(node)]
        return out

    def transform(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        blocks = []
        for tree, leaf_map in zip(self.estimators_, self._leaf_maps):
            block = np.zeros((len(X), len(leaf_map)))
            block[np.arange(len(X)), self._apply(tree, X)] = 1.0
            blocks.append(block)
        dense = np.hstack(blocks)
        if self.sparse_output:
            from scipy.sparse import csr_matrix
            return csr_matrix(dense)
        return dense


__all__ = ["RandomTreesEmbedding"]
