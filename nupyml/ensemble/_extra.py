"""AdaBoost for regression, and trees used as a feature map."""
import numpy as np

from ..base import (BaseEstimator, RegressorMixin, TransformerMixin, clone,
                    check_is_fitted)
from ..tree import DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state


class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost.R2: reweight toward the samples the ensemble predicts worst.

    HOW REGRESSION BREAKS THE ORIGINAL IDEA
    ---------------------------------------
    Classification AdaBoost reweights by whether each sample was WRONG -- a
    yes-or-no question. Regression has no such question: every prediction is
    wrong by some amount, so "misclassified" does not exist and the weight update
    has nothing to key off.

    AdaBoost.R2's answer is to normalise the errors by the worst one::

        loss_i = |y_i - pred_i| / max_j |y_j - pred_j|      in [0, 1]

    That turns "how wrong" into a relative badness, recovering something the
    exponential update can use. Every design choice below follows from this one
    substitution.

    THE COST OF THAT CHOICE
    -----------------------
    The normalisation is by the MAXIMUM, so a single outlier sets the scale for
    everyone. One absurd point makes every other sample's loss look tiny by
    comparison, the weights barely move, and boosting stalls. AdaBoost is already
    outlier-sensitive in classification; here the sensitivity is structural.

    ``loss="square"`` or ``"exponential"`` change how sharply relative error
    translates to weight, but they cannot fix the normalisation -- for genuinely
    noisy data, gradient boosting with a Huber loss is the better tool, and
    knowing why is worth more than the parameter.

    PREDICTION IS A WEIGHTED MEDIAN
    -------------------------------
    Not a weighted mean, which is the surprise. Each learner votes with weight
    ``log(1/beta)``, and the ensemble takes the weighted MEDIAN of their
    predictions -- the value where half the vote-weight lies on either side. The
    median is robust: one learner predicting a wild value cannot drag the answer,
    whereas a mean would let it. Given how much this method already concedes to
    outliers, refusing to concede here too is the sensible defence.

    Drucker (1997).
    """

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 loss="linear", random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def _loss(self, error):
        """Relative badness in [0, 1] -- see the class docstring."""
        max_error = error.max()
        if max_error <= 0:
            return np.zeros_like(error)
        scaled = error / max_error
        if self.loss == "linear":
            return scaled
        if self.loss == "square":
            return scaled ** 2
        if self.loss == "exponential":
            return 1.0 - np.exp(-scaled)
        raise ValueError(f"Unknown loss: {self.loss!r}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)

        base = self.estimator if self.estimator is not None \
            else DecisionTreeRegressor(max_depth=3)

        w = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for _ in range(self.n_estimators):
            # AdaBoost.R2 resamples according to the weights rather than passing
            # them to the learner: it must work with any regressor, and not every
            # regressor accepts sample_weight
            idx = rng.choice(n, size=n, replace=True, p=w)
            est = clone(base)
            est.fit(X[idx], y[idx])
            pred = est.predict(X)

            loss = self._loss(np.abs(y - pred))
            avg_loss = float((w * loss).sum())

            # a learner no better than a coin flip on this weighting carries no
            # information; continuing would give it a negative vote, which for
            # regression means nothing (unlike classification, where the opposite
            # of a wrong answer is informative)
            if avg_loss >= 0.5:
                if not self.estimators_:
                    # keep one, or there is no model at all
                    self.estimators_.append(est)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(avg_loss)
                break

            beta = avg_loss / (1.0 - avg_loss)
            self.estimators_.append(est)
            self.estimator_weights_.append(
                self.learning_rate * np.log(1.0 / max(beta, 1e-10)))
            self.estimator_errors_.append(avg_loss)

            # well-predicted samples get their weight multiplied down; the
            # exponent means a learner that did well overall (small beta) makes
            # a bigger distinction between its easy and hard samples
            w = w * np.power(beta, (1.0 - loss) * self.learning_rate)
            total = w.sum()
            if total <= 0:
                break
            w /= total

        self.estimator_weights_ = np.array(self.estimator_weights_)
        return self

    def predict(self, X):
        """The weighted median of the learners' predictions."""
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        preds = np.column_stack([e.predict(X) for e in self.estimators_])
        weights = self.estimator_weights_

        if len(self.estimators_) == 1:
            return preds[:, 0]

        # per sample: sort the learners by their prediction, then walk the
        # cumulative vote-weight until half of it is behind us. Sorting the
        # weights alongside the predictions is what makes it a WEIGHTED median
        order = np.argsort(preds, axis=1)
        sorted_pred = np.take_along_axis(preds, order, axis=1)
        sorted_w = weights[order]
        cdf = np.cumsum(sorted_w, axis=1)
        median_idx = np.argmax(cdf >= 0.5 * cdf[:, -1:], axis=1)
        return sorted_pred[np.arange(len(X)), median_idx]


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


__all__ = ["AdaBoostRegressor", "RandomTreesEmbedding"]
