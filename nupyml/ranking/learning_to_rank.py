"""Learning to rank: train a scoring function to ORDER items within a query.

THE SHIFT FROM REGRESSION
-------------------------
A ranker is not judged on predicting each relevance value -- only on the ORDER it
induces within each query group (search results for one query, items for one
user). Two rankers with very different scores but the same order are equally
good. So learning-to-rank optimises order directly, most simply through PAIRS:
for two items in the same query with different relevance, the more-relevant one
should score higher.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


def _pairs(y):
    """All (i, j) index pairs within a group where y[i] > y[j]."""
    y = np.asarray(y)
    hi, lo = np.where(y[:, None] > y[None, :])
    return hi, lo


class RankNet(BaseEstimator):
    """Pairwise ranking by the RankNet cross-entropy (Burges et al., 2005).

    THE OBJECTIVE
    -------------
    For every pair (i, j) in a query where item i is more relevant than j, RankNet
    wants ``P(i > j) = sigmoid(s_i - s_j)`` close to 1. Minimising the
    cross-entropy of that over all such pairs trains the scorer ``s = f(x)`` to
    reproduce the correct order. This version uses a LINEAR scorer ``s = x·w`` and
    plain gradient descent on the summed pairwise loss -- the essence of RankNet
    without the neural net, which is a drop-in replacement for ``f``.

    Training data is grouped: ``groups`` lists the size of each query block in
    ``X`` (they must be contiguous), so pairs are formed only WITHIN a query.
    """

    def __init__(self, lr=0.1, n_epochs=100, random_state=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y, groups):
        X = check_array(X)
        y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.coef_ = rng.normal(0, 0.01, X.shape[1])
        for _ in range(self.n_epochs):
            grad = np.zeros_like(self.coef_)
            start = 0
            for g in groups:
                sl = slice(start, start + g)
                start += g
                Xg, yg = X[sl], y[sl]
                hi, lo = _pairs(yg)
                if len(hi) == 0:
                    continue
                s = Xg @ self.coef_
                diff = s[hi] - s[lo]
                # d/ds of -log sigmoid(diff) is -(1 - sigmoid(diff))
                p = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
                w = (p - 1.0)                       # gradient weight per pair
                for (i, j, wij) in zip(hi, lo, w):
                    grad += wij * (Xg[i] - Xg[j])
            self.coef_ -= self.lr * grad / max(1, len(groups))
        return self

    def predict(self, X):
        return check_array(X) @ self.coef_

    def rank(self, X):
        """Indices that order the rows from most to least relevant."""
        return np.argsort(-self.predict(X))


class LambdaMART(BaseEstimator):
    """Gradient-boosted trees driven by pairwise LAMBDA gradients (Burges, 2010).

    THE LAMBDA IDEA
    ---------------
    LambdaMART is MART (gradient-boosted regression trees) whose "gradient" is not
    from a pointwise loss but from RankNet-style pairs -- and, crucially, each
    pair's gradient is SCALED by how much swapping that pair would change a ranking
    metric like NDCG (``|Delta NDCG|``). So the model spends its capacity on the
    swaps that matter most for the metric, near the top of the list, rather than
    on pairs deep down where users never look. Summing those scaled pair gradients
    per document gives a ``lambda`` target that a regression tree fits each round.

    It was the winning approach in learning-to-rank challenges for years. This is
    a compact version: real relevances, NDCG-weighted lambdas, shallow trees.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _lambdas(self, scores, rel):
        """Per-document lambda: sum of NDCG-weighted pair forces (Burges 2010)."""
        n = len(rel)
        lam = np.zeros(n)
        order = np.argsort(-scores)
        rank = np.empty(n, int); rank[order] = np.arange(n)
        discount = 1.0 / np.log2(rank + 2.0)
        gain = (2.0 ** rel - 1)
        idcg = np.sum(np.sort(gain)[::-1] / np.log2(np.arange(2, n + 2)))
        idcg = idcg or 1.0
        hi, lo = _pairs(rel)
        for i, j in zip(hi, lo):
            rho = 1.0 / (1.0 + np.exp(np.clip(scores[i] - scores[j], -30, 30)))
            # |Delta NDCG| from swapping i and j
            dndcg = abs((gain[i] - gain[j]) * (discount[i] - discount[j]) / idcg)
            force = rho * dndcg
            lam[i] += force
            lam[j] -= force
        return lam

    def fit(self, X, y, groups):
        from ..tree import DecisionTreeRegressor
        X = check_array(X)
        y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.trees_ = []
        scores = np.zeros(len(X))
        for _ in range(self.n_estimators):
            lam = np.zeros(len(X))
            start = 0
            for g in groups:
                sl = slice(start, start + g); start += g
                lam[sl] = self._lambdas(scores[sl], y[sl])
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng.randint(2 ** 31 - 1))
            tree.fit(X, lam)                        # fit trees to the lambda forces
            scores += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = check_array(X)
        s = np.zeros(len(X))
        for tree in self.trees_:
            s += self.learning_rate * tree.predict(X)
        return s

    def rank(self, X):
        return np.argsort(-self.predict(X))


__all__ = ["RankNet", "LambdaMART"]
