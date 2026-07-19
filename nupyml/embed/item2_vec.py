"""word2vec's skip-gram applied to ITEM sets (Barkan & Koenigstein, 2016)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Item2Vec(BaseEstimator):
    """word2vec's skip-gram applied to ITEM sets (Barkan & Koenigstein, 2016).

    Recommendation data is baskets/sessions -- SETS of co-purchased or co-played
    items with no natural order. Item2vec drops word2vec's window and treats every
    pair of items in a basket as a (target, context) pair, training skip-gram with
    negative sampling. Items that co-occur end up with similar vectors, so nearest
    neighbours in the embedding are "bought together" recommendations -- learned
    from co-occurrence alone, no ratings. ``most_similar`` returns the nearest
    items.
    """

    def __init__(self, dim=32, epochs=20, lr=0.025, n_negative=5, random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def fit(self, baskets):
        rng = check_random_state(self.random_state)
        items = sorted({i for b in baskets for i in b})
        self.idx_ = {it: k for k, it in enumerate(items)}
        self.items_ = items
        n = len(items)
        self.W_ = rng.normal(0, 0.1, (n, self.dim))
        self.C_ = rng.normal(0, 0.1, (n, self.dim))
        freq = np.array([sum(b.count(it) for b in baskets) for it in items], float)
        neg_p = freq ** 0.75; neg_p /= neg_p.sum()
        pairs = [(self.idx_[a], self.idx_[c]) for b in baskets
                 for a in b for c in b if a != c]
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for t, c in pairs:
                negs = rng.choice(n, self.n_negative, p=neg_p)
                self._sgns_step(t, c, negs)
        self.embedding_ = self.W_
        return self

    def _sgns_step(self, t, c, negs):
        lr = self.lr
        # positive pair
        g = (_sigmoid(self.W_[t] @ self.C_[c]) - 1.0)
        grad_t = g * self.C_[c]
        self.C_[c] -= lr * g * self.W_[t]
        for k in negs:                                 # negative samples
            gk = _sigmoid(self.W_[t] @ self.C_[k])
            grad_t += gk * self.C_[k]
            self.C_[k] -= lr * gk * self.W_[t]
        self.W_[t] -= lr * grad_t

    def most_similar(self, item, k=5):
        v = self.W_[self.idx_[item]]
        sims = self.W_ @ v / (np.linalg.norm(self.W_, axis=1) * np.linalg.norm(v) + 1e-9)
        order = np.argsort(-sims)
        return [self.items_[j] for j in order if self.items_[j] != item][:k]


__all__ = ["Item2Vec"]
