"""Embed EVERYTHING into one space and compare by similarity (Wu et al., 2018)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class StarSpace(BaseEstimator):
    """Embed EVERYTHING into one space and compare by similarity (Wu et al., 2018).

    A single model for many tasks: represent both an entity (a bag of features --
    words, tags) and a label as vectors in the SAME space, and train so a
    positive (entity, label) pair scores higher than a sampled negative by a margin
    (a ranking loss). Classification becomes "nearest label in the shared space";
    the same recipe does retrieval, recommendation, or entity similarity just by
    changing what the pairs are. ``fit(bags, labels)`` where each bag is a list of
    feature ids; ``predict`` returns the nearest label.
    """

    def __init__(self, dim=32, epochs=20, lr=0.05, margin=0.1, n_negative=5,
                 random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.n_negative = n_negative
        self.random_state = random_state

    def _bag_vec(self, bag):
        if not bag:
            return np.zeros(self.dim)
        return self.feat_[list(bag)].mean(axis=0)

    def fit(self, bags, labels):
        rng = check_random_state(self.random_state)
        n_feat = max(f for b in bags for f in b) + 1
        self.classes_ = sorted(set(labels))
        self.lidx_ = {c: k for k, c in enumerate(self.classes_)}
        self.feat_ = rng.normal(0, 0.1, (n_feat, self.dim))
        self.label_ = rng.normal(0, 0.1, (len(self.classes_), self.dim))
        y = [self.lidx_[l] for l in labels]
        for _ in range(self.epochs):
            order = rng.permutation(len(bags))
            for i in order:
                v = self._bag_vec(bags[i])
                pos = y[i]
                for _ in range(self.n_negative):
                    neg = rng.randint(len(self.classes_))
                    if neg == pos:
                        continue
                    s_pos = v @ self.label_[pos]
                    s_neg = v @ self.label_[neg]
                    if s_pos - s_neg < self.margin:    # margin ranking violation
                        self.label_[pos] += self.lr * v
                        self.label_[neg] -= self.lr * v
                        for f in bags[i]:
                            self.feat_[f] += self.lr * (self.label_[pos] - self.label_[neg]) / len(bags[i])
        return self

    def predict(self, bags):
        out = []
        for b in bags:
            v = self._bag_vec(b)
            out.append(self.classes_[int(np.argmax(self.label_ @ v))])
        return np.array(out)


__all__ = ["StarSpace"]
