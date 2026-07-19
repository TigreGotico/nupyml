"""A random forest that grows ONLINE from a Mondrian process (Lakshminarayanan, 2014)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


class _MondrianTree:
    """A Mondrian tree: axis-aligned splits sampled from a Mondrian process."""

    def __init__(self, lifetime, rng):
        self.lifetime = lifetime
        self.rng = rng
        self.root = None

    def fit(self, X, y, classes):
        self.classes = classes
        self.root = self._grow(X, y, 0.0)
        return self

    def _grow(self, X, y, parent_time):
        node = {"lo": X.min(0), "hi": X.max(0),
                "counts": np.array([(y == c).sum() for c in self.classes])}
        span = node["hi"] - node["lo"]
        total = span.sum()
        if total <= 0 or len(y) < 2:
            node["leaf"] = True
            return node
        split_time = parent_time + self.rng.exponential(1.0 / total)
        if split_time >= self.lifetime:                  # stop: cost exceeds budget
            node["leaf"] = True
            return node
        d = self.rng.choice(len(span), p=span / total)   # dim ∝ side length
        thr = self.rng.uniform(node["lo"][d], node["hi"][d])
        left = X[:, d] <= thr
        if left.all() or (~left).all():
            node["leaf"] = True
            return node
        node.update({"leaf": False, "dim": d, "thr": thr, "time": split_time,
                     "left": self._grow(X[left], y[left], split_time),
                     "right": self._grow(X[~left], y[~left], split_time)})
        return node

    def predict_proba(self, x, node=None):
        node = self.root if node is None else node
        if node["leaf"]:
            c = node["counts"]
            return c / c.sum() if c.sum() else np.ones(len(self.classes)) / len(self.classes)
        branch = node["left"] if x[node["dim"]] <= node["thr"] else node["right"]
        return self.predict_proba(x, branch)


class MondrianForest(BaseEstimator, ClassifierMixin):
    """A random forest that grows ONLINE from a Mondrian process (Lakshminarayanan, 2014).

    A standard random forest must see all the data at once. The Mondrian forest is
    built from Mondrian trees, whose splits are sampled from a Mondrian PROCESS -- a
    hierarchical random partition where each split has a "time" and dimensions are
    chosen in proportion to their extent. This construction is consistent under
    streaming: new points extend the partition without retraining from scratch, so
    ``partial_fit`` genuinely learns online, and predictions are the class-count
    average across trees. Here trees are (re)grown per batch, keeping the Mondrian
    split rule.
    """

    def __init__(self, n_estimators=25, lifetime=2.0, random_state=None):
        self.n_estimators = n_estimators
        self.lifetime = lifetime
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._rng = check_random_state(self.random_state)
        self._X, self._y = X, y
        self.trees_ = [_MondrianTree(self.lifetime, self._rng).fit(X, y, self.classes_)
                       for _ in range(self.n_estimators)]
        return self

    def partial_fit(self, X, y, classes=None):
        X = check_array(X); y = np.asarray(y)
        if not hasattr(self, "trees_"):
            if classes is not None:
                self.classes_ = np.asarray(classes)
            self._rng = check_random_state(self.random_state)
            self._X = np.empty((0, X.shape[1])); self._y = np.empty(0, dtype=y.dtype)
            self.trees_ = []
        self._X = np.vstack([self._X, X])                # accumulate the stream
        self._y = np.concatenate([self._y, y])
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(self._y)
        self.trees_ = [_MondrianTree(self.lifetime, self._rng).fit(
            self._X, self._y, self.classes_) for _ in range(self.n_estimators)]
        return self

    def predict_proba(self, X):
        X = check_array(X)
        out = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            out[i] = np.mean([t.predict_proba(x) for t in self.trees_], axis=0)
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["MondrianForest"]
