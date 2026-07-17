"""Ensembles built AROUND the imbalance, not bolted on after.

Resampling (``resample.py``) fixes the data once, then hands it to any learner.
These estimators instead fold the balancing INTO the ensemble: every base model
sees a balanced view, and the ensemble averages away the variance that
aggressive undersampling would otherwise add. That is the whole idea -- you can
undersample the majority hard (cheap, no synthetic points, keeps the boundary
clean) as long as an ensemble votes over many different undersamples so no
majority example is permanently thrown away.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, clone
from ..utils import check_X_y, check_array, check_random_state
from ..preprocessing import LabelEncoder


def _balanced_indices(y_idx, rng):
    """Row indices for a class-balanced subsample: all of the minority, and an
    equal-sized random draw (without replacement) from each other class."""
    classes, counts = np.unique(y_idx, return_counts=True)
    n_min = counts.min()
    picked = []
    for c in classes:
        rows = np.where(y_idx == c)[0]
        picked.append(rng.choice(rows, size=n_min, replace=False))
    return np.concatenate(picked)


class BalancedRandomForest(BaseEstimator, ClassifierMixin):
    """A random forest where every tree trains on a balanced bootstrap.

    THE FIX TO PLAIN BAGGING UNDER IMBALANCE
    ----------------------------------------
    A normal forest bootstraps uniformly, so each tree sees the SAME 0.1%-fraud
    imbalance the whole dataset has, and each tree learns to ignore fraud. Balanced
    random forest instead draws, for each tree, all minority samples plus an
    equal-sized random sample of the majority. Every tree sees a 50/50 problem, so
    every tree actually models the minority -- and because each tree undersamples
    a DIFFERENT majority subset, the forest as a whole still uses all the majority
    data. Balancing without discarding, paid for by the ensemble.

    Chen, Liaw & Breiman (2004).
    """

    def __init__(self, n_estimators=50, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        from ..tree import DecisionTreeClassifier
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        rng = check_random_state(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = _balanced_indices(y_idx, rng)
            # bootstrap within the balanced subsample, as a forest would
            boot = rng.choice(idx, size=len(idx), replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, max_features="sqrt",
                random_state=rng.randint(2 ** 31 - 1))
            tree.fit(X[boot], y_idx[boot])
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        k = len(self.classes_)
        proba = np.zeros((len(X), k))
        for tree in self.estimators_:
            pred = tree.predict(X).astype(int)
            proba[np.arange(len(X)), pred] += 1
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class RUSBoost(BaseEstimator, ClassifierMixin):
    """AdaBoost that random-undersamples the majority before each weak learner.

    THE IDEA
    --------
    AdaBoost reweights hard examples each round; RUSBoost adds one step before
    fitting each weak learner -- randomly undersample the majority to balance the
    round's training set. So boosting focuses on the hard MINORITY cases (via the
    weights) on a balanced sample (via the undersampling), which is far cheaper
    than SMOTEBoost's synthesis and often just as accurate. The AdaBoost weight
    update and final weighted vote are otherwise unchanged.

    Seiffert et al. (2010).
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        from ..tree import DecisionTreeClassifier
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n = len(X)
        rng = check_random_state(self.random_state)
        w = np.full(n, 1.0 / n)
        self.estimators_, self.estimator_weights_ = [], []
        for _ in range(self.n_estimators):
            # balance THIS round, sampling by the boosting weights
            bal = _balanced_indices(y_idx, rng)
            p = w[bal] / w[bal].sum()
            idx = rng.choice(bal, size=len(bal), p=p)
            est = DecisionTreeClassifier(max_depth=1,
                                         random_state=rng.randint(2 ** 31 - 1))
            est.fit(X[idx], y_idx[idx])
            pred = est.predict(X).astype(int)
            err = np.sum(w * (pred != y_idx)) / w.sum()
            if err >= 1.0 - 1.0 / k:
                continue
            err = max(err, 1e-10)
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(k - 1))
            if alpha <= 0:
                break
            w *= np.exp(alpha * (pred != y_idx))
            w /= w.sum()
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
        if not self.estimators_:
            est = DecisionTreeClassifier(max_depth=1).fit(X, y_idx)
            self.estimators_.append(est)
            self.estimator_weights_.append(1.0)
        return self

    def predict(self, X):
        X = check_array(X)
        k = len(self.classes_)
        scores = np.zeros((len(X), k))
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X).astype(int)
            scores[np.arange(len(X)), pred] += alpha
        return self.classes_[scores.argmax(axis=1)]


class EasyEnsemble(BaseEstimator, ClassifierMixin):
    """Bag several INDEPENDENT balanced learners over different undersamples.

    THE IDEA
    --------
    Undersampling the majority once throws away most of it. EasyEnsemble trains
    ``n_estimators`` separate classifiers, each on all the minority plus a
    DIFFERENT random majority undersample, and averages them. Across the whole
    ensemble almost every majority example is seen by some member, so no data is
    permanently lost -- while each member still faces a balanced, cheap-to-train
    problem. The simplest of the balanced ensembles, and a strong baseline.

    Liu, Wu & Zhou (2009).
    """

    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        from ..tree import DecisionTreeClassifier
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        rng = check_random_state(self.random_state)
        base = (self.estimator if self.estimator is not None
                else DecisionTreeClassifier(max_depth=None))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = _balanced_indices(y_idx, rng)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=rng.randint(2 ** 31 - 1))
            est.fit(X[idx], y_idx[idx])
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        k = len(self.classes_)
        proba = np.zeros((len(X), k))
        for est in self.estimators_:
            pred = est.predict(X).astype(int)
            proba[np.arange(len(X)), pred] += 1
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["BalancedRandomForest", "RUSBoost", "EasyEnsemble"]
