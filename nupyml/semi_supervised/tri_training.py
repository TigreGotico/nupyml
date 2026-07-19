"""Three classifiers; two agreeing teach the third (Zhou & Li, 2005)."""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class TriTraining(BaseEstimator, ClassifierMixin):
    """Three classifiers; two agreeing teach the third (Zhou & Li, 2005).

    Tri-training drops co-training's need for two independent views. It trains
    THREE classifiers on bootstrap samples of the labelled data; an unlabeled point
    is pseudo-labelled for a classifier only when the OTHER TWO AGREE on it. The
    majority-agreement rule is a cheap confidence proxy -- if two independent-ish
    models concur, the label is probably right -- and the final prediction is the
    three-way vote. Needs no feature split, so it applies where co-training cannot.
    """

    def __init__(self, estimator=None, n_iter=10, random_state=None):
        self.estimator = estimator
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        from ..linear_model import LogisticRegression
        X = check_array(X)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        base = self.estimator or LogisticRegression(max_iter=300)
        lab = y != -1
        Xl, yl = X[lab], y[lab]
        Xu = X[~lab]
        self.classes_ = np.unique(yl)
        n = len(yl)
        clfs = []
        for _ in range(3):                             # each on its own bootstrap
            idx = rng.choice(n, n, replace=True)
            clfs.append(clone(base).fit(Xl[idx], yl[idx]))
        for _ in range(self.n_iter):
            if len(Xu) == 0:
                break
            for i in range(3):
                j, k = [m for m in range(3) if m != i]
                pj, pk = clfs[j].predict(Xu), clfs[k].predict(Xu)
                agree = pj == pk                       # the other two concur
                if agree.sum() == 0:
                    continue
                Xaug = np.vstack([Xl, Xu[agree]])
                yaug = np.concatenate([yl, pj[agree]])
                clfs[i] = clone(base).fit(Xaug, yaug)
        self.clfs_ = clfs
        return self

    def predict(self, X):
        check_is_fitted(self, "clfs_")
        X = check_array(X)
        votes = np.array([c.predict(X) for c in self.clfs_])
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for i in range(len(X)):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            out[i] = vals[counts.argmax()]             # majority vote
        return out


__all__ = ["TriTraining"]
