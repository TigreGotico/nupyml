"""Co-training, tri-training, and positive-unlabeled learning.

These join label propagation / spreading / self-training. Co- and tri-training
use MULTIPLE learners that teach each other from unlabeled data; PU learning
tackles the case where you only have POSITIVE and UNLABELED examples -- no labelled
negatives at all. Unlabeled points are marked ``-1`` (co/tri-training); PU takes a
0/1 "labeled-positive" indicator.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state
from ..preprocessing import LabelEncoder


class CoTraining(BaseEstimator, ClassifierMixin):
    """Two classifiers on two feature VIEWS teach each other (Blum & Mitchell, 1998).

    THE IDEA
    --------
    Split the features into two views (e.g. a web page's text vs its inbound-link
    text). Train one classifier per view on the labelled data; then each
    classifier LABELS the unlabeled points it is most confident about and hands
    those pseudo-labels to the OTHER classifier's training set. Because the two
    views are (assumed) conditionally independent given the label, one view's
    confident mistakes are usually the other view's easy cases, so they correct
    each other -- extracting signal from unlabeled data that neither could alone.
    """

    def __init__(self, estimator=None, n_iter=20, top_k=5, random_state=None):
        self.estimator = estimator
        self.n_iter = n_iter
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        from ..linear_model import LogisticRegression
        X = check_array(X)
        y = np.asarray(y).copy()
        base = self.estimator or LogisticRegression(max_iter=300)
        d = X.shape[1]
        self.v1_, self.v2_ = np.arange(d // 2), np.arange(d // 2, d)
        lab = y != -1
        L = np.where(lab)[0].tolist()
        U = np.where(~lab)[0].tolist()
        self.classes_ = np.unique(y[lab])
        for _ in range(self.n_iter):
            if not U:
                break
            c1 = clone(base).fit(X[np.ix_(L, self.v1_)], y[L])
            c2 = clone(base).fit(X[np.ix_(L, self.v2_)], y[L])
            added = []
            for c, view in [(c1, self.v1_), (c2, self.v2_)]:
                p = c.predict_proba(X[np.ix_(U, view)])
                conf = p.max(axis=1)
                for j in np.argsort(-conf)[:self.top_k]:
                    y[U[j]] = c.classes_[p[j].argmax()]   # pseudo-label
                    added.append(U[j])
            for a in set(added):
                if a in U:
                    U.remove(a); L.append(a)
        self.c1_ = clone(base).fit(X[np.ix_(L, self.v1_)], y[L])
        self.c2_ = clone(base).fit(X[np.ix_(L, self.v2_)], y[L])
        return self

    def predict(self, X):
        check_is_fitted(self, "c1_")
        X = check_array(X)
        p = (self.c1_.predict_proba(X[:, self.v1_])
             + self.c2_.predict_proba(X[:, self.v2_])) / 2   # average the views
        return self.c1_.classes_[p.argmax(axis=1)]


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


class PUClassifier(BaseEstimator, ClassifierMixin):
    """Positive-Unlabeled learning by the Elkan-Noto method (2008).

    THE SETTING
    -----------
    You have some POSITIVE examples and a pile of UNLABELED ones -- no confirmed
    negatives (fraud you caught vs everything else; genes known to interact vs the
    rest). Training positive-vs-unlabeled naively is biased, because the unlabeled
    set contains hidden positives.

    THE ELKAN-NOTO CORRECTION
    -------------------------
    Under the "selected completely at random" assumption, a classifier ``g`` trained
    to predict LABELED-vs-unlabeled estimates ``c * P(y=1|x)`` where ``c =
    P(labeled | positive)`` is a constant. Estimate ``c`` as the average ``g`` on
    held-out KNOWN positives, then divide it out: ``P(y=1|x) = g(x) / c``. So a
    single calibration constant turns a biased PU classifier into an unbiased
    posterior. ``fit(X, s)`` with ``s=1`` for labelled-positive, ``s=0`` for
    unlabeled.
    """

    def __init__(self, estimator=None, calibration_fraction=0.3, random_state=None):
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, s):
        from ..linear_model import LogisticRegression
        X = check_array(X)
        s = np.asarray(s)
        rng = check_random_state(self.random_state)
        base = self.estimator or LogisticRegression(max_iter=500)
        pos = np.where(s == 1)[0]
        n_hold = max(1, int(self.calibration_fraction * len(pos)))
        hold = rng.permutation(pos)[:n_hold]           # held-out known positives
        train_mask = np.ones(len(s), bool); train_mask[hold] = False
        self.g_ = clone(base).fit(X[train_mask], s[train_mask])
        # c = average g on held-out positives = P(labeled | positive)
        self.c_ = float(self.g_.predict_proba(X[hold])[:, 1].mean())
        self.c_ = max(self.c_, 1e-3)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "g_")
        p = self.g_.predict_proba(check_array(X))[:, 1] / self.c_   # divide out c
        p = np.clip(p, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


__all__ = ["CoTraining", "TriTraining", "PUClassifier"]
