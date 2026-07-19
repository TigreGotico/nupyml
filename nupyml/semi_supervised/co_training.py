"""Two classifiers on two feature VIEWS teach each other (Blum & Mitchell, 1998)."""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


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


__all__ = ["CoTraining"]
