"""Pick the competent classifiers for EACH query (Ko et al., 2008)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


class KNORA(BaseEstimator, ClassifierMixin):
    """Pick the competent classifiers for EACH query (Ko et al., 2008).

    Static voting uses every classifier everywhere, even where a classifier is
    useless. Dynamic ensemble selection instead chooses, per test point, the
    classifiers competent in its LOCAL region -- found from its ``k`` nearest
    neighbours in a validation set. ``KNORA-Eliminate`` keeps only classifiers that
    get ALL k neighbours right (relaxing k if none qualify); ``KNORA-Union`` keeps
    every classifier weighted by how many neighbours it gets right. Either way the
    committee is tailored to the query, which beats static combination when
    different models specialise in different regions.
    """

    def __init__(self, estimators, k=7, mode="union", random_state=None):
        self.estimators = estimators
        self.k = k
        self.mode = mode
        self.random_state = random_state

    def fit(self, X, y):
        from ..model_selection import train_test_split
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        Xtr, self.Xval_, ytr, self.yval_ = train_test_split(
            X, y, test_size=0.3, random_state=rng)
        self.models_ = [clone(e).fit(Xtr, ytr) for e in self.estimators]
        # each model's correctness on every validation point
        self.correct_ = np.array([m.predict(self.Xval_) == self.yval_
                                  for m in self.models_])   # (n_models, n_val)
        return self

    def predict(self, X):
        from scipy.spatial.distance import cdist
        X = check_array(X)
        D = cdist(X, self.Xval_)
        preds = np.array([m.predict(X) for m in self.models_])   # (n_models, n)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for i in range(len(X)):
            nbrs = np.argsort(D[i])[:self.k]
            hits = self.correct_[:, nbrs].sum(axis=1)            # competence per model
            if self.mode == "eliminate":
                k = self.k
                comp = self.correct_[:, nbrs].all(axis=1)
                while not comp.any() and k > 1:                  # relax until some qualify
                    k -= 1
                    comp = self.correct_[:, nbrs[:k]].all(axis=1)
                weights = comp.astype(float)
            else:                                                # union
                weights = hits.astype(float)
            if weights.sum() == 0:
                weights = np.ones(len(self.models_))
            votes = {}
            for m, w in enumerate(weights):
                votes[preds[m, i]] = votes.get(preds[m, i], 0) + w
            out[i] = max(votes, key=votes.get)
        return out


__all__ = ["KNORA"]
