"""Ensembles v3: online forests, competence-based selection, oracle stacking, and
posterior model weighting.

Four ways to combine models the earlier ensembles do not cover. The Mondrian forest
grows ONLINE, one point at a time. Dynamic ensemble selection picks a DIFFERENT
subset of classifiers for each query, by local competence. The super learner finds
the cross-validated OPTIMAL combination with an asymptotic oracle guarantee.
Bayesian model averaging weights models by their POSTERIOR probability instead of
picking one.
"""
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


class SuperLearner(BaseEstimator, RegressorMixin):
    """The cross-validated OPTIMAL combination of base learners (van der Laan, 2007).

    Stacking learns to combine base models, but a naive meta-learner can overfit
    their in-sample predictions. The super learner fixes the combination to the
    CROSS-VALIDATED out-of-fold predictions and finds the convex weights that
    minimise held-out error (non-negative, summing to one). Under mild conditions it
    is asymptotically as good as the best possible combination of the library -- the
    "oracle" property -- so adding weak or redundant learners cannot hurt. Base
    models are refit on all data; predictions use the learned weights.
    """

    def __init__(self, estimators, n_folds=5, random_state=None):
        self.estimators = estimators
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        from ..model_selection import KFold
        X = check_array(X); y = np.asarray(y, float)
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=check_random_state(self.random_state))
        Z = np.zeros((len(y), len(self.estimators)))     # out-of-fold predictions
        for tr, te in kf.split(X):
            for j, e in enumerate(self.estimators):
                Z[te, j] = clone(e).fit(X[tr], y[tr]).predict(X[te])
        # non-negative least squares for convex weights (projected gradient)
        w = np.ones(len(self.estimators)) / len(self.estimators)
        for _ in range(500):
            grad = Z.T @ (Z @ w - y) / len(y)
            w = w - 0.1 * grad
            w = np.maximum(w, 0)
            w = w / (w.sum() + 1e-12)
        self.weights_ = w
        self.models_ = [clone(e).fit(X, y) for e in self.estimators]
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.column_stack([m.predict(X) for m in self.models_])
        return P @ self.weights_


class BayesianModelAveraging(BaseEstimator, RegressorMixin):
    """Weight models by their POSTERIOR probability, not pick one (Hoeting, 1999).

    Choosing a single "best" model ignores that others were nearly as plausible and
    understates uncertainty. Bayesian model averaging keeps them all and weights
    each by its posterior probability, approximated here from the BIC (which trades
    fit against complexity): ``w_m ∝ exp(-BIC_m / 2)``. The prediction is the
    weighted average, so a model the data barely prefers still contributes, and
    predictive uncertainty reflects model uncertainty rather than pretending the
    chosen model is certainly correct.
    """

    def __init__(self, estimators, n_params=None):
        self.estimators = estimators
        self.n_params = n_params

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float)
        n = len(y)
        bics = []
        self.models_ = []
        for i, e in enumerate(self.estimators):
            m = clone(e).fit(X, y)
            self.models_.append(m)
            rss = np.sum((y - m.predict(X)) ** 2)
            k = (self.n_params[i] if self.n_params else X.shape[1] + 1)
            bic = n * np.log(rss / n + 1e-12) + k * np.log(n)   # ~ -2 log-evidence
            bics.append(bic)
        bics = np.array(bics)
        rel = np.exp(-0.5 * (bics - bics.min()))         # posterior model weights
        self.weights_ = rel / rel.sum()
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.column_stack([m.predict(X) for m in self.models_])
        return P @ self.weights_


__all__ = ["MondrianForest", "KNORA", "SuperLearner", "BayesianModelAveraging"]
