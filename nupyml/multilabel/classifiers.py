"""Multi-label classifiers. Y is a binary matrix (n_samples, n_labels)."""
import numpy as np

from ..base import BaseEstimator, clone, check_is_fitted
from ..linear_model import LogisticRegression
from ..utils import check_array, check_random_state


def _default(base):
    return LogisticRegression(max_iter=300) if base is None else clone(base)


class BinaryRelevance(BaseEstimator):
    """One independent binary classifier per label -- the baseline.

    Decompose the multi-label problem into ``n_labels`` separate binary problems
    and solve each with its own copy of the base estimator. Trivial to implement
    and parallelise, and often a hard-to-beat baseline -- but it decides every
    label in ISOLATION, so it cannot use the fact that "sunset" makes "beach"
    more likely. That blind spot is the reason every other method here exists.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, Y):
        X, Y = check_array(X), check_array(Y)
        self.n_labels_ = Y.shape[1]
        # one classifier per label; a label that is constant in training gets a
        # trivial constant predictor rather than a crash
        self.estimators_ = []
        self.constant_ = []
        for j in range(self.n_labels_):
            col = Y[:, j]
            if len(np.unique(col)) < 2:
                self.estimators_.append(None)
                self.constant_.append(col[0])
            else:
                self.estimators_.append(_default(self.estimator).fit(X, col))
                self.constant_.append(None)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        out = np.zeros((len(X), self.n_labels_), dtype=int)
        for j, est in enumerate(self.estimators_):
            out[:, j] = self.constant_[j] if est is None else est.predict(X)
        return out


class ClassifierChain(BaseEstimator):
    """A chain of classifiers, each fed the previous labels' predictions.

    THE ONE IDEA OVER BINARY RELEVANCE
    ----------------------------------
    Order the labels into a chain. Classifier ``j`` is trained on the original
    features PLUS the true values of labels ``0..j-1``, so it can learn
    conditional relationships -- "given this is already tagged beach, is it also
    sunset?". At prediction time the earlier labels' PREDICTIONS are fed forward
    down the chain. This captures label correlations at essentially the cost of
    binary relevance (still one classifier per label), which is why it is the
    usual first upgrade.

    THE CATCHES
    -----------
    Order matters -- a bad ordering puts a hard-to-predict label early and its
    errors propagate down the chain -- and errors DO propagate, since later
    classifiers trust earlier PREDICTIONS. Ensembles of randomly-ordered chains
    (ECC) average those effects away; this is the single-chain building block.

    Read, Pfahringer, Holmes & Frank (2011).
    """

    def __init__(self, estimator=None, order=None, random_state=None):
        self.estimator = estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, Y):
        X, Y = check_array(X), check_array(Y)
        rng = check_random_state(self.random_state)
        n_labels = Y.shape[1]
        self.order_ = (list(self.order) if self.order is not None
                       else list(rng.permutation(n_labels)))
        self.estimators_ = {}
        self.constant_ = {}
        # augment the features with the TRUE earlier labels during training
        augmented = X.copy()
        for j in self.order_:
            col = Y[:, j]
            if len(np.unique(col)) < 2:
                self.estimators_[j] = None
                self.constant_[j] = col[0]
            else:
                self.estimators_[j] = _default(self.estimator).fit(augmented, col)
                self.constant_[j] = None
            # the next link sees this label too
            augmented = np.column_stack([augmented, col])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        out = np.zeros((len(X), len(self.order_)), dtype=int)
        augmented = X.copy()
        for j in self.order_:
            est = self.estimators_[j]
            # at test time we must feed forward PREDICTIONS, not true labels
            pred = (np.full(len(X), self.constant_[j]) if est is None
                    else est.predict(augmented))
            out[:, j] = pred
            augmented = np.column_stack([augmented, pred])
        return out


class LabelPowerset(BaseEstimator):
    """Turn each observed label-SET into a single class; do ordinary multiclass.

    THE IDEA AND ITS TWO FACES
    --------------------------
    Map every distinct combination of labels seen in training to its own class
    label, then train one multiclass classifier. Because a class IS a full label
    set, correlations are captured PERFECTLY -- the model predicts coherent
    combinations, never an impossible mix.

    But the number of classes is the number of distinct label-sets, which can
    explode toward ``2^n_labels``, leaving many classes with a handful of examples
    -- and any combination NOT seen in training can never be predicted. So it is
    excellent when the label sets are few and recurring, and unusable when they
    are many and sparse. ``RAkEL`` exists to keep this method's correlation
    modelling while bounding the blow-up.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, Y):
        from ..linear_model import LogisticRegression
        X, Y = check_array(X), check_array(Y)
        self.n_labels_ = Y.shape[1]
        # each unique row of Y becomes a class id
        self._combos, inverse = np.unique(Y, axis=0, return_inverse=True)
        self._single = len(self._combos) < 2
        if self._single:
            return self
        base = LogisticRegression(max_iter=300) if self.estimator is None \
            else clone(self.estimator)
        self.estimator_ = base.fit(X, inverse)
        return self

    def predict(self, X):
        check_is_fitted(self, "_combos")
        X = check_array(X)
        if self._single:                       # only one label-set ever seen
            return np.tile(self._combos[0], (len(X), 1))
        # map the predicted class id back to its label-set
        class_ids = self.estimator_.predict(X)
        return self._combos[class_ids]


class RAkEL(BaseEstimator):
    """RAndom k-labELsets: an ensemble of small label-powersets that vote.

    THE FIX FOR POWERSET'S BLOW-UP
    ------------------------------
    LabelPowerset over all labels has up to ``2^n`` classes. RAkEL instead trains
    many powersets, each over a small RANDOM subset of ``k`` labels (so each has
    at most ``2^k`` classes, k small). Every label lands in several subsets; to
    predict a label, its powersets vote and a threshold decides. This keeps
    powerset's correlation-awareness (within each small subset) while bounding the
    class count -- the standard way to make label-powerset scale.

    ``labelset_size`` is ``k``; ``n_models`` is how many random subsets to train.

    Tsoumakas, Katakis & Vlahavas (2011).
    """

    def __init__(self, estimator=None, labelset_size=3, n_models=10,
                 threshold=0.5, random_state=None):
        self.estimator = estimator
        self.labelset_size = labelset_size
        self.n_models = n_models
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, Y):
        X, Y = check_array(X), check_array(Y)
        rng = check_random_state(self.random_state)
        self.n_labels_ = Y.shape[1]
        k = min(self.labelset_size, self.n_labels_)
        self.subsets_, self.models_ = [], []
        for _ in range(self.n_models):
            subset = rng.choice(self.n_labels_, size=k, replace=False)
            self.subsets_.append(subset)
            self.models_.append(
                LabelPowerset(self.estimator).fit(X, Y[:, subset]))
        return self

    def predict(self, X):
        check_is_fitted(self, "models_")
        X = check_array(X)
        votes = np.zeros((len(X), self.n_labels_))
        counts = np.zeros(self.n_labels_)
        for subset, model in zip(self.subsets_, self.models_):
            pred = model.predict(X)
            votes[:, subset] += pred
            counts[subset] += 1
        # a label is on if it won a majority of the votes across its subsets
        counts[counts == 0] = 1
        return (votes / counts >= self.threshold).astype(int)


class MLkNN(BaseEstimator):
    """Multi-label k-nearest-neighbours with a Bayesian decision per label.

    THE IDEA
    --------
    For a new point, find its ``k`` nearest neighbours and count, per label, how
    many carry it. Then decide each label by BAYES' rule: combine the PRIOR (how
    common the label is overall) with the LIKELIHOOD (how the count of positive
    neighbours is distributed for points that do vs do not have the label,
    estimated from training). So it is not a bare vote -- it weighs the neighbour
    evidence against the label's base rate, which matters for rare labels a plain
    majority vote would always suppress.

    A lazy method (all the work is at prediction time) and a strong multi-label
    baseline, naturally handling many labels at once.

    Zhang & Zhou (2007).
    """

    def __init__(self, k=10, smoothing=1.0):
        self.k = k
        self.smoothing = smoothing

    def fit(self, X, Y):
        from scipy.spatial.distance import cdist
        X, Y = check_array(X), check_array(Y)
        self.X_ = X
        self.Y_ = Y
        n, n_labels = Y.shape
        s = self.smoothing

        # prior P(label present) with smoothing
        self.prior_ = (s + Y.sum(axis=0)) / (2 * s + n)
        # likelihood: for each label, the distribution over "how many of my k
        # neighbours have this label" given the label is present / absent
        d = cdist(X, X)
        np.fill_diagonal(d, np.inf)
        neigh = np.argsort(d, axis=1)[:, :self.k]
        k = self.k
        self._cond_present = np.zeros((n_labels, k + 1))
        self._cond_absent = np.zeros((n_labels, k + 1))
        for i in range(n):
            counts = Y[neigh[i]].sum(axis=0)       # positives among neighbours
            for j in range(n_labels):
                c = int(counts[j])
                if Y[i, j] == 1:
                    self._cond_present[j, c] += 1
                else:
                    self._cond_absent[j, c] += 1
        self._cond_present = (s + self._cond_present) / \
            (s * (k + 1) + self._cond_present.sum(axis=1, keepdims=True))
        self._cond_absent = (s + self._cond_absent) / \
            (s * (k + 1) + self._cond_absent.sum(axis=1, keepdims=True))
        return self

    def predict(self, X):
        from scipy.spatial.distance import cdist
        check_is_fitted(self, "prior_")
        X = check_array(X)
        d = cdist(X, self.X_)
        neigh = np.argsort(d, axis=1)[:, :self.k]
        n_labels = self.Y_.shape[1]
        out = np.zeros((len(X), n_labels), dtype=int)
        for i in range(len(X)):
            counts = self.Y_[neigh[i]].sum(axis=0).astype(int)
            for j in range(n_labels):
                c = counts[j]
                # posterior odds present vs absent, by Bayes
                p_yes = self.prior_[j] * self._cond_present[j, c]
                p_no = (1 - self.prior_[j]) * self._cond_absent[j, c]
                out[i, j] = int(p_yes >= p_no)
        return out


__all__ = ["BinaryRelevance", "ClassifierChain", "LabelPowerset", "RAkEL",
           "MLkNN"]
