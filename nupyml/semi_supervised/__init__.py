"""Semi-supervised learning: many unlabelled points, few labelled ones.

Labels are expensive (an expert must read every scan) while raw data is nearly
free. Semi-supervised methods use the unlabelled points too -- but only pay off
if the data satisfies an assumption:

* **Cluster assumption** -- points in the same dense region share a label, so the
  decision boundary should pass through sparse space, not through a crowd.
* **Manifold assumption** -- the data lies on a lower-dimensional surface, and
  labels vary smoothly ALONG it. Two points may be far apart in space yet
  adjacent along the surface.

If neither holds, the unlabelled data is noise and these methods do worse than
ignoring it.

THE TWO FAMILIES
----------------
* ``LabelPropagation`` / ``LabelSpreading`` build a graph connecting nearby
  points and let labels diffuse along the edges, like heat spreading through a
  network. Labels flow easily within a dense cluster and barely cross the sparse
  gaps between clusters, which is the cluster assumption made mechanical. They
  differ in whether the given labels are held FIXED (propagation) or may be
  overridden (spreading -- useful when some labels are wrong).
* ``SelfTrainingClassifier`` wraps any classifier: train on what is labelled,
  predict the rest, promote the most confident predictions to labels, repeat.
  Simple and works with any model -- and its failure mode is confirmation bias.
  A confident early mistake becomes a training label and the model trains itself
  deeper into the error, with no mechanism to notice.

Unlabelled points are marked ``-1`` in ``y``.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array


class _BaseLabelPropagation(BaseEstimator, ClassifierMixin):
    """Unlabeled points are marked with -1 in ``y``."""

    def __init__(self, kernel="rbf", gamma=20.0, n_neighbors=7, alpha=None,
                 max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _affinity(self, X):
        if self.kernel == "rbf":
            return np.exp(-self.gamma * cdist(X, X, "sqeuclidean"))
        if self.kernel == "knn":
            n = len(X)
            tree = cKDTree(X)
            _, idx = tree.query(X, k=min(self.n_neighbors + 1, n))
            W = np.zeros((n, n))
            rows = np.repeat(np.arange(n), idx.shape[1] - 1)
            W[rows, idx[:, 1:].ravel()] = 1.0
            return np.maximum(W, W.T)
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        labeled = y != -1
        self.classes_ = np.unique(y[labeled])
        k = len(self.classes_)
        n = len(X)
        Y_static = np.zeros((n, k))
        idx = np.searchsorted(self.classes_, y[labeled])
        Y_static[np.where(labeled)[0], idx] = 1.0

        W = self._affinity(X)
        np.fill_diagonal(W, 0.0)
        T = self._normalize(W)
        F = Y_static.copy()
        alpha = self._alpha()
        for it in range(self.max_iter):
            F_new = T @ F
            if alpha is None:  # propagation: clamp labeled rows hard
                F_new[labeled] = Y_static[labeled]
            else:              # spreading: soft clamping
                F_new = alpha * F_new + (1 - alpha) * Y_static
            if np.abs(F_new - F).max() < self.tol:
                F = F_new
                break
            F = F_new
        # points unreachable from any labeled node keep an all-zero
        # distribution: there is no evidence to normalize
        norm = F.sum(axis=1, keepdims=True)
        self.label_distributions_ = F / np.where(norm > 0, norm, 1.0)
        self.reachable_ = norm.ravel() > 0
        self.transduction_ = self.classes_[np.argmax(F, axis=1)]
        self.n_iter_ = it + 1
        self._X_fit = X
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "label_distributions_")
        X = check_array(X)
        # weight training distributions by affinity to the query points
        if self.kernel == "rbf":
            W = np.exp(-self.gamma * cdist(X, self._X_fit, "sqeuclidean"))
        else:
            tree = cKDTree(self._X_fit)
            _, idx = tree.query(X, k=min(self.n_neighbors, len(self._X_fit)))
            W = np.zeros((len(X), len(self._X_fit)))
            rows = np.repeat(np.arange(len(X)), idx.shape[1])
            W[rows, idx.ravel()] = 1.0
        P = W @ self.label_distributions_
        norm = P.sum(axis=1, keepdims=True)
        return P / np.where(norm > 0, norm, 1.0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LabelPropagation(_BaseLabelPropagation):
    """Hard clamping on labeled points, row-normalized transition matrix."""

    def _alpha(self):
        return None

    @staticmethod
    def _normalize(W):
        d = W.sum(axis=1, keepdims=True)
        return W / np.where(d > 0, d, 1.0)


class LabelSpreading(_BaseLabelPropagation):
    """Soft clamping with the symmetric normalized Laplacian."""

    def __init__(self, kernel="rbf", gamma=20.0, n_neighbors=7, alpha=0.2,
                 max_iter=30, tol=1e-3):
        super().__init__(kernel, gamma, n_neighbors, alpha, max_iter, tol)

    def _alpha(self):
        return self.alpha

    @staticmethod
    def _normalize(W):
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):
    """Iteratively pseudo-label the most confident unlabeled points."""

    def __init__(self, estimator, threshold=0.75, criterion="threshold", k_best=10,
                 max_iter=10):
        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        has_label = y != -1
        self.classes_ = np.unique(y[has_label])
        transduction = y.copy()
        self.labeled_iter_ = np.where(has_label, 0, -1)
        for it in range(1, self.max_iter + 1):
            if has_label.all():
                break
            est = clone(self.estimator).fit(X[has_label], transduction[has_label])
            proba = est.predict_proba(X[~has_label])
            conf = proba.max(axis=1)
            pred = est.classes_[np.argmax(proba, axis=1)]
            if self.criterion == "threshold":
                selected = conf >= self.threshold
            else:  # k_best
                selected = np.zeros(len(conf), dtype=bool)
                selected[np.argsort(-conf)[: self.k_best]] = True
            if not selected.any():
                break
            unlabeled_idx = np.where(~has_label)[0]
            newly = unlabeled_idx[selected]
            transduction[newly] = pred[selected]
            self.labeled_iter_[newly] = it
            has_label[newly] = True
        self.estimator_ = clone(self.estimator).fit(X[has_label],
                                                    transduction[has_label])
        self.transduction_ = transduction
        self.n_iter_ = it
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(check_array(X))


__all__ = ["LabelPropagation", "LabelSpreading", "SelfTrainingClassifier"]
