"""Model selection: estimating how well a model will do on data it has not seen.

THE ONLY REAL RULE
------------------
Never evaluate on data you fitted on. A model's training error measures memory,
not learning -- a 1-nearest-neighbour classifier scores 100% on its own training
set and may be worthless.

Every tool here is a way of enforcing that rule while wasting as little data as
possible.

WHY CROSS-VALIDATION
--------------------
A single train/test split wastes the test set (never learned from) and gives a
noisy estimate that depends on which rows landed where. K-fold reuses everything:
each fold is held out exactly once, and the k scores both average to a better
estimate and reveal its VARIANCE -- if the folds disagree wildly, no single
number was ever going to be meaningful.

CHOOSING THE SPLITTER IS A MODELLING DECISION
---------------------------------------------
The default assumes rows are independent and interchangeable. When that is
false, plain KFold LEAKS and reports a score you will never reproduce:

* ``StratifiedKFold`` -- keeps class proportions per fold. With a rare class,
  random folds may contain none of it at all.
* ``GroupKFold`` -- keeps a group (patient, user, document) entirely on one
  side. Otherwise two readings from the same patient land on both sides and the
  model is scored on near-copies of what it trained on.
* ``TimeSeriesSplit`` -- always trains on the past and tests on the future.
  Random folds let a model learn from tomorrow to predict today, which no
  deployed model can do.

THE SUBTLER LEAK
----------------
Scaling or selecting features BEFORE splitting leaks the test set's statistics
into training. It is invisible and it inflates scores. Use a ``Pipeline`` and
pass THAT to ``cross_val_score``: every step is then refitted inside each fold,
which is exactly the point of pipelines.

SEARCHING
---------
* ``GridSearchCV`` -- try every combination. Exhaustive, and exponential in the
  number of parameters.
* ``RandomizedSearchCV`` -- sample combinations. Usually better for the same
  budget: most parameters do not matter much, and random sampling tries more
  distinct values of the ones that do.
* ``HalvingGridSearchCV`` -- start every candidate on a little data, keep the
  best, give them more. Spends the budget where it is earning.

Tuning on cross-validation and then reporting that same score is itself
overfitting -- with enough candidates something wins by luck. A held-out set,
untouched until the end, is the only honest final number.
"""
import itertools

import numpy as np

from ..base import clone
from ..utils import check_random_state, column_or_1d


def train_test_split(*arrays, test_size=0.25, train_size=None, random_state=None,
                     shuffle=True, stratify=None):
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("All arrays must have the same length")
    n_test = int(np.ceil(n * test_size)) if isinstance(test_size, float) else int(test_size)
    if train_size is not None:
        n_train = int(np.floor(n * train_size)) if isinstance(train_size, float) else int(train_size)
    else:
        n_train = n - n_test
    rng = check_random_state(random_state)
    if stratify is not None:
        stratify = column_or_1d(stratify)
        test_idx, train_idx = [], []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            if shuffle:
                cls_idx = rng.permutation(cls_idx)
            k = int(round(len(cls_idx) * n_test / n))
            test_idx.append(cls_idx[:k])
            train_idx.append(cls_idx[k:k + int(round(len(cls_idx) * n_train / n))])
        test_idx = np.concatenate(test_idx)
        train_idx = np.concatenate(train_idx)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:n_test + n_train]
    out = []
    for a in arrays:
        a = np.asarray(a) if not hasattr(a, "toarray") else a
        out.append(a[train_idx])
        out.append(a[test_idx])
    return out


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(idx)
        sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        sizes[: n % self.n_splits] += 1
        start = 0
        for size in sizes:
            test = idx[start:start + size]
            train = np.concatenate([idx[:start], idx[start + size:]])
            yield train, test
            start += size

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = column_or_1d(y)
        rng = check_random_state(self.random_state)
        # assign each sample a fold, stratified per class
        folds = np.empty(len(y), dtype=int)
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                cls_idx = rng.permutation(cls_idx)
            folds[cls_idx] = np.arange(len(cls_idx)) % self.n_splits
        for k in range(self.n_splits):
            test = np.where(folds == k)[0]
            train = np.where(folds != k)[0]
            yield train, test

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


class LeaveOneOut:
    def split(self, X, y=None):
        n = len(X)
        for i in range(n):
            yield np.r_[np.arange(i), np.arange(i + 1, n)], np.array([i])

    def get_n_splits(self, X, y=None):
        return len(X)


def _check_cv(cv, y=None, classifier=False):
    if cv is None:
        cv = 5
    if isinstance(cv, int):
        if classifier and y is not None:
            return StratifiedKFold(n_splits=cv)
        return KFold(n_splits=cv)
    return cv


def _index(X, idx):
    if hasattr(X, "toarray"):  # sparse
        return X[idx]
    return np.asarray(X)[idx]


def cross_val_score(estimator, X, y=None, cv=None, scoring=None):
    from ._scoring import check_scoring
    scoring = check_scoring(scoring)
    is_clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = _check_cv(cv, y, classifier=is_clf)
    scores = []
    for train, test in cv.split(X, y):
        est = clone(estimator)
        y_tr = None if y is None else np.asarray(y)[train]
        est.fit(_index(X, train), y_tr)
        if scoring is None:
            scores.append(est.score(_index(X, test), np.asarray(y)[test]))
        else:
            scores.append(scoring(est, _index(X, test), np.asarray(y)[test]))
    return np.array(scores)


def cross_val_predict(estimator, X, y=None, cv=None):
    is_clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = _check_cv(cv, y, classifier=is_clf)
    n = len(X) if not hasattr(X, "shape") else X.shape[0]
    preds = None
    for train, test in cv.split(X, y):
        est = clone(estimator)
        est.fit(_index(X, train), np.asarray(y)[train])
        p = est.predict(_index(X, test))
        if preds is None:
            preds = np.empty((n,) + p.shape[1:], dtype=p.dtype)
        preds[test] = p
    return preds


class ParameterGrid:
    def __init__(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        for grid in self.param_grid:
            keys = sorted(grid)
            for values in itertools.product(*(grid[k] for k in keys)):
                yield dict(zip(keys, values))

    def __len__(self):
        return sum(
            int(np.prod([len(v) for v in grid.values()])) if grid else 1
            for grid in self.param_grid
        )


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=None, scoring=None, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    # BaseEstimator-style params for nesting
    def get_params(self, deep=True):
        return {"estimator": self.estimator, "param_grid": self.param_grid,
                "cv": self.cv, "scoring": self.scoring, "refit": self.refit}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        results = {"params": [], "mean_test_score": [], "std_test_score": []}
        best_score, best_params = -np.inf, None
        for params in ParameterGrid(self.param_grid):
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean = float(scores.mean())
            results["params"].append(params)
            results["mean_test_score"].append(mean)
            results["std_test_score"].append(float(scores.std()))
            if mean > best_score:
                best_score, best_params = mean, params
        self.cv_results_ = {k: (np.array(v) if k != "params" else v)
                            for k, v in results.items()}
        self.best_score_ = best_score
        self.best_params_ = best_params
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**best_params)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


from ._splitters import (  # noqa: E402
    GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, ShuffleSplit,
    StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold,
    LeavePOut, PredefinedSplit,
)
from ._search import (  # noqa: E402
    ParameterSampler, RandomizedSearchCV, HalvingGridSearchCV,
    learning_curve, validation_curve, permutation_importance,
)
from ._scoring import get_scorer, get_scorer_names, check_scoring  # noqa: E402

__all__ = [
    "train_test_split", "KFold", "StratifiedKFold", "LeaveOneOut",
    "cross_val_score", "cross_val_predict", "ParameterGrid", "GridSearchCV",
    "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit", "ShuffleSplit",
    "StratifiedShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold",
    "LeavePOut", "PredefinedSplit",
    "ParameterSampler", "RandomizedSearchCV", "HalvingGridSearchCV",
    "learning_curve", "validation_curve", "permutation_importance",
    "get_scorer", "get_scorer_names", "check_scoring",
]
