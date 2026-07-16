"""Randomized and successive-halving search, curves, permutation importance."""
import numpy as np

from ..base import clone
from ..utils import check_random_state
from ._scoring import check_scoring


def _scorer_or_default(scoring):
    scorer = check_scoring(scoring)
    return scorer if scorer is not None else (lambda est, X, y: est.score(X, y))


def _index(X, idx):
    if hasattr(X, "toarray"):
        return X[idx]
    return np.asarray(X)[idx]


class ParameterSampler:
    """Sample parameter dicts; values may be lists or scipy frozen dists."""

    def __init__(self, param_distributions, n_iter, random_state=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        keys = sorted(self.param_distributions)
        for _ in range(self.n_iter):
            params = {}
            for k in keys:
                v = self.param_distributions[k]
                if hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state=rng)
                else:
                    params[k] = v[rng.randint(len(v))]
            yield params

    def __len__(self):
        return self.n_iter


class _BaseSearchCV:
    def __init__(self, estimator, cv=None, scoring=None, refit=True):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def get_params(self, deep=True):
        return dict(self.__dict__)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _evaluate(self, candidates, X, y):
        from . import cross_val_score
        scorer = check_scoring(self.scoring)
        results = {"params": [], "mean_test_score": [], "std_test_score": []}
        best_score, best_params = -np.inf, None
        for params in candidates:
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=scorer)
            mean = float(scores.mean())
            results["params"].append(params)
            results["mean_test_score"].append(mean)
            results["std_test_score"].append(float(scores.std()))
            if mean > best_score:
                best_score, best_params = mean, params
        return results, best_score, best_params

    def _finalize(self, results, best_score, best_params, X, y):
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


class RandomizedSearchCV(_BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, cv=None,
                 scoring=None, refit=True, random_state=None):
        super().__init__(estimator, cv, scoring, refit)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        sampler = ParameterSampler(self.param_distributions, self.n_iter,
                                   self.random_state)
        return self._finalize(*self._evaluate(sampler, X, y), X, y)


class HalvingGridSearchCV(_BaseSearchCV):
    """Successive halving over a full grid, growing the sample budget."""

    def __init__(self, estimator, param_grid, factor=3, min_resources=None,
                 cv=None, scoring=None, refit=True, random_state=None):
        super().__init__(estimator, cv, scoring, refit)
        self.param_grid = param_grid
        self.factor = factor
        self.min_resources = min_resources
        self.random_state = random_state

    def fit(self, X, y=None):
        from . import ParameterGrid
        rng = check_random_state(self.random_state)
        candidates = list(ParameterGrid(self.param_grid))
        n = len(X) if not hasattr(X, "shape") else X.shape[0]
        n_resources = self.min_resources or max(20, n // (self.factor **
                                                          max(1, int(np.log(len(candidates)) / np.log(self.factor)))))
        all_results = {"params": [], "mean_test_score": [], "std_test_score": [],
                       "iter": [], "n_resources": []}
        rounds = 0
        while True:
            n_r = min(n_resources * self.factor ** rounds, n)
            idx = rng.choice(n, size=n_r, replace=False)
            Xs, ys = _index(X, idx), None if y is None else np.asarray(y)[idx]
            results, _, _ = self._evaluate(candidates, Xs, ys)
            order = np.argsort(-np.asarray(results["mean_test_score"]))
            all_results["params"].extend(results["params"])
            all_results["mean_test_score"].extend(results["mean_test_score"])
            all_results["std_test_score"].extend(results["std_test_score"])
            all_results["iter"].extend([rounds] * len(candidates))
            all_results["n_resources"].extend([n_r] * len(candidates))
            if len(candidates) <= 1 or n_r >= n:
                best_i = order[0]
                best_params = results["params"][best_i]
                best_score = results["mean_test_score"][best_i]
                break
            keep = max(1, len(candidates) // self.factor)
            candidates = [results["params"][i] for i in order[:keep]]
            rounds += 1
        return self._finalize(all_results, best_score, best_params, X, y)


def learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
                   cv=None, scoring=None, random_state=None):
    from . import _check_cv
    scorer = _scorer_or_default(scoring)
    is_clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = _check_cv(cv, y, classifier=is_clf)
    y_arr = np.asarray(y)
    sizes_abs = None
    train_scores, test_scores = [], []
    for train, test in cv.split(X, y):
        row_tr, row_te = [], []
        this_sizes = []
        for frac in train_sizes:
            k = int(np.ceil(frac * len(train))) if isinstance(frac, float) \
                else int(frac)
            k = max(2, min(k, len(train)))
            this_sizes.append(k)
            sub = train[:k]
            est = clone(estimator).fit(_index(X, sub), y_arr[sub])
            row_tr.append(scorer(est, _index(X, sub), y_arr[sub]))
            row_te.append(scorer(est, _index(X, test), y_arr[test]))
        sizes_abs = this_sizes
        train_scores.append(row_tr)
        test_scores.append(row_te)
    return (np.array(sizes_abs), np.array(train_scores).T, np.array(test_scores).T)


def validation_curve(estimator, X, y, param_name, param_range, cv=None,
                     scoring=None):
    from . import _check_cv
    scorer = _scorer_or_default(scoring)
    is_clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = _check_cv(cv, y, classifier=is_clf)
    y_arr = np.asarray(y)
    train_scores = np.empty((len(param_range), cv.get_n_splits(X, y)))
    test_scores = np.empty_like(train_scores)
    for i, val in enumerate(param_range):
        for j, (train, test) in enumerate(cv.split(X, y)):
            est = clone(estimator).set_params(**{param_name: val})
            est.fit(_index(X, train), y_arr[train])
            train_scores[i, j] = scorer(est, _index(X, train), y_arr[train])
            test_scores[i, j] = scorer(est, _index(X, test), y_arr[test])
    return train_scores, test_scores


def permutation_importance(estimator, X, y, n_repeats=5, scoring=None,
                           random_state=None):
    rng = check_random_state(random_state)
    scorer = _scorer_or_default(scoring)
    X = np.array(X, dtype=np.float64)
    y = np.asarray(y)
    baseline = scorer(estimator, X, y)
    n_features = X.shape[1]
    importances = np.empty((n_features, n_repeats))
    for j in range(n_features):
        Xp = X.copy()
        for r in range(n_repeats):
            Xp[:, j] = X[rng.permutation(len(X)), j]
            importances[j, r] = baseline - scorer(estimator, Xp, y)
    class Result:
        pass
    res = Result()
    res.importances = importances
    res.importances_mean = importances.mean(axis=1)
    res.importances_std = importances.std(axis=1)
    return res


__all__ = ["ParameterSampler", "RandomizedSearchCV", "HalvingGridSearchCV",
           "learning_curve", "validation_curve", "permutation_importance"]
