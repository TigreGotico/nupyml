"""Feature selection: filters, wrappers, and model-based selection."""
import numpy as np
import scipy.stats

from ..base import BaseEstimator, TransformerMixin, check_is_fitted, clone
from ..utils import check_X_y, check_array, column_or_1d


# ---------------------------------------------------------------------------
# univariate scoring functions
# ---------------------------------------------------------------------------

def f_classif(X, y):
    """One-way ANOVA F-value per feature."""
    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n, d = X.shape
    k = len(classes)
    overall_mean = X.mean(axis=0)
    ss_between = np.zeros(d)
    ss_within = np.zeros(d)
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        ss_between += len(Xc) * (mc - overall_mean) ** 2
        ss_within += ((Xc - mc) ** 2).sum(axis=0)
    df_between, df_within = k - 1, n - k
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ss_between / df_between) / np.maximum(ss_within / df_within, 1e-30)
    p = scipy.stats.f.sf(F, df_between, df_within)
    return F, p


def f_regression(X, y):
    X, y = check_X_y(X, y, y_numeric=True)
    n = len(y)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, Xc.T @ yc / denom, 0.0)
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        F = r ** 2 / np.maximum(1 - r ** 2, 1e-30) * df
    p = scipy.stats.f.sf(F, 1, df)
    return F, p


def chi2(X, y):
    """Chi-squared statistic between non-negative features and class labels."""
    X = check_array(X, accept_sparse=True)
    y = column_or_1d(y)
    import scipy.sparse as sp
    Xd = np.asarray(X.todense()) if sp.issparse(X) else X
    if (Xd < 0).any():
        raise ValueError("chi2 requires non-negative features")
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    observed = Y.T @ Xd                                   # (k, d)
    feature_sum = Xd.sum(axis=0)
    class_prob = Y.mean(axis=0)
    expected = np.outer(class_prob, feature_sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    stat = terms.sum(axis=0)
    p = scipy.stats.chi2.sf(stat, len(classes) - 1)
    return stat, p


def _mi_knn_cc(x, y, k=3):
    """Kraskov kNN mutual information between two continuous 1-d variables."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    rng = np.random.RandomState(0)
    x = x + 1e-10 * rng.normal(size=n)
    y = y + 1e-10 * rng.normal(size=n)
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    # chebyshev distance to kth neighbor
    dist, _ = tree.query(xy, k=k + 1, p=np.inf)
    eps = dist[:, -1]
    nx = np.array([len(cKDTree(x[:, None]).query_ball_point([xi], e - 1e-12))
                   for xi, e in zip(x, eps)]) - 1
    ny = np.array([len(cKDTree(y[:, None]).query_ball_point([yi], e - 1e-12))
                   for yi, e in zip(y, eps)]) - 1
    mi = (digamma(k) + digamma(n)
          - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return max(0.0, float(mi))


def _mi_knn_cd(x, y_discrete, k=3):
    """Ross kNN mutual information between continuous x and discrete y."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    rng = np.random.RandomState(0)
    x = x + 1e-10 * rng.normal(size=n)
    full_tree = cKDTree(x[:, None])
    mi_terms = np.empty(n)
    label_counts = {}
    for lbl in np.unique(y_discrete):
        label_counts[lbl] = (y_discrete == lbl).sum()
    for i in range(n):
        same = np.where(y_discrete == y_discrete[i])[0]
        if len(same) <= k:
            mi_terms[i] = 0.0
            continue
        tree = cKDTree(x[same, None])
        dist, _ = tree.query([x[i]], k=k + 1)
        eps = np.atleast_1d(np.squeeze(dist))[-1]
        m = len(full_tree.query_ball_point([x[i]], eps - 1e-12)) - 1
        mi_terms[i] = (digamma(n) - digamma(label_counts[y_discrete[i]])
                       + digamma(k) - digamma(max(m, 1)))
    return max(0.0, float(mi_terms.mean()))


def mutual_info_classif(X, y, n_neighbors=3):
    X, y = check_X_y(X, y)
    return np.array([_mi_knn_cd(X[:, j], y, k=n_neighbors)
                     for j in range(X.shape[1])])


def mutual_info_regression(X, y, n_neighbors=3):
    X, y = check_X_y(X, y, y_numeric=True)
    return np.array([_mi_knn_cc(X[:, j], y, k=n_neighbors)
                     for j in range(X.shape[1])])


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

class _SelectorMixin(TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        """A selector passes names through: it drops columns, never renames."""
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            return np.array([f"x{i}" for i in self.get_support(indices=True)])
        return np.asarray([str(c) for c in input_features])[self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return np.where(self.support_)[0] if indices else self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = check_array(X, accept_sparse=True)
        return X[:, self.get_support(indices=True)]

    def inverse_transform(self, Xt):
        check_is_fitted(self, "support_")
        Xt = np.asarray(Xt)
        out = np.zeros((len(Xt), len(self.support_)))
        out[:, self.support_] = Xt
        return out


class VarianceThreshold(_SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.variances_ = X.var(axis=0)
        self.support_ = self.variances_ > self.threshold
        if not self.support_.any():
            raise ValueError("No feature meets the variance threshold")
        return self


class SelectKBest(_SelectorMixin, BaseEstimator):
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        out = self.score_func(X, y)
        self.scores_, self.pvalues_ = out if isinstance(out, tuple) else (out, None)
        k = min(self.k, len(self.scores_)) if self.k != "all" else len(self.scores_)
        self.support_ = np.zeros(len(self.scores_), dtype=bool)
        scores = np.where(np.isnan(self.scores_), -np.inf, self.scores_)
        self.support_[np.argsort(-scores)[:k]] = True
        return self


class SelectPercentile(_SelectorMixin, BaseEstimator):
    def __init__(self, score_func=f_classif, percentile=10):
        self.score_func = score_func
        self.percentile = percentile

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        out = self.score_func(X, y)
        self.scores_, self.pvalues_ = out if isinstance(out, tuple) else (out, None)
        k = max(1, int(np.ceil(len(self.scores_) * self.percentile / 100)))
        self.support_ = np.zeros(len(self.scores_), dtype=bool)
        scores = np.where(np.isnan(self.scores_), -np.inf, self.scores_)
        self.support_[np.argsort(-scores)[:k]] = True
        return self


class SelectFpr(_SelectorMixin, BaseEstimator):
    def __init__(self, score_func=f_classif, alpha=0.05):
        self.score_func = score_func
        self.alpha = alpha

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        self.scores_, self.pvalues_ = self.score_func(X, y)
        self.support_ = self.pvalues_ < self.alpha
        return self


def _importances(est):
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        return np.abs(coef) if coef.ndim == 1 else np.abs(coef).sum(axis=0)
    raise ValueError("Estimator exposes neither coef_ nor feature_importances_")


class SelectFromModel(_SelectorMixin, BaseEstimator):
    def __init__(self, estimator, threshold=None, max_features=None,
                 prefit=False):
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.prefit = prefit

    def _resolve_threshold(self, importances):
        t = self.threshold
        if t is None:
            t = "mean"
        if isinstance(t, str):
            if "*" in t:
                factor, _, name = t.partition("*")
                base = np.mean(importances) if name.strip() == "mean" \
                    else np.median(importances)
                return float(factor) * base
            return np.mean(importances) if t == "mean" else np.median(importances)
        return float(t)

    def fit(self, X, y=None):
        self.estimator_ = self.estimator if self.prefit \
            else clone(self.estimator).fit(X, y)
        importances = _importances(self.estimator_)
        thr = self._resolve_threshold(importances)
        support = importances >= thr
        if self.max_features is not None and support.sum() > self.max_features:
            keep = np.argsort(-importances)[: self.max_features]
            support = np.zeros_like(support)
            support[keep] = True
        self.support_ = support
        return self


class RFE(_SelectorMixin, BaseEstimator):
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X, y):
        X = check_array(X)
        n_features = X.shape[1]
        target = self.n_features_to_select or n_features // 2
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        while support.sum() > target:
            est = clone(self.estimator).fit(X[:, support], y)
            imp = _importances(est)
            n_remove = min(int(self.step) if self.step >= 1
                           else max(1, int(self.step * support.sum())),
                           support.sum() - target)
            worst_local = np.argsort(imp)[:n_remove]
            idx_global = np.where(support)[0][worst_local]
            support[idx_global] = False
            ranking[~support] += 1
        self.support_ = support
        self.ranking_ = ranking
        self.estimator_ = clone(self.estimator).fit(X[:, support], y)
        self.n_features_ = int(support.sum())
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.transform(X))

    def score(self, X, y):
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(self.transform(X), y)


class RFECV(RFE):
    def __init__(self, estimator, step=1, cv=None, scoring=None,
                 min_features_to_select=1):
        super().__init__(estimator, None, step)
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select

    def fit(self, X, y):
        from ..model_selection import cross_val_score
        X = check_array(X)
        n_features = X.shape[1]
        # evaluate CV score for every candidate size along one elimination path
        support = np.ones(n_features, dtype=bool)
        path = []
        while support.sum() >= self.min_features_to_select:
            score = cross_val_score(clone(self.estimator), X[:, support], y,
                                    cv=self.cv, scoring=self.scoring).mean()
            path.append((support.copy(), score))
            if support.sum() == self.min_features_to_select:
                break
            est = clone(self.estimator).fit(X[:, support], y)
            imp = _importances(est)
            n_remove = min(int(self.step), support.sum()
                           - self.min_features_to_select)
            worst_local = np.argsort(imp)[:max(1, n_remove)]
            support[np.where(support)[0][worst_local]] = False
        best_support, best_score = max(path, key=lambda t: t[1])
        self.support_ = best_support
        self.cv_results_ = {"mean_test_score": np.array([s for _, s in path][::-1])}
        self.n_features_ = int(best_support.sum())
        self.estimator_ = clone(self.estimator).fit(X[:, best_support], y)
        return self


class SequentialFeatureSelector(_SelectorMixin, BaseEstimator):
    def __init__(self, estimator, n_features_to_select=None, direction="forward",
                 cv=None, scoring=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        from ..model_selection import cross_val_score
        X = check_array(X)
        n_features = X.shape[1]
        target = self.n_features_to_select or n_features // 2
        if self.direction == "forward":
            selected = np.zeros(n_features, dtype=bool)
            while selected.sum() < target:
                best_j, best_score = -1, -np.inf
                for j in np.where(~selected)[0]:
                    trial = selected.copy()
                    trial[j] = True
                    score = cross_val_score(clone(self.estimator), X[:, trial],
                                            y, cv=self.cv,
                                            scoring=self.scoring).mean()
                    if score > best_score:
                        best_j, best_score = j, score
                selected[best_j] = True
        else:
            selected = np.ones(n_features, dtype=bool)
            while selected.sum() > target:
                best_j, best_score = -1, -np.inf
                for j in np.where(selected)[0]:
                    trial = selected.copy()
                    trial[j] = False
                    score = cross_val_score(clone(self.estimator), X[:, trial],
                                            y, cv=self.cv,
                                            scoring=self.scoring).mean()
                    if score > best_score:
                        best_j, best_score = j, score
                selected[best_j] = False
        self.support_ = selected
        return self


__all__ = [
    "f_classif", "f_regression", "chi2", "mutual_info_classif",
    "mutual_info_regression", "VarianceThreshold", "SelectKBest",
    "SelectPercentile", "SelectFpr", "SelectFromModel", "RFE", "RFECV",
    "SequentialFeatureSelector",
]
