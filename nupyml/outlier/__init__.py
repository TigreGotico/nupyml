"""Outlier and novelty detection."""
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array, check_random_state


class OutlierMixin:
    _estimator_type = "outlier_detector"

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class IsolationForest(BaseEstimator, OutlierMixin):
    """Anomaly score from the average path length in random isolation trees."""

    def __init__(self, n_estimators=100, max_samples="auto", contamination=0.1,
                 max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state

    @staticmethod
    def _average_path_length(n):
        """Expected path length of an unsuccessful BST search over n points."""
        n = np.asarray(n, dtype=np.float64)
        out = np.zeros_like(n)
        mask = n > 2
        # 2*H(n-1) - 2*(n-1)/n, with H the harmonic number
        out[mask] = (2 * (np.log(n[mask] - 1) + np.euler_gamma)
                     - 2 * (n[mask] - 1) / n[mask])
        out[n == 2] = 1.0
        return out

    def _build(self, X, rng, depth, max_depth):
        n, d = X.shape
        if depth >= max_depth or n <= 1:
            return ("leaf", n)
        # split on a random feature at a random value in its observed range
        for _ in range(d):
            j = rng.randint(d)
            lo, hi = X[:, j].min(), X[:, j].max()
            if lo < hi:
                break
        else:
            return ("leaf", n)
        thr = rng.uniform(lo, hi)
        mask = X[:, j] < thr
        if not mask.any() or mask.all():
            return ("leaf", n)
        return ("split", j, thr,
                self._build(X[mask], rng, depth + 1, max_depth),
                self._build(X[~mask], rng, depth + 1, max_depth))

    def _path_length(self, tree, x, depth=0):
        while tree[0] == "split":
            _, j, thr, left, right = tree
            tree = left if x[j] < thr else right
            depth += 1
        return depth + self._average_path_length(np.array([tree[1]]))[0]

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        n_sub = min(256, n) if self.max_samples == "auto" else (
            int(self.max_samples * n) if isinstance(self.max_samples, float)
            else min(int(self.max_samples), n))
        self._n_sub = n_sub
        max_depth = int(np.ceil(np.log2(max(n_sub, 2))))
        n_feat = max(1, int(self.max_features * X.shape[1])) \
            if isinstance(self.max_features, float) else int(self.max_features)
        self.trees_ = []
        self.features_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n_sub, replace=False)
            feats = rng.choice(X.shape[1], size=n_feat, replace=False) \
                if n_feat < X.shape[1] else np.arange(X.shape[1])
            self.trees_.append(self._build(X[idx][:, feats], rng, 0, max_depth))
            self.features_.append(feats)
        scores = self.score_samples(X)
        self.offset_ = np.percentile(scores, 100 * self.contamination)
        return self

    def score_samples(self, X):
        """Higher (closer to 0) is more normal; sklearn's sign convention."""
        check_is_fitted(self, "trees_")
        X = check_array(X)
        depths = np.zeros(len(X))
        for tree, feats in zip(self.trees_, self.features_):
            Xf = X[:, feats]
            depths += [self._path_length(tree, x) for x in Xf]
        avg = depths / len(self.trees_)
        c = self._average_path_length(np.array([self._n_sub]))[0]
        return -(2 ** (-avg / c))

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """-1 for outliers, 1 for inliers."""
        return np.where(self.decision_function(X) < 0, -1, 1)


class LocalOutlierFactor(BaseEstimator, OutlierMixin):
    """Density ratio against the local neighborhood (Breunig et al. 2000)."""

    def __init__(self, n_neighbors=20, contamination=0.1, novelty=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty

    @staticmethod
    def _lrd(dist_k, k_dist_of_neighbors):
        """Local reachability density = 1 / mean reachability distance."""
        reach = np.maximum(dist_k, k_dist_of_neighbors)
        return 1.0 / (reach.mean(axis=1) + 1e-10)

    def fit(self, X, y=None):
        X = check_array(X)
        self._fit_X = X
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        self._k = k
        tree = cKDTree(X)
        dist, idx = tree.query(X, k=k + 1)
        # drop the self-match in column 0
        self._neighbors_dist = dist[:, 1:]
        self._neighbors_idx = idx[:, 1:]
        self._k_distance = self._neighbors_dist[:, -1]
        self._lrd_fit = self._lrd(self._neighbors_dist,
                                  self._k_distance[self._neighbors_idx])
        lof = (self._lrd_fit[self._neighbors_idx].mean(axis=1) / self._lrd_fit)
        self.negative_outlier_factor_ = -lof
        self.offset_ = np.percentile(self.negative_outlier_factor_,
                                     100 * self.contamination)
        self._tree = tree
        return self

    def _score_new(self, X):
        check_is_fitted(self, "_fit_X")
        X = check_array(X)
        dist, idx = self._tree.query(X, k=self._k)
        if self._k == 1:
            dist, idx = dist[:, None], idx[:, None]
        lrd_new = self._lrd(dist, self._k_distance[idx])
        return -(self._lrd_fit[idx].mean(axis=1) / lrd_new)

    def score_samples(self, X):
        if not self.novelty:
            raise AttributeError(
                "score_samples is only available when novelty=True; for "
                "outlier detection on the training set use "
                "negative_outlier_factor_")
        return self._score_new(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X=None):
        if X is None or not self.novelty:
            check_is_fitted(self, "negative_outlier_factor_")
            return np.where(self.negative_outlier_factor_ < self.offset_, -1, 1)
        return np.where(self.decision_function(X) < 0, -1, 1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.negative_outlier_factor_ < self.offset_, -1, 1)


class OneClassSVM(BaseEstimator, OutlierMixin):
    """Schölkopf one-class SVM; the dual is a simplex-constrained QP."""

    def __init__(self, nu=0.5, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 max_iter=1000):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter

    def _kernel_fn(self, X):
        from ..svm import _kernel_fn
        if self.gamma == "scale":
            g = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            g = 1.0 / X.shape[1]
        else:
            g = float(self.gamma)
        return _kernel_fn(self.kernel, g, self.degree, self.coef0)

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        self._kfn = self._kernel_fn(X)
        K = self._kfn(X, X)
        C = 1.0 / (self.nu * n)

        # min 0.5 a'Ka  s.t. 0 <= a <= C, sum(a) = 1
        def obj(a):
            Ka = K @ a
            return 0.5 * a @ Ka, Ka

        res = scipy.optimize.minimize(
            obj, np.full(n, 1.0 / n), jac=True, method="SLSQP",
            bounds=[(0.0, C)] * n,
            constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.0,
                          "jac": lambda a: np.ones_like(a)}],
            options={"maxiter": self.max_iter})
        alpha = np.clip(res.x, 0.0, C)
        sv = alpha > 1e-8
        self._sv_X = X[sv]
        self._sv_alpha = alpha[sv]
        self.support_vectors_ = self._sv_X
        # rho from the margin support vectors (0 < alpha < C)
        margin = sv & (alpha < C - 1e-8)
        scores_all = self._kfn(X, self._sv_X) @ self._sv_alpha
        self.offset_ = -float(np.mean(scores_all[margin])) if margin.any() \
            else -float(np.percentile(scores_all, 100 * self.nu))
        return self

    def score_samples(self, X):
        check_is_fitted(self, "_sv_X")
        X = check_array(X)
        return self._kfn(X, self._sv_X) @ self._sv_alpha

    def decision_function(self, X):
        return self.score_samples(X) + self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)


class EllipticEnvelope(BaseEstimator, OutlierMixin):
    """Gaussian ellipse fit with the Minimum Covariance Determinant estimator."""

    def __init__(self, contamination=0.1, support_fraction=None,
                 n_trials=10, max_iter=30, random_state=None):
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _mahalanobis(X, loc, cov):
        diff = X - loc
        try:
            L = scipy.linalg.cholesky(cov, lower=True)
        except scipy.linalg.LinAlgError:
            cov = cov + 1e-6 * np.eye(len(cov))
            L = scipy.linalg.cholesky(cov, lower=True)
        sol = scipy.linalg.solve_triangular(L, diff.T, lower=True)
        return (sol ** 2).sum(axis=0)

    def _c_step(self, X, support, h):
        """Concentration step: refit on the h points with smallest distance."""
        for _ in range(self.max_iter):
            loc = X[support].mean(axis=0)
            cov = np.cov(X[support].T, bias=False)
            cov = np.atleast_2d(cov) + 1e-9 * np.eye(X.shape[1])
            d = self._mahalanobis(X, loc, cov)
            new_support = np.zeros(len(X), dtype=bool)
            new_support[np.argsort(d)[:h]] = True
            if np.array_equal(new_support, support):
                break
            support = new_support
        det = np.linalg.det(cov)
        return support, loc, cov, det

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        h = int(self.support_fraction * n) if self.support_fraction \
            else (n + d + 1) // 2
        h = max(h, d + 1)
        best = None
        for _ in range(self.n_trials):
            init = np.zeros(n, dtype=bool)
            init[rng.choice(n, size=h, replace=False)] = True
            support, loc, cov, det = self._c_step(X, init, h)
            if best is None or det < best[3]:
                best = (support, loc, cov, det)
        self.support_, self.location_, self.covariance_, _ = best
        self.precision_ = np.linalg.pinv(self.covariance_)
        dist = self.mahalanobis(X)
        self.offset_ = -np.percentile(dist, 100 * (1 - self.contamination))
        return self

    def mahalanobis(self, X):
        check_is_fitted(self, "location_")
        return self._mahalanobis(check_array(X), self.location_, self.covariance_)

    def score_samples(self, X):
        return -self.mahalanobis(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)


__all__ = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM",
           "EllipticEnvelope", "OutlierMixin"]
